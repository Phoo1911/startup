"""
graph.py — 전체 파이프라인 (새 노드 통합 버전)

노드 순서:
  intent_classifier          ← NEW: 의도 분류
  → planner
  → query_expansion
  → retrieve
  → doc_type_router          ← NEW: 타입 기반 재정렬 + geo 삽입
  → rerank
  → freshness_rerank         ← NEW: 최신성 가중 재정렬
  → filter                   ← 기존 + region/field 필터 추가
  → dedup                    ← NEW: 중복 제거
  → generate

CrossDocLinker은 build_agentic_graph() 초기화 시 모든 문서에 대해 실행되어
BIZ 문서의 deadline을 상속하고 검색 결과를 enrichment한다.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from functools import partial
from typing import Any, Optional

from agentic_hybrid.config import AgenticHybridConfig
from agentic_hybrid.nodes.filter_node import filter_node
from agentic_hybrid.nodes.generate_node import generate_node
from agentic_hybrid.nodes.intent_classifier_node import intent_classifier_node
from agentic_hybrid.nodes.planner_node import planner_node
from agentic_hybrid.nodes.query_expansion_node import query_expansion_node
from agentic_hybrid.nodes.rerank_node import rerank_node
from agentic_hybrid.nodes.retrieve_node import retrieve_node
from agentic_hybrid.rerankers.cross_encoder_reranker import CrossEncoderReranker
from agentic_hybrid.retrieval.bm25_retriever import BM25Retriever
from agentic_hybrid.retrieval.cross_doc_linker import CrossDocLinker
from agentic_hybrid.retrieval.freshness_reranker import FreshnessReranker
from agentic_hybrid.retrieval.geo_retriever import GeoRetriever
from agentic_hybrid.retrieval.vector_retriever import VectorRetriever
from agentic_hybrid.tools.deadline_tool import (
    _extract_period_end_date,
    _parse_date,
    passes_deadline_constraint,
)
from agentic_hybrid.state import AgentState

# 새 노드

from agentic_hybrid.nodes.doc_type_router_node import doc_type_router_node
from agentic_hybrid.nodes.dedup_node import dedup_node
from agentic_hybrid.nodes.revise_node import revise_node


def _is_pure_baseline_mode(cfg: AgenticHybridConfig) -> bool:
    return (
        str(getattr(cfg, "RETRIEVAL_MODE", "")).upper() == "VECTOR"
        and not bool(getattr(cfg, "USE_HYDE", True))
        and not bool(getattr(cfg, "USE_RERANKER", True))
        and not bool(getattr(cfg, "USE_FILTER", True))
        and not bool(getattr(cfg, "USE_AGENTIC_PLANNER", True))
        and not bool(getattr(cfg, "USE_DOC_TYPE_ROUTER", True))
        and not bool(getattr(cfg, "USE_CROSS_DOC_ENRICH", True))
        and not bool(getattr(cfg, "USE_REVISE", True))
        and not bool(getattr(cfg, "USE_FRESHNESS_RERANK", True))
        and not bool(getattr(cfg, "USE_DEADLINE_GUARD", True))
    )


def build_agentic_graph(cfg: AgenticHybridConfig, llm: Any):
    try:
        from langgraph.graph import END, StateGraph
    except Exception as exc:
        raise ImportError("LangGraph is required. pip install langgraph") from exc

    # ── 검색기 초기화 ──────────────────────────────────────────────────
    vector_retriever = VectorRetriever(
        cfg.faiss_index_path,
        cfg.docs_pickle_path,
        cfg.EMBEDDING_MODEL_NAME,
    )
    bm25_retriever: Optional[BM25Retriever] = (
        BM25Retriever(cfg.docs_pickle_path) if cfg.RETRIEVAL_MODE == "HYBRID" else None
    )
    reranker: Optional[CrossEncoderReranker] = (
        CrossEncoderReranker(cfg.CROSS_ENCODER_MODEL_NAME) if cfg.USE_RERANKER else None
    )

    # ── 새 모듈 초기화 ─────────────────────────────────────────────────
    all_docs = vector_retriever.documents

    # CrossDocLinker: BIZ deadline 상속 + ANN-BIZ 연결
    linker = CrossDocLinker(all_docs)
    print(f"[CrossDocLinker] {linker.stats}")

    # GeoRetriever: 공간/센터 위치 검색
    geo_retriever = GeoRetriever(all_docs)
    print(f"[GeoRetriever] indexed {geo_retriever.indexed_count} geo docs")

    # FreshnessReranker: 최신성 가중 재정렬
    freshness_reranker = FreshnessReranker(
        decay_days=180,
        freshness_weight=0.15,
    )

    # ── 그래프 빌드 ────────────────────────────────────────────────────
    workflow = StateGraph(AgentState)
    if _is_pure_baseline_mode(cfg):
        workflow.add_node("retrieve", lambda state: retrieve_node(state, vector_retriever, bm25_retriever, cfg))
        workflow.add_node("generate", lambda state: generate_node(state, llm))
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()


  
    workflow.add_node("intent_classifier", lambda state: intent_classifier_node(state, llm=llm, cfg=cfg))
    workflow.add_node("query_expansion", lambda state: query_expansion_node(state, llm, cfg))
    workflow.add_node("retrieve",        lambda state: retrieve_node(state, vector_retriever, bm25_retriever, cfg))

    # doc_type_router: retrieve 후 타입별 재정렬 (Rule-based, 빠름)
    workflow.add_node(
        "doc_type_router",
        lambda state: doc_type_router_node(state, geo_retriever=geo_retriever, cfg=cfg),
    )

    workflow.add_node("rerank", lambda state: rerank_node(state, reranker, cfg))
    workflow.add_node("planner", lambda state: planner_node(state, llm, cfg))

    # freshness_rerank: cross-encoder 후 최신성 보정
    workflow.add_node(
        "freshness_rerank",
        lambda state: _freshness_rerank_node(state, freshness_reranker, cfg=cfg),
    )

    # inherit_deadline: filter 전에 BIZ 문서에 ANN deadline 상속 적용
    workflow.add_node(
        "inherit_deadline",
        lambda state: _inherit_deadline_node(state, linker, cfg=cfg),
    )

    # llm_deadline_review: 규칙으로 판정 불가한 문서만 LLM이 보조 판정
    workflow.add_node(
        "llm_deadline_review",
        lambda state: _llm_deadline_review_node(state, llm=llm, cfg=cfg),
    )

    # filter: 기존 + region/field 추가 (filter_node.py 업데이트 필요)
    workflow.add_node("filter", lambda state: filter_node(state, cfg, llm=llm))

    # dedup: 중복 제거
    workflow.add_node("dedup", lambda state: dedup_node(state, cfg=cfg))

    # cross_doc_enrich: 링크 문서 정보 병합
    workflow.add_node(
        "cross_doc_enrich",
        lambda state: _cross_doc_enrich_node(state, linker, cfg=cfg),
    )

    workflow.add_node(
        "final_policy_gate",
        lambda state: _final_policy_gate_node(state, cfg=cfg),
    )

    workflow.add_node("generate", lambda state: generate_node(state, llm))
    workflow.add_node(
        "revise",
        lambda state: revise_node(
            state,
            llm=llm,
            cfg=cfg,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
        ),
    )

    # ── 엣지 ──────────────────────────────────────────────────────────
    from langgraph.graph import END
    workflow.set_entry_point("intent_classifier")
    workflow.add_edge("intent_classifier",  "query_expansion")
    workflow.add_edge("query_expansion",    "retrieve")
    workflow.add_edge("retrieve",           "doc_type_router")
    workflow.add_edge("doc_type_router",    "rerank")
    workflow.add_edge("rerank",             "planner")
    workflow.add_edge("planner",            "inherit_deadline")
    workflow.add_edge("inherit_deadline",   "llm_deadline_review")
    workflow.add_edge("llm_deadline_review","filter")
    workflow.add_edge("filter",             "freshness_rerank")
    workflow.add_edge("freshness_rerank",   "dedup")
    workflow.add_edge("dedup",              "cross_doc_enrich")
    workflow.add_edge("cross_doc_enrich",   "final_policy_gate")
    workflow.add_edge("final_policy_gate",  "generate")
    workflow.add_edge("generate",           "revise")
    workflow.add_edge("revise",             END)

    return workflow.compile()


# ── 인라인 노드 헬퍼 (별도 파일로 분리 가능) ────────────────────────────

def _freshness_rerank_node(
    state: Dict[str, Any],
    freshness_reranker: FreshnessReranker,
    cfg: Any = None,
) -> Dict[str, Any]:
    """reranked_docs에 최신성 점수를 적용한다."""
    from typing import Dict, Any
    docs = list(
        state.get("filtered_docs")
        or state.get("final_docs")
        or state.get("reranked_docs")
        or state.get("retrieved_docs")
        or []
    )
    trace = list(state.get("reasoning_trace", []))

    use_freshness = bool(getattr(cfg, "USE_FRESHNESS_RERANK", True))
    if docs and use_freshness:
        docs = freshness_reranker.rerank(docs)
        trace.append(
            f"freshness_rerank: applied decay=180d, weight=0.15 on {len(docs)} docs"
        )
    elif docs and not use_freshness:
        trace.append(f"freshness_rerank: skipped by mode flag on {len(docs)} docs")
    else:
        trace.append("freshness_rerank: no docs to rerank")

    out = dict(state)
    out["reranked_docs"] = docs
    out["filtered_docs"] = docs
    out["final_docs"] = docs
    out["reasoning_trace"] = trace
    return out


def _cross_doc_enrich_node(
    state: Dict[str, Any],
    linker: CrossDocLinker,
    cfg: Any = None,
) -> Dict[str, Any]:
    """final_docs에 연결 문서 정보(linked_docs)를 병합한다."""
    from typing import Dict, Any
    docs = list(state.get("final_docs") or state.get("filtered_docs") or [])
    trace = list(state.get("reasoning_trace", []))

    if cfg is not None and not bool(getattr(cfg, "USE_CROSS_DOC_ENRICH", True)):
        trace.append("cross_doc_enrich: skipped by mode flag")
    elif docs:
        docs = linker.enrich_retrieved(docs)
        linked_count = sum(1 for d in docs if "linked_docs" in d)
        trace.append(f"cross_doc_enrich: enriched {linked_count}/{len(docs)} docs with linked info")

    out = dict(state)
    out["final_docs"] = docs
    out["reasoning_trace"] = trace
    return out


def _inherit_deadline_node(
    state: Dict[str, Any],
    linker: CrossDocLinker,
    cfg: Any = None,
) -> Dict[str, Any]:
    """filter 전에 BIZ 문서에 연결된 ANN deadline을 상속한다."""
    from typing import Dict, Any
    docs = list(state.get("reranked_docs") or state.get("retrieved_docs") or [])
    trace = list(state.get("reasoning_trace", []))

    use_deadline_guard = bool(getattr(cfg, "USE_DEADLINE_GUARD", True))
    if not use_deadline_guard:
        trace.append("inherit_deadline: skipped by mode flag")
        out = dict(state)
        out["reasoning_trace"] = trace
        return out

    if not docs:
        trace.append("inherit_deadline: no docs")
        out = dict(state)
        out["reasoning_trace"] = trace
        return out

    before_missing = sum(
        1 for d in docs
        if (d.get("metadata") or {}).get("type") == "business"
        and not str((d.get("metadata") or {}).get("deadline") or "").strip()
    )
    docs = linker.enrich_biz_deadline(docs)
    after_missing = sum(
        1 for d in docs
        if (d.get("metadata") or {}).get("type") == "business"
        and not str((d.get("metadata") or {}).get("deadline") or "").strip()
    )
    inherited = max(before_missing - after_missing, 0)
    trace.append(
        f"inherit_deadline: filled={inherited} business docs "
        f"(missing {before_missing}->{after_missing})"
    )

    out = dict(state)
    out["reranked_docs"] = docs
    out["reasoning_trace"] = trace
    return out


def _llm_complete_graph(llm: Any, prompt: str, system_prompt: str = "") -> str:
    if llm is None:
        return ""
    try:
        if hasattr(llm, "complete"):
            return str(llm.complete(prompt, system_prompt=system_prompt)).strip()
        if hasattr(llm, "generate"):
            return str(llm.generate(prompt, system_prompt, max_tokens=300)).strip()
        if callable(llm):
            return str(llm(prompt)).strip()
    except Exception:
        return ""
    return ""


def _needs_llm_deadline_review(doc: Dict[str, Any]) -> bool:
    meta = doc.get("metadata", {}) or {}
    doc_type = str(meta.get("type") or "")
    if doc_type not in {"business", "announcement"}:
        return False

    # Deterministic signals already available -> skip LLM review.
    if str(meta.get("status") or "").strip():
        return False
    if _parse_date(meta.get("deadline") or meta.get("confmdoc_expr_dt")) is not None:
        return False
    if _extract_period_end_date(meta) is not None:
        return False

    text = str(doc.get("text") or "")
    if not text:
        return False

    # Only review if the text likely contains deadline/application-period clues.
    hints = ("접수기간", "신청기간", "모집기간", "공고기간", "신청기한", "접수마감", "마감", "상시모집")
    return any(h in text for h in hints)


def _policy_deadline_unknown(meta: Dict[str, Any]) -> bool:
    """Policy docs with unknown deadline/open status should be treated conservatively."""
    if str(meta.get("type") or "") not in {"announcement", "business"}:
        return False
    if str(meta.get("status") or "").strip():
        return False
    if _parse_date(meta.get("deadline") or meta.get("confmdoc_expr_dt")) is not None:
        return False
    if _extract_period_end_date(meta) is not None:
        return False
    if _parse_date(meta.get("llm_deadline_end_date")) is not None:
        return False
    if isinstance(meta.get("llm_deadline_is_open"), bool):
        return False
    return True


def _announcement_has_timing_signal(meta: Dict[str, Any]) -> bool:
    return any(
        str(meta.get(key) or "").strip()
        for key in (
            "status",
            "rcrt_prgs_yn",
            "deadline",
            "pbanc_rcpt_end_dt",
            "apply_period",
            "llm_deadline_end_date",
        )
    )


def _llm_deadline_review_node(
    state: Dict[str, Any],
    llm: Any,
    cfg: Any = None,
) -> Dict[str, Any]:
    """규칙 필터로 판정 불가한 문서에 한해 LLM이 마감 여부를 보조 판정한다."""
    from typing import Dict, Any

    docs = list(state.get("reranked_docs") or state.get("retrieved_docs") or [])
    trace = list(state.get("reasoning_trace", []))
    intent = state.get("intent") or {}
    parsed = state.get("parsed_query") or {}
    intent_label = str(intent.get("label") or "general")
    time_preference = str(
        parsed.get("time_preference_effective")
        or parsed.get("time_preference")
        or "open_now"
    )

    if not docs or llm is None:
        trace.append("llm_deadline_review: skipped (no docs or no llm)")
        out = dict(state)
        out["reasoning_trace"] = trace
        return out

    use_deadline_guard = bool(getattr(cfg, "USE_DEADLINE_GUARD", True))
    if not use_deadline_guard:
        trace.append("llm_deadline_review: skipped by mode flag")
        out = dict(state)
        out["reasoning_trace"] = trace
        return out

    if time_preference == "include_upcoming":
        trace.append("llm_deadline_review: skipped (include_upcoming mode)")
        out = dict(state)
        out["reasoning_trace"] = trace
        return out

    policy_intents = {"find_policy", "check_status", "compare"}
    review_all_policy = intent_label in policy_intents
    default_max = int(getattr(cfg, "TOP_K_RERANK", 15)) if cfg else 15
    max_reviews = int(getattr(cfg, "LLM_DEADLINE_REVIEW_MAX_DOCS", default_max)) if cfg else default_max
    reviewed = 0
    closed_marked = 0
    unknown_marked = 0
    open_marked = 0

    out_docs = []
    for doc in docs:
        doc_copy = dict(doc)
        meta = dict(doc_copy.get("metadata", {}) or {})
        doc_copy["metadata"] = meta

        doc_type = str(meta.get("type") or "")
        should_review = False
        if review_all_policy and doc_type in {"announcement", "business"}:
            should_review = True
        elif _needs_llm_deadline_review(doc_copy):
            should_review = True

        if reviewed >= max_reviews or not should_review:
            out_docs.append(doc_copy)
            continue

        reviewed += 1
        title = str(meta.get("title") or doc_copy.get("title") or "").strip()
        text = str(doc_copy.get("text") or "")[:1200]
        prompt = (
            "다음 문서에서 현재 접수/신청이 마감되었는지 판단하세요. "
            "마감 여부가 불명확하면 unknown으로 답하세요. JSON만 출력하세요.\n"
            f"오늘 날짜(기준): {datetime.now().date().isoformat()}\n"
            f"제목: {title}\n"
            f"메타데이터(일부): {json.dumps({k: meta.get(k) for k in ['type','status','deadline','confmdoc_expr_dt','apply_period','start_date']}, ensure_ascii=False)}\n"
            f"문서 일부:\n{text}\n"
            'schema={"is_open": true|false|null, "end_date": "YYYY-MM-DD|null", "evidence": "짧은근거"}'
        )
        raw = _llm_complete_graph(llm, prompt, system_prompt="JSON only.")
        if not raw:
            out_docs.append(doc_copy)
            continue

        cleaned = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
        try:
            payload = json.loads(cleaned)
        except Exception:
            out_docs.append(doc_copy)
            continue

        decision = payload.get("is_open")
        end_date = payload.get("end_date")
        meta["llm_deadline_reviewed"] = True
        if isinstance(decision, bool):
            meta["llm_deadline_is_open"] = decision
            ev = payload.get("evidence")
            if ev not in (None, ""):
                meta["llm_deadline_evidence"] = str(ev)[:200]
            if decision is False:
                closed_marked += 1
            else:
                open_marked += 1
        else:
            meta["llm_deadline_is_open"] = None
            meta["llm_deadline_unknown"] = True
            unknown_marked += 1

        if end_date not in (None, ""):
            parsed_end = _parse_date(end_date)
            if parsed_end is not None:
                meta["llm_deadline_end_date"] = parsed_end.isoformat()

        out_docs.append(doc_copy)

    trace.append(
        "llm_deadline_review: "
        f"intent={intent_label}, reviewed={reviewed}, open={open_marked}, "
        f"closed={closed_marked}, unknown={unknown_marked}, max={max_reviews}"
    )

    out = dict(state)
    out["reranked_docs"] = out_docs
    out["reasoning_trace"] = trace
    return out


def _final_policy_gate_node(state: Dict[str, Any], cfg: Any = None) -> Dict[str, Any]:
    """Generate 직전 정책 문서 최종 게이트: 만료 문서만 제거."""
    from typing import Dict, Any

    docs = list(state.get("final_docs") or state.get("filtered_docs") or [])
    trace = list(state.get("reasoning_trace", []))
    intent = state.get("intent") or {}
    parsed = state.get("parsed_query") or {}
    intent_label = str(intent.get("label") or "general")
    time_preference = str(parsed.get("time_preference") or "open_now")
    use_deadline_guard = bool(getattr(cfg, "USE_DEADLINE_GUARD", True))
    top_k_final = int(getattr(cfg, "TOP_K_FINAL", len(docs) or 0) or 0)

    if not use_deadline_guard:
        trace.append("final_policy_gate: skipped by mode flag")
        out = dict(state)
        out["final_docs"] = docs[:top_k_final] if top_k_final > 0 else docs
        out["filtered_docs"] = docs
        out["reasoning_trace"] = trace
        return out

    if intent_label not in {"find_policy", "check_status", "compare"}:
        trace.append("final_policy_gate: skipped (non-policy intent)")
        out = dict(state)
        out["final_docs"] = docs[:top_k_final] if top_k_final > 0 else docs
        out["reasoning_trace"] = trace
        return out

    kept = []
    dropped = 0
    for doc in docs:
        meta = (doc.get("metadata") or {})
        doc_type = str(meta.get("type") or "")
        if doc_type == "announcement" and not _announcement_has_timing_signal(meta):
            dropped += 1
            continue
        if doc_type not in {"announcement", "business"}:
            kept.append(doc)
            continue
        if time_preference == "include_upcoming":
            # Keep open + upcoming; reject explicit past-deadline docs only.
            today = datetime.now().date()
            end_d = (
                _parse_date(meta.get("deadline") or meta.get("confmdoc_expr_dt"))
                or _extract_period_end_date(meta)
                or _parse_date(meta.get("llm_deadline_end_date"))
            )
            if end_d is not None and end_d < today:
                dropped += 1
                continue
        else:
            if not passes_deadline_constraint(meta):
                dropped += 1
                continue
        kept.append(doc)

    final_docs = kept[:top_k_final] if top_k_final > 0 else kept
    trace.append(
        f"final_policy_gate: kept={len(final_docs)}/{len(docs)} dropped={dropped} top_k_final={top_k_final}"
    )

    out = dict(state)
    out["final_docs"] = final_docs
    out["filtered_docs"] = kept
    out["reasoning_trace"] = trace
    return out
