from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Set

from agentic_hybrid.retrieval.hybrid_rrf import reciprocal_rank_fusion


def _doc_identity(doc: Dict[str, Any]) -> str:
    md = doc.get("metadata", {}) or {}
    return str(md.get("doc_id") or doc.get("doc_id") or doc.get("id") or md.get("id") or "").strip()


def _doc_pairs(doc: Dict[str, Any]) -> List[tuple[str, str]]:
    md = doc.get("metadata", {}) or {}
    pairs: List[tuple[str, str]] = []
    for key, label in (
        ("region", "지원 지역"),
        ("supt_regin", "지원 지역"),
        ("age_limit", "대상 연령"),
        ("biz_trgt_age", "대상 연령"),
        ("startup_period", "창업 기간"),
        ("biz_enyy", "창업 기간"),
        ("field", "지원 분야"),
        ("supt_biz_clsfc", "지원 분야"),
        ("apply_target", "지원 대상"),
        ("aply_trgt", "지원 대상"),
        ("deadline", "마감일"),
        ("pbanc_rcpt_end_dt", "마감일"),
    ):
        value = str(md.get(key) or "").strip()
        if value:
            pairs.append((label, value))
    return pairs


def _question_slots(question: str) -> List[str]:
    q = str(question or "")
    slots: List[str] = []
    checks = (
        ("대상 연령", ["연령", "나이", "몇 살"]),
        ("지원 지역", ["지역", "어디", "소재지"]),
        ("창업 기간", ["창업 기간", "몇 년", "업력"]),
        ("지원 분야", ["분야", "어떤 분야"]),
        ("지원 대상", ["대상", "누구", "어떤 기업"]),
        ("마감일", ["마감", "접수", "언제까지"]),
    )
    for slot, hints in checks:
        if any(hint in q for hint in hints):
            slots.append(slot)
    return slots


def _slot_is_answered(answer: str, slot: str) -> bool:
    text = str(answer or "")
    patterns = {
        "대상 연령": [r"\d+\s*세", r"만\s*\d+", r"연령"],
        "지원 지역": [r"전국|서울|경기|인천|부산|대구|광주|대전|울산|세종|강원|충북|충남|전북|전남|경북|경남|제주"],
        "창업 기간": [r"\d+\s*년", r"미만", r"이내", r"예비창업"],
        "지원 분야": [r"분야", r"사업화", r"교육", r"공간", r"보육", r"멘토링"],
        "지원 대상": [r"대상", r"기업", r"창업자", r"예비창업", r"누구나"],
        "마감일": [r"\d{4}[-./]\d{1,2}[-./]\d{1,2}", r"마감", r"접수기간"],
    }
    return any(re.search(p, text) for p in patterns.get(slot, []))


def _llm_complete(llm: Any, prompt: str, system_prompt: Optional[str] = None) -> str:
    if llm is None:
        return ""
    if hasattr(llm, "complete"):
        return str(llm.complete(prompt, system_prompt=system_prompt)).strip()
    if hasattr(llm, "generate"):
        return str(llm.generate(prompt, system_prompt or "", max_tokens=700)).strip()
    return ""


def _slot_specific_query(question: str, slot: str, answer: str, llm: Any = None) -> str:
    if llm is not None:
        prompt = (
            "질문의 누락된 정보를 보완하기 위한 검색 질의를 1개만 생성하세요.\n"
            "짧고 검색 친화적으로 작성하세요.\n"
            "JSON only: {\"query\": \"...\"}\n\n"
            f"원문 질문: {question}\n"
            f"현재 답변: {answer}\n"
            f"누락 슬롯: {slot}\n"
        )
        raw = _llm_complete(llm, prompt, "Return strict JSON only.")
        try:
            match = re.search(r"\{[\s\S]*\}", raw)
            if match:
                obj = json.loads(match.group(0))
                query = str(obj.get("query") or "").strip()
                if query:
                    return query
        except Exception:
            pass
    return f"{question} {slot}"


def _retrieve_followup_docs(
    query: str,
    excluded_ids: Set[str],
    vector_retriever: Any,
    bm25_retriever: Any,
    cfg: Any,
) -> List[Dict[str, Any]]:
    search_k = max(int(getattr(cfg, "TOP_K_RETRIEVAL", 20) or 20), 20)
    if str(getattr(cfg, "RETRIEVAL_MODE", "HYBRID")).upper() == "VECTOR" or bm25_retriever is None:
        docs = vector_retriever.search(query, search_k)
    else:
        vector_docs = vector_retriever.search(query, search_k)
        bm25_docs = bm25_retriever.search(query, search_k)
        docs = reciprocal_rank_fusion([vector_docs, bm25_docs], k=getattr(cfg, "RRF_K", 60), top_k=search_k)

    filtered = [doc for doc in docs if _doc_identity(doc) and _doc_identity(doc) not in excluded_ids]
    return filtered[: min(search_k, 10)]


def _slot_answer_from_docs(slot: str, docs: List[Dict[str, Any]]) -> str:
    for idx, doc in enumerate(docs[:8], start=1):
        for label, value in _doc_pairs(doc):
            if label == slot:
                return f"{slot}: {value} [{idx}]"
    return ""


def revise_node(
    state: Dict[str, Any],
    llm: Any = None,
    cfg: Any = None,
    vector_retriever: Any = None,
    bm25_retriever: Any = None,
) -> Dict[str, Any]:
    question = str(state.get("question") or "")
    answer = str(state.get("answer") or "").strip()
    docs = list(state.get("final_docs") or state.get("filtered_docs") or state.get("retrieved_docs") or [])
    trace = list(state.get("reasoning_trace") or [])

    if cfg is not None and not bool(getattr(cfg, "USE_REVISE", True)):
        trace.append("revise: skipped by mode flag")
        out = dict(state)
        out["reasoning_trace"] = trace
        out["completeness_check"] = {
            "requested_slots": [],
            "missing_slots": [],
            "status": "disabled",
            "followup_query": "",
        }
        return out

    requested_slots = _question_slots(question)
    if not requested_slots or not answer or not docs:
        trace.append("revise: skipped (no slots or no answer/docs)")
        out = dict(state)
        out["reasoning_trace"] = trace
        out["completeness_check"] = {
            "requested_slots": requested_slots,
            "missing_slots": [],
            "status": "skipped",
            "followup_query": "",
        }
        return out

    missing_slots = [slot for slot in requested_slots if not _slot_is_answered(answer, slot)]
    if not missing_slots:
        trace.append("revise: all requested slots already covered")
        out = dict(state)
        out["reasoning_trace"] = trace
        out["completeness_check"] = {
            "requested_slots": requested_slots,
            "missing_slots": [],
            "status": "complete",
            "followup_query": "",
        }
        return out

    excluded_ids = {_doc_identity(doc) for doc in docs if _doc_identity(doc)}
    followup_query = _slot_specific_query(question, missing_slots[0], answer, llm=llm)
    supplements: List[str] = []
    followup_docs: List[Dict[str, Any]] = []

    if vector_retriever is not None and cfg is not None:
        followup_docs = _retrieve_followup_docs(
            followup_query,
            excluded_ids=excluded_ids,
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            cfg=cfg,
        )
        for slot in missing_slots:
            supplement = _slot_answer_from_docs(slot, followup_docs)
            if supplement:
                supplements.append(f"- {supplement}")

    if supplements:
        answer = f"{answer}\n\n보완 정보:\n" + "\n".join(supplements)
        trace.append(
            f"revise: follow-up retrieval applied once "
            f"(query={followup_query!r}, missing_slots={missing_slots}, new_docs={len(followup_docs)})"
        )
        status = "supplemented_with_followup"
    else:
        trace.append(
            f"revise: follow-up retrieval checked once "
            f"(query={followup_query!r}, missing_slots={missing_slots}, new_docs={len(followup_docs)})"
        )
        status = "checked_with_followup"

    out = dict(state)
    out["answer"] = answer
    out["reasoning_trace"] = trace
    out["completeness_check"] = {
        "requested_slots": requested_slots,
        "missing_slots": missing_slots,
        "status": status,
        "followup_query": followup_query,
        "followup_docs_count": len(followup_docs),
    }
    if followup_docs:
        out["followup_docs"] = followup_docs
    return out
