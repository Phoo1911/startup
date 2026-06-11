"""
nodes/filter_node.py — Hybrid filtering: Rule-based (fast, accurate) + LLM (selective)

Policy:
  - Rule-based checks (age, deadline, startup, region): ALWAYS (fast & accurate)
  - LLM evaluation: SELECTIVE (only for ambiguous field/industry matching)
  
Performance target: filter 20docs < 0.5s (not 40s)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional
import re
import logging

from agentic_hybrid.tools.age_tool import has_explicit_age_constraint, passes_age_constraint
from agentic_hybrid.tools.deadline_tool import (
    _extract_period_end_date,
    _parse_date,
    passes_deadline_constraint,
)
from agentic_hybrid.tools.exclusion_tool import passes_exclusion_constraint
from agentic_hybrid.tools.startup_tool import has_explicit_startup_constraint, passes_startup_constraint
from agentic_hybrid.tools.region_tool import passes_region_constraint
from agentic_hybrid.tools.field_tool import passes_field_constraint
from agentic_hybrid.tools.target_type_tool import passes_target_type_constraint


logger = logging.getLogger(__name__)


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unknown", "n/a"}:
        return None
    return text


def _validated_parsed_query(parsed: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(parsed)

    age = out.get("age")
    try:
        age_int = int(age) if age not in (None, "") else None
    except (ValueError, TypeError):
        age_int = None
    out["age"] = age_int if age_int is not None and 0 <= age_int <= 120 else None

    startup_years = out.get("startup_years")
    try:
        startup_float = float(startup_years) if startup_years not in (None, "") else None
    except (ValueError, TypeError):
        startup_float = None
    out["startup_years"] = startup_float if startup_float is not None and 0.0 <= startup_float <= 100.0 else None

    out["industry"] = _clean_text(out.get("industry"))
    out["region"] = _clean_text(out.get("region"))
    out["target_type"] = _clean_text(out.get("target_type"))

    time_preference = _clean_text(out.get("time_preference")) or "open_now"
    out["time_preference"] = (
        time_preference if time_preference in {"open_now", "include_upcoming"} else "open_now"
    )

    specials = out.get("special_conditions") or []
    if not isinstance(specials, list):
        specials = [specials]
    deduped: List[str] = []
    seen = set()
    for item in specials:
        text = _clean_text(item)
        if text and text not in seen:
            seen.add(text)
            deduped.append(text)
    out["special_conditions"] = deduped
    return out


def _doc_identity(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata", {}) or {}
    return str(
        meta.get("doc_id")
        or doc.get("doc_id")
        or doc.get("id")
        or meta.get("id")
        or ""
    )


def _merge_priority_docs(
    primary_docs: List[Dict[str, Any]],
    anchor_docs: List[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    seen = set()
    for doc in list(primary_docs) + list(anchor_docs):
        identity = _doc_identity(doc)
        key = identity or str(id(doc))
        if key in seen:
            continue
        seen.add(key)
        merged.append(doc)
        if limit > 0 and len(merged) >= limit:
            break
    return merged


def _doc_brief(doc: Dict[str, Any], max_chars: int = 1200) -> str:
    text = str(doc.get("text") or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def _doc_title(doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata", {}) or {}
    return str(doc.get("title") or meta.get("title") or meta.get("biz_pbanc_nm") or meta.get("spce_nm") or meta.get("cntr_nm") or "").strip()


def _agent1_candidate_answer(question: str, doc: Dict[str, Any]) -> str:
    meta = doc.get("metadata", {}) or {}
    parts: List[str] = []
    title = _doc_title(doc)
    if title:
        parts.append(title)
    for key in (
        "apply_target",
        "aply_trgt",
        "apply_target_desc",
        "supt_regin",
        "region",
        "age_limit",
        "biz_trgt_age",
        "startup_period",
        "biz_enyy",
        "deadline",
        "pbanc_rcpt_end_dt",
        "field",
        "supt_biz_clsfc",
    ):
        value = str(meta.get(key) or "").strip()
        if value:
            parts.append(value)
    if not parts:
        parts.append(_doc_brief(doc, max_chars=220))
    return " | ".join(parts)[:500]


def _soft_filter_penalty(
    *,
    age_soft_fail: bool,
    startup_ok: bool,
    startup_explicit: bool,
    region_ok: bool,
    field_ok: bool,
    target_type_ok: bool,
) -> tuple[float, List[str]]:
    """Soft penalty for minor mismatches (not hard rejections)."""
    penalty = 0.0
    reasons: List[str] = []
    if age_soft_fail:
        penalty += 1.0
        reasons.append("age")
    if not startup_ok and not startup_explicit:
        penalty += 1.0
        reasons.append("startup")
    if not region_ok:
        penalty += 0.7
        reasons.append("region")
    if not field_ok:
        penalty += 0.7
        reasons.append("field")
    if not target_type_ok:
        penalty += 0.8
        reasons.append("target_type")
    return penalty, reasons


def _agent1_candidate_answer(question: str, doc: Dict[str, Any]) -> str:
    """Extract key metadata for ranking."""
    meta = doc.get("metadata", {}) or {}
    parts: List[str] = []
    title = _doc_title(doc)
    if title:
        parts.append(title)
    for key in (
        "apply_target",
        "aply_trgt",
        "apply_target_desc",
        "supt_regin",
        "region",
        "age_limit",
        "biz_trgt_age",
        "startup_period",
        "biz_enyy",
        "deadline",
        "pbanc_rcpt_end_dt",
        "field",
        "supt_biz_clsfc",
    ):
        value = str(meta.get(key) or "").strip()
        if value:
            parts.append(value)
    if not parts:
        parts.append(_doc_brief(doc, max_chars=220))
    return " | ".join(parts)[:500]


def _heuristic_triplet_score(question: str, doc: Dict[str, Any], candidate_answer: str) -> float:
    """Heuristic relevance scoring (no LLM)."""
    text = " ".join([
        question or "",
        _doc_title(doc),
        str(candidate_answer or ""),
        _doc_brief(doc, max_chars=500),
    ]).lower()
    score = 0.0
    for token in re.findall(r"[0-9a-zA-Z가-힣]+", str(question or "").lower()):
        if len(token) >= 2 and token in text:
            score += 0.12
    for hint in ("연령", "나이", "지역", "대상", "지원", "공간", "센터", "모집", "기간", "분야"):
        if hint in str(question or "") and hint in text:
            score += 0.2
    return score


def filter_node(state: Dict[str, Any], cfg: Any = None, llm: Any = None) -> Dict[str, Any]:
    """
    Hybrid filter: Rule-based hard constraints + soft penalties.
    
    Hard rejections (rule-based, fast):
      - Age mismatch (if explicitly stated in doc)
      - Expired deadline
      - Excluded target type
      - Startup mismatch (if explicitly stated in doc)
    
    Soft penalties (no LLM call, heuristic):
      - Age soft fail (doc has no age constraint, user has one)
      - Startup soft fail
      - Region/field/target_type mismatch
      
    All < 0.5s for 20 documents.
    """
    if cfg is None:
        raise ValueError("cfg is required for filter_node")

    docs: List[Dict[str, Any]] = list(
        state.get("reranked_docs") or state.get("retrieved_docs") or []
    )
    parsed: Dict[str, Any] = _validated_parsed_query(dict(state.get("parsed_query", {})))
    intent: Dict[str, Any] = dict(state.get("intent", {}))
    trace: List[str] = list(state.get("reasoning_trace", []))
    top_k_final = int(getattr(cfg, "TOP_K_FINAL", len(docs) or 0) or 0)
    preserve_n = min(
        max(int(getattr(cfg, "FALLBACK_DOCS_COUNT", 2) or 0), 0),
        max(top_k_final, 0),
    )
    question = str(state.get("question") or "")

    # ── 필터 비활성화 ──────────────────────────────────────────────────
    if not cfg.USE_FILTER:
        final_docs = docs[:top_k_final] if top_k_final > 0 else docs
        trace.append(
            f"filter: disabled (preserve_n={preserve_n}, top_k_final={top_k_final})"
        )
        out = dict(state)
        out["filtered_docs"] = final_docs
        out["final_docs"] = final_docs
        out["doc_answer_triplets"] = []
        out["retrieval_anchor_docs"] = []
        out["reasoning_trace"] = trace
        return out

    # ── 제약조건 추출 ──────────────────────────────────────────────────
    age: Any           = parsed.get("age")
    startup_years: Any = parsed.get("startup_years")
    specials: List[str] = parsed.get("special_conditions", [])
    region: Any        = parsed.get("region")
    industry: Any      = parsed.get("industry")
    target_type: Any   = parsed.get("target_type")
    time_preference: str = str(parsed.get("time_preference") or "open_now")
    effective_time_preference: str = time_preference

    kept: List[Dict[str, Any]] = []
    rejects: Dict[str, int] = {
        "age": 0, "deadline": 0, "startup": 0,
        "exclusion": 0, "region": 0, "field": 0,
        "target_type": 0,
    }
    soft_penalties_by_id: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        metadata: Dict[str, Any] = doc.get("metadata", {}) or {}

        # ── Hard reject: Age ────────────────────────────────────────
        age_ok = passes_age_constraint(age, metadata)
        age_explicit = has_explicit_age_constraint(metadata)
        age_soft_fail = False
        if not age_ok:
            rejects["age"] += 1
            if age_explicit:
                continue  # Hard reject
            age_soft_fail = True  # Soft penalty only

        # ── Hard reject: Deadline ───────────────────────────────────
        if time_preference == "include_upcoming":
            today = datetime.now().date()
            end_d = (
                _parse_date(metadata.get("deadline") or metadata.get("confmdoc_expr_dt"))
                or _extract_period_end_date(metadata)
                or _parse_date(metadata.get("llm_deadline_end_date"))
            )
            if end_d is not None and end_d < today:
                rejects["deadline"] += 1
                continue
        else:
            if not passes_deadline_constraint(metadata):
                rejects["deadline"] += 1
                continue

        # ── Hard reject: Exclusion ──────────────────────────────────
        if not passes_exclusion_constraint(specials, metadata):
            rejects["exclusion"] += 1
            continue

        # ── Hard reject: Startup (explicit only) ────────────────────
        startup_ok = passes_startup_constraint(startup_years, metadata)
        startup_explicit = has_explicit_startup_constraint(metadata)
        if not startup_ok:
            rejects["startup"] += 1
            if startup_explicit:
                continue  # Hard reject
            # Soft penalty applied later

        # ── Soft checks: Region ─────────────────────────────────────
        region_ok = passes_region_constraint(
            region,
            metadata,
            embedding_model_name=getattr(cfg, "EMBEDDING_MODEL_NAME", None),
        )
        if not region_ok:
            rejects["region"] += 1

        # ── Soft checks: Technical field/industry ──────────────

        field_ok = passes_field_constraint(
            industry,
            metadata,
            embedding_model_name=getattr(cfg, "EMBEDDING_MODEL_NAME", None),
        )
        if not field_ok:
            rejects["field"] += 1

        # ── Soft checks: Target type ────────────────────────────────
        target_type_ok = passes_target_type_constraint(
            target_type,
            metadata,
            embedding_model_name=getattr(cfg, "EMBEDDING_MODEL_NAME", None),
        )
        if not target_type_ok:
            rejects["target_type"] += 1

        # Keep document and calculate soft penalty
        kept.append(doc)
        identity = _doc_identity(doc)
        penalty, reasons = _soft_filter_penalty(
            age_soft_fail=age_soft_fail,
            startup_ok=startup_ok,
            startup_explicit=startup_explicit,
            region_ok=region_ok,
            field_ok=field_ok,
            target_type_ok=target_type_ok,
        )
        if identity:
            soft_penalties_by_id[identity] = {
                "penalty": penalty,
                "reasons": reasons,
            }

    # ── Fallback: auto-relax deadline if all rejected ──────────────
    if (
        not kept
        and docs
        and time_preference == "open_now"
        and rejects["deadline"] == len(docs)
    ):
        today = datetime.now().date()
        retry_kept: List[Dict[str, Any]] = []
        for doc in docs:
            metadata = doc.get("metadata", {}) or {}
            end_d = (
                _parse_date(metadata.get("deadline") or metadata.get("confmdoc_expr_dt"))
                or _extract_period_end_date(metadata)
                or _parse_date(metadata.get("llm_deadline_end_date"))
            )
            if end_d is None or end_d >= today:
                retry_kept.append(doc)
        if retry_kept:
            kept = retry_kept
            effective_time_preference = "include_upcoming"
            trace.append(
                "filter: auto-relax timing open_now -> include_upcoming "
                f"(kept={len(kept)}/{len(docs)})"
            )

    # ── Fallback: preserve top-N if all hard rejected ───────────────
    fallback_n = getattr(cfg, "FALLBACK_DOCS_COUNT", 2)
    if not kept and docs:
        today = datetime.now().date()
        fallback_pool: List[Dict[str, Any]] = []
        for doc in docs:
            metadata = doc.get("metadata", {}) or {}
            end_d = (
                _parse_date(metadata.get("deadline") or metadata.get("confmdoc_expr_dt"))
                or _extract_period_end_date(metadata)
                or _parse_date(metadata.get("llm_deadline_end_date"))
            )
            if end_d is not None and end_d < today:
                continue
            if not passes_deadline_constraint(metadata) and end_d is not None:
                continue
            fallback_pool.append(doc)

        if fallback_pool:
            kept = fallback_pool[:fallback_n]
            trace.append(
                f"filter: all {len(docs)} docs hard rejected (rejects={rejects}); "
                f"fallback to top-{len(kept)} deadline-safe docs"
            )
        else:
            trace.append(
                f"filter: all {len(docs)} docs hard rejected (rejects={rejects}); "
                "no deadline-safe fallback docs preserved"
            )
    else:
        trace.append(
            f"filter: rule-based kept={len(kept)} from {len(docs)} | rejects={rejects}"
        )

    primary_final_docs = kept[:top_k_final] if top_k_final > 0 else kept
    retrieval_anchor_docs = kept[:preserve_n] if preserve_n > 0 else []
    final_docs = _merge_priority_docs(primary_final_docs, retrieval_anchor_docs, top_k_final)

    if preserve_n > 0 and retrieval_anchor_docs:
        trace.append(
            f"filter: preserved top-{len(retrieval_anchor_docs)} retrieval anchors "
            f"into final_docs (final={len(final_docs)})"
        )

    # ── Heuristic ranking (no LLM) ──────────────────────────────────
    triplets: List[Dict[str, Any]] = []
    if kept:
        for doc in kept:
            candidate_answer = _agent1_candidate_answer(question, doc)
            score = _heuristic_triplet_score(question, doc, candidate_answer)
            identity = _doc_identity(doc)
            penalty_info = soft_penalties_by_id.get(identity or "", {})
            penalty = float(penalty_info.get("penalty") or 0.0)
            reasons = list(penalty_info.get("reasons") or [])
            
            if penalty > 0:
                score = score - penalty
            
            reason_str = "+".join(reasons) if reasons else "pass"
            triplets.append({
                "doc_id": identity,
                "title": _doc_title(doc),
                "candidate_answer": candidate_answer,
                "judge_verdict": "Yes",
                "judge_score": float(score),
                "judge_reason": f"heuristic; soft_penalties={reason_str}" if reasons else "heuristic",
            })

        # ── Soft penalty statistics ────────────────────────────────
        soft_hits = sum(1 for info in soft_penalties_by_id.values() if float(info.get("penalty") or 0.0) > 0)
        if soft_hits:
            trace.append(
                f"filter: soft penalties applied to {soft_hits} docs "
                f"(age={rejects['age']}, startup={rejects['startup']}, "
                f"region={rejects['region']}, field={rejects['field']}, "
                f"target_type={rejects['target_type']})"
            )

    parsed_out = dict(parsed)
    parsed_out["time_preference_effective"] = effective_time_preference
    
    out = dict(state)
    out["parsed_query"] = parsed_out
    out["filtered_docs"] = kept
    out["final_docs"] = final_docs
    out["doc_answer_triplets"] = triplets
    out["retrieval_anchor_docs"] = retrieval_anchor_docs
    out["reasoning_trace"] = trace
    return out
