"""
nodes/doc_type_router_node.py — Rule-based document routing (no per-document LLM)

Fast re-ranking based on:
  - Intent detection from intent_classifier_node
  - Type matching (soft boost, not hard filter)
  - Title lexical overlap
  - Geographic location (when applicable)

Performance: filter 20 docs in < 10ms
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

_TYPE_BOOST = 2.0
_POLICY_TYPE_BOOST = 1.6
_SPACE_TYPE_BOOST = 1.6
_TITLE_MATCH_BOOST = 1.35


def _tokenize(text: str) -> List[str]:
    """Extract tokens for title matching."""
    return re.findall(r"[0-9A-Za-z가-힣]+", str(text or "").lower())


def _is_autorag_passthrough_mode(cfg: Any) -> bool:
    if cfg is None:
        return False
    if not bool(getattr(cfg, "USE_DOC_TYPE_ROUTER", True)):
        return True
    return (
        str(getattr(cfg, "RETRIEVAL_MODE", "")).upper() == "HYBRID"
        and bool(getattr(cfg, "USE_HYDE", False))
        and not bool(getattr(cfg, "USE_RERANKER", True))
        and not bool(getattr(cfg, "USE_FILTER", True))
        and not bool(getattr(cfg, "USE_AGENTIC_PLANNER", True))
        and not bool(getattr(cfg, "USE_FRESHNESS_RERANK", True))
        and not bool(getattr(cfg, "USE_DEADLINE_GUARD", True))
    )


def _boost_score(doc: Dict[str, Any], multiplier: float) -> Dict[str, Any]:
    """Apply score multiplier for type matching."""
    doc = dict(doc)
    for key in ("cross_encoder_score", "rrf_score", "combined_score", "score"):
        if key in doc and doc[key] is not None:
            doc[key] = float(doc[key]) * multiplier
    doc["type_boosted"] = True
    return doc


def _title_lexical_boost(doc: Dict[str, Any], question: str) -> Dict[str, Any]:
    """Boost score if title has strong overlap with query."""
    title = str(doc.get("title") or "").strip()
    if not title or not question:
        return doc

    q_tokens = {tok for tok in _tokenize(question) if len(tok) >= 2}
    t_tokens = {tok for tok in _tokenize(title) if len(tok) >= 2}
    if not q_tokens or not t_tokens:
        return doc

    overlap = len(q_tokens & t_tokens) / max(len(q_tokens), 1)
    q_norm = re.sub(r"\s+", "", question.lower())
    t_norm = re.sub(r"\s+", "", title.lower())
    strong_phrase = len(t_norm) >= 8 and (t_norm in q_norm or q_norm in t_norm)

    if overlap >= 0.5 or strong_phrase:
        boosted = dict(doc)
        factor = _TITLE_MATCH_BOOST + min(overlap, 0.4)
        for key in ("cross_encoder_score", "rrf_score", "combined_score", "score"):
            if key in boosted and boosted[key] is not None:
                boosted[key] = float(boosted[key]) * factor
        boosted["title_lexical_boosted"] = True
        boosted["title_overlap"] = overlap
        return boosted
    return doc


def doc_type_router_node(
    state: Dict[str, Any],
    geo_retriever: Optional[Any] = None,
    cfg: Any = None,
    llm: Any = None,
) -> Dict[str, Any]:
    """
    Rule-based document routing (no per-document LLM calls).
    Fast re-ranking using intent + type matching + title overlap.
    """
    docs: List[Dict[str, Any]] = list(state.get("retrieved_docs", []))
    question: str = str(state.get("question") or "")
    intent: Dict[str, Any] = state.get("intent") or {}
    trace: List[str] = list(state.get("reasoning_trace", []))
    selected_doc_types: List[str] = list(state.get("selected_doc_types") or [])

    target_types: List[str] = list(intent.get("doc_types", []))
    intent_label: str = str(intent.get("label") or "general")
    geo: Optional[str] = intent.get("geo")

    if _is_autorag_passthrough_mode(cfg):
        if cfg is not None and not bool(getattr(cfg, "USE_DOC_TYPE_ROUTER", True)):
            trace.append("doc_type_router: disabled by config")
        else:
            trace.append("doc_type_router: passthrough in autorag mode")
        out = dict(state)
        out["retrieved_docs"] = docs
        out["reasoning_trace"] = trace
        return out

    # UI override: strict type filter
    if selected_doc_types:
        typed_only = [
            d for d in docs
            if (d.get("metadata", {}) or {}).get("type") in selected_doc_types
        ]
        trace.append(
            f"doc_type_router: strict UI filter kept {len(typed_only)}/{len(docs)} docs "
            f"(selected={selected_doc_types})"
        )
        out = dict(state)
        out["retrieved_docs"] = typed_only
        out["reasoning_trace"] = trace
        return out

    # Geo-search: find spatial resources near specified city
    if intent_label == "find_space" and geo_retriever is not None and geo:
        geo_docs = geo_retriever.search_by_city(
            city_name=geo,
            radius_km=5.0,
            doc_types=["space", "center"],
            top_k=5,
        )
        if geo_docs:
            existing_ids = {d.get("id") for d in docs}
            new_geo = [d for d in geo_docs if d.get("id") not in existing_ids]
            docs = new_geo + docs
            trace.append(
                f"doc_type_router: inserted {len(new_geo)} geo docs near '{geo}'"
            )

    if not target_types or intent_label == "general":
        trace.append("doc_type_router: no type filter (general intent)")
        out = dict(state)
        out["retrieved_docs"] = docs
        out["reasoning_trace"] = trace
        return out

    # Soft routing: boost matching types, preserve baseline candidates
    if intent_label == "find_space":
        target_types = ["space", "center"]
        trace.append(
            "doc_type_router: soft routing for find_space "
            "(boost target types, preserve non-matching)"
        )
    elif intent_label in {"find_policy", "check_status"}:
        target_types = list(dict.fromkeys(target_types + ["announcement", "business"]))
        trace.append(
            "doc_type_router: soft routing for policy intent "
            "(boost policy types, preserve non-matching)"
        )

    typed_count = sum(
        1 for d in docs
        if (d.get("metadata", {}) or {}).get("type") in target_types
    )

    boosted: List[Dict[str, Any]] = []
    for doc in docs:
        doc_type = (doc.get("metadata", {}) or {}).get("type", "")
        multiplier = None
        if doc_type in target_types:
            if intent_label in {"find_policy", "check_status"}:
                multiplier = _POLICY_TYPE_BOOST
            elif intent_label == "find_space":
                multiplier = _SPACE_TYPE_BOOST
            else:
                multiplier = _TYPE_BOOST
        current = _boost_score(doc, multiplier) if multiplier is not None else doc
        current = _title_lexical_boost(current, question)
        boosted.append(current)

    def _sort_key(d: Dict[str, Any]) -> float:
        for key in ("cross_encoder_score", "rrf_score", "combined_score", "score"):
            v = d.get(key)
            if v is not None:
                return float(v)
        return 0.0

    boosted.sort(key=_sort_key, reverse=True)

    typed_ratio = typed_count / max(len(docs), 1)
    trace.append(
        f"doc_type_router: target_types={target_types}, "
        f"typed_ratio={typed_ratio:.0%}, boosted={typed_count} docs"
    )
    lexical_count = sum(1 for d in boosted if d.get("title_lexical_boosted"))
    if lexical_count:
        trace.append(f"doc_type_router: lexical boost applied to {lexical_count} docs")

    out = dict(state)
    out["retrieved_docs"] = boosted
    out["reasoning_trace"] = trace
    return out
