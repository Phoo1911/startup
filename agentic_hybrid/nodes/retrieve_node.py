from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_hybrid.retrieval.hybrid_rrf import reciprocal_rank_fusion


def _filter_doc_types(docs: List[Dict[str, Any]], selected_doc_types: List[str]) -> List[Dict[str, Any]]:
    if not selected_doc_types:
        return docs
    wanted = set(selected_doc_types)
    return [d for d in docs if (d.get("metadata", {}) or {}).get("type") in wanted]


def _candidate_k(cfg: Any, vector_retriever: Any, selected_doc_types: List[str]) -> int:
    base_k = int(cfg.TOP_K_RETRIEVAL)
    if not selected_doc_types:
        return base_k
    try:
        total_docs = len(getattr(vector_retriever, "documents", []) or [])
    except Exception:
        total_docs = base_k * 8
    return min(max(base_k * 8, base_k), max(total_docs, base_k))


def retrieve_node(
    state: Dict[str, Any],
    vector_retriever: Any,
    bm25_retriever: Optional[Any] = None,
    cfg: Any = None,
) -> Dict[str, Any]:
    if cfg is None:
        raise ValueError("cfg is required for retrieve_node")
    query = str(state.get("expanded_query") or state.get("question") or "")
    trace = list(state.get("reasoning_trace", []))
    selected_doc_types: List[str] = list(state.get("selected_doc_types") or [])
    search_k = _candidate_k(cfg, vector_retriever, selected_doc_types)

    if cfg.RETRIEVAL_MODE == "VECTOR" or bm25_retriever is None:
        vector_docs = vector_retriever.search(query, search_k)
        retrieved = _filter_doc_types(vector_docs, selected_doc_types)[: cfg.TOP_K_RETRIEVAL]
        trace.append(
            f"retrieve: VECTOR mode raw={len(vector_docs)}, filtered={len(retrieved)}"
            + (f" (selected={selected_doc_types})" if selected_doc_types else "")
        )
    else:
        vector_docs = vector_retriever.search(query, search_k)
        bm25_docs = bm25_retriever.search(query, search_k)
        if selected_doc_types:
            vector_docs = _filter_doc_types(vector_docs, selected_doc_types)
            bm25_docs = _filter_doc_types(bm25_docs, selected_doc_types)
        retrieved = reciprocal_rank_fusion([vector_docs, bm25_docs], k=cfg.RRF_K, top_k=cfg.TOP_K_RETRIEVAL)
        trace.append(
            f"retrieve: HYBRID mode vector={len(vector_docs)}, bm25={len(bm25_docs)}, fused={len(retrieved)}"
            + (f" (selected={selected_doc_types})" if selected_doc_types else "")
        )

    out = dict(state)
    out["retrieved_docs"] = retrieved
    out["reasoning_trace"] = trace
    return out
