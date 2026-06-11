from __future__ import annotations

from typing import Any, Dict

def rerank_node(state: Dict[str, Any], reranker: Any = None, cfg: Any = None) -> Dict[str, Any]:
    if cfg is None:
        raise ValueError("cfg is required for rerank_node")
    docs = list(state.get("retrieved_docs", []))
    trace = list(state.get("reasoning_trace", []))
    question = str(state.get("question", ""))

    if cfg.USE_RERANKER and reranker is not None and docs:
        try:
            reranked = reranker.rerank(question, docs, cfg.TOP_K_RERANK)
            trace.append(f"rerank: applied cross-encoder ({len(docs)} -> {len(reranked)})")
        except Exception as exc:
            reranked = docs[: cfg.TOP_K_RERANK]
            trace.append(f"rerank: failed ({exc}), fallback to retrieval order")
    else:
        reranked = docs[: cfg.TOP_K_RERANK]
        trace.append("rerank: disabled")

    out = dict(state)
    out["reranked_docs"] = reranked
    out["reasoning_trace"] = trace
    return out
