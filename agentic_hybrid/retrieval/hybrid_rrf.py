"""
retrieval/hybrid_rrf.py — Reciprocal Rank Fusion

Fix applied:
  - doc_id used `doc.get("id") or fallback` which treats integer 0 as falsy.
    Changed to explicit None check so id=0 is preserved as a valid key.
  - fusion_sources stored as set during merge then converted to sorted list at end.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Merge N ranked lists with Reciprocal Rank Fusion.

    RRF score = Σ_i  1 / (k + rank_i)

    Each document is identified by its ``id`` field when present (including
    integer 0), otherwise by a positional key ``"<source_idx>:<rank>"``.
    """
    fused: Dict[str, Dict[str, Any]] = {}

    for source_idx, docs in enumerate(ranked_lists):
        for rank, doc in enumerate(docs, start=1):
            raw_id = doc.get("id")
            # FIX: explicit None check — integer 0 is a valid id
            if raw_id is not None and raw_id != "":
                doc_id = str(raw_id)
            else:
                doc_id = f"{source_idx}:{rank}"

            if doc_id not in fused:
                fused[doc_id] = dict(doc)
                fused[doc_id]["rrf_score"] = 0.0
                fused[doc_id]["_fusion_sources"] = set()

            fused[doc_id]["rrf_score"] += 1.0 / (k + rank)
            fused[doc_id]["_fusion_sources"].add(
                doc.get("retrieval_source", f"source_{source_idx}")
            )

    results: List[Dict[str, Any]] = []
    for item in fused.values():
        item = dict(item)
        item["retrieval_source"] = "hybrid_rrf"
        item["fusion_sources"] = sorted(item.pop("_fusion_sources", set()))
        item["score"] = float(item.get("rrf_score", 0.0))
        results.append(item)

    results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)
    return results[:top_k] if top_k is not None else results