"""
nodes/dedup_node.py — 중복 문서 제거 노드

위치: filter → dedup → generate

중복 발생 원인:
  1. intg_pbanc_yn="Y" 인 ANN 문서들이 같은 BIZ를 다수 참조
     → 동일 사업명(intg_pbanc_biz_nm)을 가진 여러 ANN이 검색됨
  2. BIZ와 ANN이 거의 동일한 내용으로 함께 검색됨
  3. 같은 공고가 벡터/BM25 두 경로에서 모두 검색되어 RRF 후에도 남음

제거 전략 (우선순위 순):
  1. 동일 id → 즉시 제거
  2. ANN 문서: intg_pbanc_biz_nm 동일 → deadline 가장 늦은 것 1개만 유지
  3. 타이틀 유사도 > threshold → 점수 높은 것 유지 (Jaccard 기반, 외부 의존성 없음)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


def _jaccard(a: str, b: str) -> float:
    """두 문자열의 음절 단위 Jaccard 유사도 (0~1)."""
    sa = set(a)
    sb = set(b)
    if not sa and not sb:
        return 1.0
    intersection = len(sa & sb)
    union = len(sa | sb)
    return intersection / union if union > 0 else 0.0


def _score(doc: Dict[str, Any]) -> float:
    """문서의 최선 점수를 반환."""
    for key in ("cross_encoder_score", "combined_score", "rrf_score", "score"):
        v = doc.get(key)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return 0.0


def _deadline_key(doc: Dict[str, Any]) -> str:
    """정렬용 deadline 문자열 (없으면 빈 문자열)."""
    return str((doc.get("metadata") or {}).get("deadline") or "")


def dedup_node(
    state: Dict[str, Any],
    title_sim_threshold: float = 0.85,
    cfg: Any = None,
) -> Dict[str, Any]:
    """
    final_docs / filtered_docs에서 중복 문서를 제거한다.

    Parameters
    ----------
    state                : AgentState
    title_sim_threshold  : 이 값 이상의 타이틀 유사도를 중복으로 판정 (기본 0.85)
    cfg                  : AgenticHybridConfig (현재 미사용, 확장용)
    """
    docs: List[Dict[str, Any]] = list(
        state.get("final_docs") or state.get("filtered_docs") or []
    )
    trace: List[str] = list(state.get("reasoning_trace", []))
    original_count = len(docs)

    if not docs:
        out = dict(state)
        out["reasoning_trace"] = trace
        return out

    # ── Step 1: id 기반 중복 제거 ────────────────────────────────────
    seen_ids: Set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for doc in docs:
        doc_id = str(doc.get("id") or "")
        if doc_id and doc_id in seen_ids:
            continue
        if doc_id:
            seen_ids.add(doc_id)
        deduped.append(doc)

    # ── Step 2: ANN 동일 사업명 중복 — deadline 최신 것 유지 ─────────
    ann_by_biz: Dict[str, List[Dict[str, Any]]] = {}
    non_ann: List[Dict[str, Any]] = []

    for doc in deduped:
        meta = doc.get("metadata", {}) or {}
        if meta.get("type") != "announcement":
            non_ann.append(doc)
            continue
        biz_name = str(meta.get("intg_biz_name") or meta.get("title") or "").strip()
        if not biz_name:
            non_ann.append(doc)
            continue
        ann_by_biz.setdefault(biz_name, []).append(doc)

    ann_kept: List[Dict[str, Any]] = []
    for biz_name, group in ann_by_biz.items():
        if len(group) == 1:
            ann_kept.append(group[0])
            continue
        # deadline 내림차순 → 점수 내림차순
        best = max(group, key=lambda d: (_deadline_key(d), _score(d)))
        ann_kept.append(best)

    deduped = non_ann + ann_kept

    # ── Step 3: 타이틀 유사도 기반 중복 제거 ────────────────────────
    final: List[Dict[str, Any]] = []
    titles_kept: List[str] = []

    for doc in sorted(deduped, key=_score, reverse=True):
        title = str(doc.get("title") or (doc.get("metadata") or {}).get("title") or "")
        is_dup = any(
            _jaccard(title, kept) >= title_sim_threshold
            for kept in titles_kept
        )
        if is_dup:
            continue
        titles_kept.append(title)
        final.append(doc)

    removed = original_count - len(final)
    top_k_final = int(getattr(cfg, "TOP_K_FINAL", len(final) or 0) or 0)
    if top_k_final > 0:
        final = final[:top_k_final]
    trace.append(
        f"dedup: {original_count} → {len(final)} docs (removed {removed} duplicates, top_k_final={top_k_final})"
    )

    out = dict(state)
    out["final_docs"] = final
    out["filtered_docs"] = final
    out["reasoning_trace"] = trace
    return out
