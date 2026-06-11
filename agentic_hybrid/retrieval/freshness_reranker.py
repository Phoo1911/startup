"""
retrieval/freshness_reranker.py — 최신성 가중 재정렬

문제:
  - 크로스인코더 점수만으로 정렬하면 1년 전 만료 공고가 최신 공고보다 상위에 오를 수 있음
  - EDU, CNT, STAT 문서는 reg_dt/mdfcn_dt가 있지만 랭킹에 반영되지 않음

해결:
  score_final = score_semantic * (1 - α) + score_freshness * α
  score_freshness = exp(−days_elapsed / decay_days)

  α (freshness_weight) 기본값 0.15 — 너무 크면 내용 관련성이 눌림

사용 예:
    fr = FreshnessReranker(decay_days=180, freshness_weight=0.15)
    reranked = fr.rerank(docs)
"""

from __future__ import annotations

import math
from datetime import date, datetime
from typing import Any, Dict, List, Optional


_DATE_KEYS_BY_TYPE: Dict[str, List[str]] = {
    "announcement": ["start_date", "deadline"],
    "lecture":      ["mdfcn_dt", "reg_dt"],
    "content":      ["reg_date"],
    "statistical":  ["last_mdfcn_dt", "first_reg_dt"],
    "business":     [],   # 날짜 필드 없음 → 중립 점수
    "space":        [],
    "center":       [],
    "product":      ["confmdoc_isu_dt"],
    "corporate":    ["confmdoc_isu_dt"],
    "institution":  ["fndn_dt"],
}

_PARSE_FMTS = ("%Y-%m-%d", "%Y.%m.%d", "%Y/%m/%d", "%Y%m%d")


def _parse_date(value: Any) -> Optional[date]:
    if not value:
        return None
    s = str(value).strip()[:10]
    for fmt in _PARSE_FMTS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _days_since(d: date, today: date) -> int:
    return max(0, (today - d).days)


class FreshnessReranker:
    """
    기존 크로스인코더 점수(doc["cross_encoder_score"] 또는 doc["score"])와
    최신성 점수를 가중 합산하여 최종 순위를 결정한다.

    Parameters
    ----------
    decay_days       : 신선도가 절반으로 줄어드는 기간 (일). 기본 180일.
    freshness_weight : 최신성 가중치 α ∈ [0, 1]. 기본 0.15.
    """

    def __init__(
        self,
        decay_days: int = 180,
        freshness_weight: float = 0.15,
    ) -> None:
        if not 0.0 <= freshness_weight <= 1.0:
            raise ValueError("freshness_weight must be in [0, 1]")
        self.decay_days = decay_days
        self.freshness_weight = freshness_weight

    def _freshness_score(
        self,
        doc: Dict[str, Any],
        today: date,
    ) -> float:
        """0~1 사이 최신성 점수. 날짜 정보가 없으면 중립값 0.5 반환."""
        meta = doc.get("metadata", {}) or {}
        doc_type = meta.get("type", "")
        date_keys = _DATE_KEYS_BY_TYPE.get(doc_type, [])

        best_date: Optional[date] = None
        for key in date_keys:
            d = _parse_date(meta.get(key))
            if d is None:
                continue
            # 가장 최신 날짜를 기준으로 삼음
            if best_date is None or d > best_date:
                best_date = d

        if best_date is None:
            return 0.5  # 날짜 없음 → 중립

        days = _days_since(best_date, today)
        # 지수 감쇠: exp(−days / decay_days)
        return math.exp(-days / max(self.decay_days, 1))

    def _semantic_score(self, doc: Dict[str, Any]) -> float:
        """크로스인코더 점수 또는 일반 점수를 0~1로 정규화."""
        score = doc.get("cross_encoder_score") or doc.get("rrf_score") or doc.get("score") or 0.0
        return float(score)

    def rerank(
        self,
        docs: List[Dict[str, Any]],
        today: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """
        문서 리스트에 freshness_score와 combined_score를 추가하고
        combined_score 내림차순으로 정렬하여 반환한다.
        """
        today = today or datetime.now().date()

        # 시맨틱 점수를 0~1로 min-max 정규화
        raw_scores = [self._semantic_score(d) for d in docs]
        s_min, s_max = min(raw_scores, default=0.0), max(raw_scores, default=1.0)
        s_range = s_max - s_min if s_max != s_min else 1.0

        scored: List[Dict[str, Any]] = []
        for doc, raw_s in zip(docs, raw_scores):
            doc = dict(doc)
            norm_semantic = (raw_s - s_min) / s_range
            freshness = self._freshness_score(doc, today)

            combined = (
                norm_semantic * (1.0 - self.freshness_weight)
                + freshness * self.freshness_weight
            )
            doc["freshness_score"] = round(freshness, 4)
            doc["combined_score"] = round(combined, 4)
            scored.append(doc)

        scored.sort(key=lambda x: x["combined_score"], reverse=True)
        return scored
