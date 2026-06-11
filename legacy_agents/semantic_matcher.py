# agents/semantic_matcher.py

from __future__ import annotations
from typing import List, Optional, Dict, Any
from datetime import date
from dateutil import parser
import numpy as np

from legacy_agents.base import AgenticAgent
from models.data import UserProfile, MatchResult
from utils.text import clean_text


class SemanticMatchingAgent(AgenticAgent):
    """
    통합 임베딩 기반 의미 매칭 시스템
    - RAGSystem.embed()를 사용 → SBERT / OpenAI / Simple 모두 지원
    - cosine similarity 기반 의미 점수 계산
    - 자동 threshold
    - soft boost (지역/대상/단계)
    - 중복 제거 (같은 공고 / 같은 링크)
    """

    def __init__(self, rag_system: object, llm_client: Optional[object] = None):
        super().__init__("SemanticMatcher", llm_client)
        self.rag = rag_system  # RAG embed() 사용


    # -------------------------------------------
    # Utility
    # -------------------------------------------
    @staticmethod
    def _normalize(text: str) -> str:
        return clean_text(text or "").replace(" ", "").lower()

    @staticmethod
    def _today() -> date:
        return date.today()

    @staticmethod
    def _parse_date(value: str) -> Optional[date]:
        if not value:
            return None
        try:
            return parser.parse(str(value)).date()
        except Exception:
            return None

    def _is_closed(self, metadata: Dict[str, Any]) -> bool:
        """
        마감된 공고인지 체크
        - status 값이 'N', '마감', '종료' 등인 경우
        - deadline 날짜가 오늘 이전인 경우
        """
        status = str(metadata.get("status", "")).strip()
        if status in ("N", "마감", "종료", "접수마감", "마감완료"):
            return True

        deadline_str = (
            metadata.get("deadline")
            or metadata.get("confmdoc_expr_dt")
            or ""
        )
        d = self._parse_date(deadline_str)
        if d and d < self._today():
            return True

        return False


    # -------------------------------------------
    # 의미 유사도 (RAG embed → numpy)
    # -------------------------------------------
    def _semantic_similarity(self, text1: str, text2: str) -> float:
        try:
            v1 = self.rag.embed(text1)
            v2 = self.rag.embed(text2)

            v1 = v1 / (np.linalg.norm(v1) + 1e-9)
            v2 = v2 / (np.linalg.norm(v2) + 1e-9)

            return float(np.dot(v1, v2))
        except Exception as e:
            print("❌ similarity error:", e)
            return 0.0


    # -------------------------------------------
    # 자동 threshold 계산
    # -------------------------------------------
    def _auto_threshold(self, scores: List[float]) -> float:
        if not scores:
            return 0.0

        arr = np.array(scores, dtype=np.float32)
        mean = float(arr.mean())
        std = float(arr.std())

        thr = mean - 0.35 * std
        return max(thr, 0.05)  # 음수 방지


    # -------------------------------------------
    # Soft boost (지역/대상/단계)
    # -------------------------------------------
    def _profile_soft_boost(self, metadata: Dict[str, Any], profile: UserProfile):
        boost = 1.0
        reasons = []

        # 지역
        user_region = self._normalize(profile.region or "")
        meta_region = self._normalize(
            f"{metadata.get('region','')} {metadata.get('address','')} {metadata.get('supt_regin','')}"
        )

        if user_region and user_region in meta_region:
            boost += 0.15
            reasons.append("활동 지역과 공고 지역이 일치합니다.")

        # 신청 대상
        target_text = self._normalize(
            f"{metadata.get('apply_target','')} {metadata.get('apply_target_desc','')}"
        )
        if profile.target_type and self._normalize(profile.target_type) in target_text:
            boost += 0.10
            reasons.append("신청 대상이 사용자 조건과 잘 맞습니다.")

        # 창업 단계
        startup_period = self._normalize(metadata.get("startup_period", ""))
        if profile.business_stage and self._normalize(profile.business_stage) in startup_period:
            boost += 0.08
            reasons.append("창업 단계가 사용자와 일치합니다.")

        return boost, reasons


    # -------------------------------------------
    # Main Matching
    # -------------------------------------------
    def match(
        self,
        profile: UserProfile,
        top_k: int = 10,
        exclude_closed: bool = False,
        desired_data_types: Optional[List[str]] = None,
    ) -> List[MatchResult]:

        # 기본 검색 쿼리
        query = (
            (profile.business_field or "").strip()
            or getattr(profile, "need", "").strip()
            or "창업 지원사업"
        )

        # 1) RAG 검색
        raw_hits = self.rag.search(
            query=query,
            top_k=top_k * 5,
            filters={"type": {"$in": desired_data_types}} if desired_data_types else None,
        )

        candidates = []
        score_list: List[float] = []

        # 2) semantic similarity 계산
        for hit in raw_hits:
            doc = hit.get("document")
            if not doc:
                continue

            metadata = doc.metadata or {}

            # 마감된 공고 제외 옵션
            if exclude_closed and self._is_closed(metadata):
                continue

            # 제목 + 내용 기반으로 의미 비교
            text_blob = f"{doc.text} {metadata.get('title','')} {metadata.get('desc','')}"

            sc = self._semantic_similarity(query, text_blob)
            candidates.append((doc, metadata, sc))
            score_list.append(sc)

        # 후보가 하나도 없으면 바로 종료
        if not candidates:
            return []

        # 3) threshold 계산
        threshold = self._auto_threshold(score_list)
        print("AUTO THRESHOLD =", threshold)

        raw_results: List[MatchResult] = []

        for doc, metadata, sc in candidates:
            if sc < threshold:
                continue

            # boost 적용
            boost, boost_reasons = self._profile_soft_boost(metadata, profile)
            final_score = sc * boost

            # priority
            if final_score >= 0.70:
                priority = "HIGH"
            elif final_score >= 0.45:
                priority = "MID"
            else:
                priority = "LOW"

            # 접수기간, 링크 등 안전하게 꺼내기
            apply_period = metadata.get("apply_period", "")
            deadline = metadata.get("deadline", "")

            match = MatchResult(
                id=str(doc.id),
                title=metadata.get("title", doc.text[:50]),
                data_type=metadata.get("type", ""),
                region=metadata.get("region", ""),
                field=metadata.get("field", ""),
                deadline=deadline,
                status=str(metadata.get("status", "")),
                apply_period=apply_period,
                priority=priority,
                match_score=round(final_score * 100, 1),
                score=final_score,
                reasons=boost_reasons or ["사용자와 의미적으로 높은 관련성이 있습니다."],
                extra={
                    "semantic_score": sc,
                    "threshold": threshold,
                    "detail_url": metadata.get("detail_url", ""),
                    "apply_url": metadata.get("apply_url", ""),
                    "guide_url": metadata.get("guide_url", ""),
                },
                metadata=metadata,
                warnings=[],
            )

            raw_results.append(match)

        # 4) 중복 제거 (같은 공고/같은 링크는 하나만 남기기)
        unique: Dict[str, MatchResult] = {}
        for m in raw_results:
            md = m.metadata or {}
            dtype = m.data_type

            if dtype == "announcement":
                key = md.get("pbanc_sn") or md.get("detail_url") or m.title
            elif dtype == "business":
                key = md.get("detail_url") or m.title
            else:
                key = f"{dtype}:{m.title}:{md.get('detail_url','')}"

            if key not in unique or m.score > unique[key].score:
                unique[key] = m

        results = list(unique.values())

        # 5) 정렬 후 최종 top_k 반환
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]
