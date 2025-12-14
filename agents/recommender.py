# agents/recommender.py
"""
추천 에이전트 (개선 버전)
"""
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime

from agents.base import AgenticAgent
from models.data import UserProfile, MatchResult


class RecommendationAgent(AgenticAgent):
    """추천 생성 에이전트"""

    def __init__(self, llm_client: Optional[object] = None):
        super().__init__("Recommender", llm_client)

    def create_report(
        self,
        matches: List[MatchResult],
        profile: UserProfile,
        llm_summary: Optional[str] = None,
        top_n: int = 10,
    ) -> Dict:
        self.think("최종 리포트 생성", action=f"상위 {top_n}개", confidence=1.0)

        if not matches:
            return {
                "status": "NO_MATCH",
                "message": "매칭 결과 없음",
                "recommendations": [],
                "total_matches": 0,
                "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

        top_matches = matches[:top_n]
        recommendations: List[Dict] = []

        for idx, match in enumerate(top_matches, 1):
            try:
                # 안전하게 속성 가져오기
                raw_data = getattr(match, "raw_data", {}) or {}
                extra = getattr(match, "extra", {}) or {}
                metadata = getattr(match, "metadata", {}) or {}

                # 기관명 정리
                host_org = getattr(match, "host_org", "") or metadata.get("host_org", "")
                if host_org:
                    s = str(host_org).strip()
                    low = s.lower()
                    if low.startswith("cmrczn") or "_tab" in low:
                        host_org = ""

                # 날짜 정보
                apply_period_raw = extra.get("apply_period") or metadata.get("apply_period", "")
                start_date = extra.get("start_date") or metadata.get("start_date", "")
                deadline_raw = getattr(match, "deadline", "") or metadata.get("deadline", "")

                # 최종 접수기간
                if apply_period_raw:
                    deadline_display = apply_period_raw
                elif start_date and deadline_raw:
                    deadline_display = f"{start_date} ~ {deadline_raw}"
                elif deadline_raw:
                    deadline_display = f"~ {deadline_raw}"
                else:
                    deadline_display = "정보 없음"

                # 추천 카드 생성
                rec: Dict = {
                    "rank": idx,
                    "id": getattr(match, "id", f"doc_{idx}"),
                    "title": getattr(match, "title", "제목없음"),
                    "data_type": getattr(match, "data_type", "unknown"),
                    "match_score": round(float(getattr(match, "match_score", 0)), 1),
                    "priority": getattr(match, "priority", "LOW"),
                    "reasons": list(getattr(match, "reasons", [])),
                    "warnings": list(getattr(match, "warnings", [])),
                    "deadline": deadline_raw,
                    "apply_period": deadline_display,
                    "start_date": start_date,
                    "detail_url": (
                        getattr(match, "detail_url", "")
                        or extra.get("detail_url", "")
                        or metadata.get("detail_url", "")
                        or raw_data.get("detl_pg_url", "")
                    ),
                    "apply_url": (
                        getattr(match, "apply_url", "")
                        or extra.get("apply_url", "")
                        or metadata.get("apply_url", "")
                        or raw_data.get("biz_aply_url", "")
                    ),
                    "guide_url": (
                        getattr(match, "guide_url", "")
                        or extra.get("guide_url", "")
                        or metadata.get("guide_url", "")
                        or raw_data.get("biz_gdnc_url", "")
                    ),
                    "summary": (
                        getattr(match, "summary", "")
                        or extra.get("summary", "")
                        or metadata.get("summary", "")
                    ),
                    "field": (
                        getattr(match, "field", "")
                        or extra.get("field", "")
                        or metadata.get("field", "")
                    ),
                    "region": (
                        getattr(match, "region", "")
                        or extra.get("region", "")
                        or metadata.get("region", "")
                    ),
                    "host_org": host_org,
                    "status": (
                        getattr(match, "status", "")
                        or extra.get("status", "")
                        or metadata.get("status", "")
                    ),
                    "apply_target": extra.get("apply_target", "") or metadata.get("apply_target", ""),
                    "startup_period": extra.get("startup_period", "") or metadata.get("startup_period", ""),
                    "age_limit": extra.get("age_limit", "") or metadata.get("age_limit", ""),
                    "contact_no": (
                        extra.get("contact_no", "")
                        or metadata.get("contact_no", "")
                        or raw_data.get("prch_cnpl_no", "")
                    ),
                    "score_detail": {
                    "semantic_score": float(getattr(m, "score", 0.0)),
                    "rerank_bonus": float(getattr(m, "rerank_bonus", 0.0)),
                },
                    "tags": list(getattr(match, "tags", [])),
                }

                # extra 필드도 포함
                rec["extra"] = {
                    "apply_period": deadline_display,
                    "start_date": start_date,
                    "deadline": deadline_raw,
                    "apply_target": rec["apply_target"],
                    "startup_period": rec["startup_period"],
                    "age_limit": rec["age_limit"],
                    "contact_no": rec["contact_no"],
                    "field": rec["field"],
                    "region": rec["region"],
                    "status": rec["status"],
                }

                recommendations.append(rec)
                
            except Exception as e:
                print(f"⚠️  매칭 항목 처리 실패: {e}")
                continue

        # 통계 정보
        by_type = defaultdict(int)
        by_priority = defaultdict(int)
        for m in matches:
            by_type[getattr(m, "data_type", "unknown")] += 1
            by_priority[getattr(m, "priority", "LOW")] += 1

        report = {
            "status": "SUCCESS",
            "profile": profile.to_dict() if profile else {},
            "total_matches": len(matches),
            "by_type": dict(by_type),
            "by_priority": dict(by_priority),
            "recommendations": recommendations,
            "llm_summary": llm_summary or "",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        self.think("리포트 생성 완료", result=f"{len(recommendations)}개", confidence=1.0)
        return report