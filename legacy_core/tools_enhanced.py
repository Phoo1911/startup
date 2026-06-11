# core/tools_enhanced.py
"""
API 파라미터를 적극 활용한 강화된 도구 시스템
- 검색 시 pbanc_rcpt_bgng_dt, pbanc_rcpt_end_dt 등 활용
- 필터링 강화 (나이, 장애인, 지역 등)
- 랭킹 알고리즘 개선
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from legacy_core.tools import Tool, ToolRegistry


# ═══════════════════════════════════════════════
# 🔍 강화된 검색 도구 (API 파라미터 활용)
# ═══════════════════════════════════════════════

def create_advanced_search_tool(rag_system, data_agent) -> Tool:
    """
    API 파라미터를 활용한 고급 검색 도구
    - 날짜 범위 필터
    - 지역 필터
    - 분야 필터
    """
    
    def advanced_search(
        query: str,
        top_k: int = 10,
        data_types: Optional[List[str]] = None,
        date_range_days: Optional[int] = None,
        region: Optional[str] = None,
        field: Optional[str] = None,
    ) -> List[Dict]:
        """
        고급 검색 (API 파라미터 + RAG 결합)
        
        Args:
            query: 검색 쿼리
            top_k: 결과 개수
            data_types: 데이터 타입
            date_range_days: 최근 N일 이내 공고만
            region: 지역 필터
            field: 분야 필터
        """
        print(f"\n[ADVANCED SEARCH] 시작")
        print(f"  - Query: {query}")
        print(f"  - Date range: {date_range_days} days" if date_range_days else "")
        print(f"  - Region: {region}" if region else "")
        print(f"  - Field: {field}" if field else "")

        # 1️⃣ RAG 검색 (의미 기반)
        search_results = rag_system.search(
            query=query,
            top_k=top_k * 3,  # 여유롭게 검색
            filters={"type": {"$in": data_types}} if data_types else None
        )

        documents = []
        
        for result in search_results:
            doc = result.get("document")
            if not doc:
                continue

            meta = doc.metadata or {}
            score = result.get("score", 0)

            # 2️⃣ 날짜 필터 (API의 pbanc_rcpt_end_dt 활용)
            if date_range_days:
                deadline = meta.get("deadline") or meta.get("pbanc_rcpt_end_dt")
                if deadline:
                    try:
                        deadline_clean = deadline.replace("-", "")
                        if len(deadline_clean) >= 8:
                            deadline_date = datetime.strptime(deadline_clean[:8], "%Y%m%d")
                            cutoff = datetime.now() - timedelta(days=date_range_days)
                            
                            if deadline_date < cutoff:
                                continue  # 너무 오래된 공고 제외
                    except:
                        pass

            # 3️⃣ 지역 필터 (API의 supt_regin 활용)
            if region and region != "전국":
                doc_region = meta.get("region") or meta.get("supt_regin") or ""
                if doc_region and doc_region != "전국":
                    if region not in doc_region:
                        continue

            # 4️⃣ 분야 필터 (API의 supt_biz_clsfc 활용)
            if field:
                doc_field = meta.get("field") or meta.get("supt_biz_clsfc") or ""
                content = doc.text.lower()
                
                if field.lower() not in doc_field.lower() and field.lower() not in content:
                    continue

            # 결과 포맷팅
            documents.append({
                "id": doc.id,
                "title": meta.get("title", "제목없음"),
                "type": meta.get("type", "unknown"),
                "content": doc.text[:500],
                "score": float(score),
                "metadata": {
                    "region": meta.get("region") or meta.get("supt_regin") or "",
                    "field": meta.get("field") or meta.get("supt_biz_clsfc") or "",
                    "deadline": meta.get("deadline") or meta.get("pbanc_rcpt_end_dt") or "",
                    "apply_period": meta.get("apply_period") or "",
                    "status": meta.get("status") or meta.get("rcrt_prgs_yn") or "",
                    "apply_target": meta.get("apply_target") or meta.get("aply_trgt") or "",
                    "startup_period": meta.get("startup_period") or meta.get("biz_enyy") or "",
                    "age_limit": meta.get("age_limit") or meta.get("biz_trgt_age") or "",
                    "host_org": meta.get("host_org") or meta.get("pbanc_ntrp_nm") or "",
                    "detail_url": meta.get("detail_url") or meta.get("detl_pg_url") or "",
                    "apply_url": meta.get("apply_url") or meta.get("biz_aply_url") or "",
                    "guide_url": meta.get("guide_url") or meta.get("biz_gdnc_url") or "",
                }
            })

        print(f"[ADVANCED SEARCH] 완료: {len(documents)}개")
        return documents[:top_k]

    return Tool(
        name="advanced_search",
        description="API 파라미터를 활용한 고급 검색 (날짜, 지역, 분야 필터 지원)",
        parameters={
            "query": {"type": "string"},
            "top_k": {"type": "integer", "default": 10},
            "data_types": {"type": "array"},
            "date_range_days": {"type": "integer", "description": "최근 N일 이내"},
            "region": {"type": "string"},
            "field": {"type": "string"},
        },
        function=advanced_search
    )


# ═══════════════════════════════════════════════
# 🎯 다단계 필터링 도구
# ═══════════════════════════════════════════════

def create_multi_stage_filter() -> Tool:
    """
    다단계 필터링 (API의 여러 필드 활용)
    - 1단계: 필수 조건 (마감, 지역)
    - 2단계: 선호 조건 (나이, 대상)
    - 3단계: 가점 조건 (분야 일치)
    """
    
    def multi_stage_filter(
        documents: List[Dict],
        user_profile: Dict[str, Any],
        strict_mode: bool = False
    ) -> List[Dict]:
        """
        다단계 필터링
        
        Args:
            documents: 검색 결과
            user_profile: 사용자 프로필
            strict_mode: True면 모든 조건 충족 필요
        """
        print(f"\n[MULTI-STAGE FILTER] 시작: {len(documents)}개")

        filtered = []
        stats = {
            "stage1_pass": 0,
            "stage2_pass": 0,
            "stage3_pass": 0
        }

        for doc in documents:
            meta = doc.get("metadata", {})
            
            # ━━━ STAGE 1: 필수 조건 ━━━
            stage1_pass = True
            
            # 마감 체크
            status = meta.get("status") or ""
            deadline = meta.get("deadline") or ""
            
            if status in ("N", "마감", "종료"):
                stage1_pass = False
            
            if deadline:
                try:
                    deadline_clean = deadline.replace("-", "")
                    if len(deadline_clean) >= 8:
                        deadline_date = datetime.strptime(deadline_clean[:8], "%Y%m%d").date()
                        if deadline_date < datetime.now().date():
                            stage1_pass = False
                except:
                    pass

            # 지역 체크
            user_region = user_profile.get("region", "")
            doc_region = meta.get("region", "")
            
            if user_region and user_region != "전국" and doc_region:
                if doc_region != "전국" and user_region not in doc_region:
                    if strict_mode:
                        stage1_pass = False

            if not stage1_pass:
                continue

            stats["stage1_pass"] += 1

            # ━━━ STAGE 2: 선호 조건 ━━━
            stage2_score = 0
            stage2_reasons = []

            # 나이 조건
            user_age = user_profile.get("age")
            age_limit = meta.get("age_limit", "")
            
            if user_age and age_limit:
                try:
                    import re
                    numbers = re.findall(r'\d+', age_limit)
                    if numbers:
                        limit = int(numbers[0])
                        if "이하" in age_limit and user_age <= limit:
                            stage2_score += 10
                            stage2_reasons.append(f"나이 조건 충족 ({limit}세 이하)")
                        elif "이상" in age_limit and user_age >= limit:
                            stage2_score += 10
                            stage2_reasons.append(f"나이 조건 충족 ({limit}세 이상)")
                except:
                    pass

            # 대상 유형
            target_type = user_profile.get("target_type", "")
            apply_target = meta.get("apply_target", "")
            
            if target_type and apply_target:
                if target_type in apply_target:
                    stage2_score += 15
                    stage2_reasons.append(f"대상 유형 일치 ({target_type})")

            # 장애인 필터
            is_disabled = user_profile.get("is_disabled", False)
            if not is_disabled:
                if "장애인" in doc.get("title", ""):
                    continue

            stats["stage2_pass"] += 1

            # ━━━ STAGE 3: 가점 조건 ━━━
            stage3_score = 0

            # 분야 일치
            user_field = user_profile.get("business_field", "")
            doc_field = meta.get("field", "")
            
            if user_field and doc_field:
                if user_field.lower() in doc_field.lower():
                    stage3_score += 20
                    stage2_reasons.append(f"분야 일치 ({user_field})")

            # 창업 단계 일치
            user_stage = user_profile.get("business_stage", "")
            startup_period = meta.get("startup_period", "")
            
            if user_stage and startup_period:
                if user_stage in startup_period:
                    stage3_score += 10
                    stage2_reasons.append(f"단계 일치 ({user_stage})")

            stats["stage3_pass"] += 1

            # 최종 스코어 계산
            doc["filter_score"] = stage2_score + stage3_score
            doc["filter_reasons"] = stage2_reasons
            filtered.append(doc)

        # 점수순 정렬
        filtered.sort(key=lambda x: x.get("filter_score", 0), reverse=True)

        print(f"[MULTI-STAGE FILTER] 완료")
        print(f"  - Stage 1 통과: {stats['stage1_pass']}개")
        print(f"  - Stage 2 통과: {stats['stage2_pass']}개")
        print(f"  - Stage 3 통과: {stats['stage3_pass']}개")
        print(f"  - 최종: {len(filtered)}개")

        return filtered

    return Tool(
        name="multi_stage_filter",
        description="다단계 필터링 (필수조건 → 선호조건 → 가점조건)",
        parameters={
            "documents": {"type": "array"},
            "user_profile": {"type": "object"},
            "strict_mode": {"type": "boolean", "default": False}
        },
        function=multi_stage_filter
    )


# ═══════════════════════════════════════════════
# 🏆 ML 기반 랭킹 도구
# ═══════════════════════════════════════════════

def create_ml_ranking_tool(embedding_fn) -> Tool:
    """
    ML 기반 랭킹 (임베딩 유사도 + 휴리스틱)
    """
    
    def ml_ranking(
        documents: List[Dict],
        user_profile: Dict[str, Any],
        weights: Optional[Dict[str, float]] = None
    ) -> List[Dict]:
        """
        ML 기반 랭킹
        
        Args:
            documents: 문서 리스트
            user_profile: 사용자 프로필
            weights: 가중치 {"semantic": 0.4, "deadline": 0.3, "match": 0.3}
        """
        print(f"\n[ML RANKING] 시작: {len(documents)}개")

        if not documents:
            return []

        # 기본 가중치
        weights = weights or {
            "semantic": 0.4,  # 의미 유사도
            "deadline": 0.3,  # 마감 임박도
            "match": 0.3      # 조건 매칭
        }

        # 사용자 쿼리 생성
        query_parts = [
            user_profile.get("region", ""),
            user_profile.get("business_field", ""),
            user_profile.get("business_stage", "")
        ]
        user_query = " ".join([p for p in query_parts if p])

        try:
            import numpy as np
            
            # 사용자 쿼리 임베딩
            query_emb = embedding_fn([user_query])[0]

            for doc in documents:
                # ━━━ 1️⃣ 의미 유사도 ━━━
                content = doc.get("content", "")
                title = doc.get("title", "")
                text = f"{title} {content}"
                
                doc_emb = embedding_fn([text[:500]])[0]
                
                # 코사인 유사도
                similarity = np.dot(query_emb, doc_emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_emb) + 1e-9
                )
                semantic_score = float(similarity)

                # ━━━ 2️⃣ 마감 임박도 ━━━
                deadline_score = 0.0
                deadline = doc.get("metadata", {}).get("deadline", "")
                
                if deadline:
                    try:
                        deadline_clean = deadline.replace("-", "")
                        if len(deadline_clean) >= 8:
                            deadline_date = datetime.strptime(deadline_clean[:8], "%Y%m%d").date()
                            days_left = (deadline_date - datetime.now().date()).days
                            
                            if 0 <= days_left <= 7:
                                deadline_score = 1.0  # 매우 임박
                            elif 0 <= days_left <= 14:
                                deadline_score = 0.7
                            elif 0 <= days_left <= 30:
                                deadline_score = 0.5
                            else:
                                deadline_score = 0.3
                    except:
                        pass

                # ━━━ 3️⃣ 조건 매칭 ━━━
                match_score = doc.get("filter_score", 0) / 100.0  # 0~1로 정규화

                # ━━━ 최종 점수 ━━━
                final_score = (
                    semantic_score * weights["semantic"] +
                    deadline_score * weights["deadline"] +
                    match_score * weights["match"]
                )

                doc["ml_score"] = final_score
                doc["score_breakdown"] = {
                    "semantic": semantic_score,
                    "deadline": deadline_score,
                    "match": match_score
                }

        except Exception as e:
            print(f"⚠️ ML 랭킹 오류: {e}")
            # 폴백: filter_score만 사용
            for doc in documents:
                doc["ml_score"] = doc.get("filter_score", 0) / 100.0

        # 정렬
        documents.sort(key=lambda x: x.get("ml_score", 0), reverse=True)

        # 순위 부여
        for idx, doc in enumerate(documents, 1):
            doc["rank"] = idx

        print(f"[ML RANKING] 완료: {len(documents)}개")
        return documents

    return Tool(
        name="ml_ranking",
        description="ML 기반 랭킹 (의미 유사도 + 마감 임박도 + 조건 매칭)",
        parameters={
            "documents": {"type": "array"},
            "user_profile": {"type": "object"},
            "weights": {"type": "object", "optional": True}
        },
        function=ml_ranking
    )


# ═══════════════════════════════════════════════
# 📊 분석 도구
# ═══════════════════════════════════════════════

def create_analytics_tool() -> Tool:
    """결과 분석 도구"""
    
    def analyze_results(
        documents: List[Dict],
        user_profile: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        추천 결과 분석
        - 지역별 분포
        - 분야별 분포
        - 마감일 분포
        - 평균 점수
        """
        from collections import Counter

        if not documents:
            return {"error": "문서 없음"}

        # 통계 수집
        regions = []
        fields = []
        deadlines = []
        scores = []

        for doc in documents:
            meta = doc.get("metadata", {})
            
            region = meta.get("region")
            if region:
                regions.append(region)
            
            field = meta.get("field")
            if field:
                fields.append(field)
            
            deadline = meta.get("deadline")
            if deadline:
                deadlines.append(deadline)
            
            score = doc.get("ml_score") or doc.get("score", 0)
            scores.append(score)

        analysis = {
            "total_count": len(documents),
            "region_distribution": dict(Counter(regions).most_common(5)),
            "field_distribution": dict(Counter(fields).most_common(5)),
            "average_score": sum(scores) / len(scores) if scores else 0,
            "max_score": max(scores) if scores else 0,
            "min_score": min(scores) if scores else 0,
            "deadlines_soon": len([d for d in deadlines if d]),
        }

        return analysis

    return Tool(
        name="analyze_results",
        description="추천 결과 분석 (지역/분야 분포, 점수 통계)",
        parameters={
            "documents": {"type": "array"},
            "user_profile": {"type": "object"}
        },
        function=analyze_results
    )