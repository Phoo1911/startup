# core/tools.py
"""
Agentic AI를 위한 도구 시스템 (완성본)
"""

from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import json


@dataclass
class Tool:
    """에이전트가 사용할 수 있는 도구"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable


class ToolRegistry:
    """도구 레지스트리"""

    def __init__(self):
        self.tools: Dict[str, Tool] = {}

    def register(self, tool: Tool):
        """도구 등록"""
        self.tools[tool.name] = tool

    def get_tool(self, name: str) -> Optional[Tool]:
        """도구 가져오기"""
        return self.tools.get(name)

    def list_tools(self) -> List[str]:
        """사용 가능한 도구 목록"""
        return list(self.tools.keys())

    def get_tools_description(self) -> str:
        """LLM에게 제공할 도구 설명"""
        descriptions = []
        for name, tool in self.tools.items():
            params = json.dumps(tool.parameters, ensure_ascii=False, indent=2)
            descriptions.append(
                f"Tool: {name}\n"
                f"Description: {tool.description}\n"
                f"Parameters: {params}\n"
            )
        return "\n".join(descriptions)


def create_search_tool(rag_system) -> Tool:
    """RAG 검색 도구"""

    def search(
        query: str,
        top_k: int = 5,
        data_types: Optional[List[str]] = None,
    ) -> List[Dict]:
        print(f"\n[SEARCH] RAG 검색 시작")
        print(f"  - query: {query}")
        print(f"  - top_k: {top_k}")
        print(f"  - data_types: {data_types}")
        
        results = rag_system.retrieve(
            query=query,
            top_k=top_k,
            allowed_types=data_types,
        )
        
        print(f"[SEARCH] RAG 검색 결과: {len(results)}개")

        documents: List[Dict] = []
        for idx, (doc, score) in enumerate(results):
            meta = doc.metadata or {}
            
            if idx < 3:
                print(f"  [{idx+1}] {meta.get('title', '제목없음')[:50]} (score: {score:.3f})")

            documents.append(
                {
                    "id": doc.id,
                    "title": meta.get("title", "제목없음"),
                    "type": meta.get("type", "unknown"),
                    "content": doc.text[:400],
                    "score": float(score),
                    "metadata": {
                        "region": meta.get("region", ""),
                        "field": meta.get("field", ""),
                        "deadline": meta.get("deadline", ""),
                        "apply_period": meta.get("apply_period", ""),
                        "status": meta.get("status", ""),
                        "apply_target": meta.get("apply_target", ""),
                        "startup_period": meta.get("startup_period", ""),
                        "age_limit": meta.get("age_limit", ""),
                        "host_org": meta.get("host_org", ""),
                        "detail_url": meta.get("detail_url", ""),
                        "apply_url": meta.get("apply_url", ""),
                        "guide_url": meta.get("guide_url", ""),
                        "reg_date": meta.get("reg_date", ""),
                        "view_cnt": meta.get("view_cnt", ""),
                        "play_time": meta.get("play_time", 0),
                        "address": meta.get("address", ""),
                        "center_name": meta.get("center_name", ""),
                        "rent": meta.get("rent", 0),
                        "seat_count": meta.get("seat_count", ""),
                    },
                }
            )

        return documents

    return Tool(
        name="search_database",
        description=(
            "벡터 데이터베이스에서 지원사업, 통합공고, 교육·강좌, 창업공간/센터, "
            "인증 제품/기업, 기관·통계 자료 등을 검색합니다."
        ),
        parameters={
            "query": {"type": "string", "description": "검색 쿼리"},
            "top_k": {"type": "integer", "description": "결과 개수", "default": 5},
            "data_types": {
                "type": "array",
                "description": "검색할 데이터 타입 목록",
                "items": [
                    "announcement",
                    "business",
                    "content",
                    "statistical",
                    "lecture",
                    "space",
                    "center",
                    "product",
                    "corporate",
                    "institution",
                ],
            },
        },
        function=search,
    )


def create_filter_tool() -> Tool:
    """조건 필터링 도구 (나이 + 장애인 필터 포함)"""
    from datetime import datetime

    def filter_by_conditions(
        documents: List[Dict],
        region: Optional[str] = None,
        age: Optional[int] = None,
        business_stage: Optional[str] = None,
        is_disabled: Optional[bool] = None,
        exclude_closed: bool = True,
    ) -> List[Dict]:
        print(f"\n[FILTER] 필터링 시작: {len(documents)}개 문서")
        print(f"  - region: {region}")
        print(f"  - age: {age}")
        print(f"  - is_disabled: {is_disabled}")
        print(f"  - exclude_closed: {exclude_closed}")
        
        filtered: List[Dict] = []
        today = datetime.now().date()
        
        excluded_count = {"region": 0, "age": 0, "disabled": 0, "closed": 0}

        for doc in documents:
            meta = doc.get("metadata", {})
            doc_type = doc.get("type", "") or doc.get("data_type", "")

            # 1️⃣ 지역 필터
            if region and region != "전국":
                doc_region = (meta.get("region") or doc.get("region") or "").strip()
                if doc_region and doc_region not in ["전국", "", "전 지역"]:
                    if region not in doc_region:
                        excluded_count["region"] += 1
                        continue

            # 2️⃣ 나이 필터
            if age is not None and age > 0:
                age_limit_str = (meta.get("age_limit") or doc.get("age_limit") or "").strip()
                if age_limit_str:
                    try:
                        import re
                        numbers = re.findall(r'\d+', age_limit_str)
                        if numbers:
                            limit_age = int(numbers[0])
                            
                            if any(keyword in age_limit_str for keyword in ["이하", "까지", "만"]):
                                if age > limit_age:
                                    excluded_count["age"] += 1
                                    continue
                            elif "미만" in age_limit_str:
                                if age >= limit_age:
                                    excluded_count["age"] += 1
                                    continue
                            elif "이상" in age_limit_str:
                                if age < limit_age:
                                    excluded_count["age"] += 1
                                    continue
                    except Exception:
                        pass

            # 3️⃣ 장애인 여부 필터
            if is_disabled is False:
                title = doc.get("title", "").lower()
                apply_target = (meta.get("apply_target") or doc.get("apply_target") or "").lower()
                content = doc.get("content", "")[:500].lower()
                
                if "장애인" in title or "장애인" in apply_target:
                    excluded_count["disabled"] += 1
                    print(f"[FILTER 제외-장애인] {doc.get('title', '')[:40]}...")
                    continue
                
                strong_keywords = ["장애인전용", "장애인기업", "장애인만", "장애인 기업", "장애인 전용"]
                if any(kw in content for kw in strong_keywords):
                    excluded_count["disabled"] += 1
                    print(f"[FILTER 제외-장애인] {doc.get('title', '')[:40]}...")
                    continue

            # 4️⃣ 마감 필터
            if exclude_closed and doc_type in {"announcement", "business", "product"}:
                status = (meta.get("status") or doc.get("status") or "").strip().upper()
                if status in {"N", "마감", "종료", "CLOSED"}:
                    excluded_count["closed"] += 1
                    continue

                deadline = (meta.get("deadline") or doc.get("deadline") or "").strip()
                if deadline:
                    try:
                        deadline_clean = deadline.replace("-", "").replace(".", "").replace("/", "")
                        if len(deadline_clean) >= 8 and deadline_clean[:8].isdigit():
                            deadline_date = datetime.strptime(deadline_clean[:8], "%Y%m%d").date()
                            if deadline_date < today:
                                excluded_count["closed"] += 1
                                continue
                    except Exception:
                        pass

            doc["filter_reasons"] = []
            if region and (meta.get("region") or doc.get("region")):
                doc_region = meta.get("region") or doc.get("region")
                if doc_region == "전국":
                    doc["filter_reasons"].append(f"전국 지원 대상")
                elif region in doc_region:
                    doc["filter_reasons"].append(f"지역 일치: {region}")

            filtered.append(doc)
        
        print(f"[FILTER] 필터링 완료: {len(filtered)}개 남음")
        print(f"  - 지역 제외: {excluded_count['region']}개")
        print(f"  - 나이 제외: {excluded_count['age']}개")
        print(f"  - 장애인 제외: {excluded_count['disabled']}개")
        print(f"  - 마감 제외: {excluded_count['closed']}개")

        return filtered

    return Tool(
        name="filter_results",
        description="검색 결과를 사용자 조건에 맞게 필터링합니다.",
        parameters={
            "documents": {"type": "array"},
            "region": {"type": "string"},
            "age": {"type": "integer"},
            "business_stage": {"type": "string"},
            "is_disabled": {"type": "boolean"},
            "exclude_closed": {"type": "boolean", "default": True},
        },
        function=filter_by_conditions,
    )


def create_ranking_tool() -> Tool:
    """순위 매기기 도구 (디버깅 로그 포함)"""

    def rank_documents(
        documents: List[Dict],
        user_profile: Dict,
        criteria: str = "score",
    ) -> List[Dict]:
        from datetime import datetime

        print(f"\n[RANK] 순위 매기기 시작: {len(documents)}개 문서")

        seen_keys = set()
        unique_docs = []
        duplicate_count = 0
        
        for doc in documents:
            doc_id = doc.get("id")
            title = (doc.get("title") or "").strip()
            host_org = (doc.get("host_org") or doc.get("metadata", {}).get("host_org") or "").strip()
            
            if not title:
                print(f"[RANK SKIP] 제목 없음")
                continue
            
            if doc_id:
                key = f"id:{doc_id}"
            elif title and host_org:
                key = f"title_org:{title}_{host_org}"
            else:
                key = f"title:{title}"
            
            if key in seen_keys:
                duplicate_count += 1
                print(f"[RANK DUPLICATE] {title[:40]}...")
                continue
            
            seen_keys.add(key)
            unique_docs.append(doc)

        print(f"[RANK] 중복 제거 완료: {len(unique_docs)}개 남음 (중복 {duplicate_count}개 제거)")

        for doc in unique_docs:
            meta = doc.get("metadata", {})
            reasons = doc.get("filter_reasons", []).copy()

            score = doc.get("score", 0)
            if score > 0.8:
                reasons.append(f"✅ 매우 높은 유사도 ({score:.1%})")
            elif score > 0.6:
                reasons.append(f"✅ 높은 유사도 ({score:.1%})")

            user_region = user_profile.get("region", "")
            doc_region = meta.get("region", "") or doc.get("region", "")
            if user_region and doc_region:
                if doc_region == "전국":
                    reasons.append("🌏 전국 대상 사업")
                elif user_region in doc_region:
                    reasons.append(f"📍 지역 일치: {user_region}")

            user_field = user_profile.get("business_field", "")
            doc_field = meta.get("field", "") or doc.get("field", "")
            content = doc.get("content", "").lower()
            if user_field and (user_field.lower() in content or user_field in doc_field):
                reasons.append(f"💼 사업분야 일치: {user_field}")

            user_target = user_profile.get("target_type", "")
            apply_target = meta.get("apply_target", "") or doc.get("apply_target", "")
            if user_target and user_target in apply_target:
                reasons.append(f"👥 대상유형 일치: {user_target}")

            user_stage = user_profile.get("business_stage", "")
            startup_period = meta.get("startup_period", "") or doc.get("startup_period", "")
            if user_stage and user_stage in startup_period:
                reasons.append(f"📈 창업단계 적합: {user_stage}")

            deadline = meta.get("deadline", "") or doc.get("deadline", "")
            if deadline:
                try:
                    deadline_digits = deadline.replace("-", "")
                    if len(deadline_digits) == 8:
                        d = datetime.strptime(deadline_digits, "%Y%m%d").date()
                        today = datetime.now().date()
                        days_left = (d - today).days

                        if 0 <= days_left <= 7:
                            reasons.append(f"⚠️ 마감 임박 (D-{days_left})")
                        elif 0 <= days_left <= 30:
                            reasons.append(f"📅 신청 가능 (D-{days_left})")
                except Exception:
                    pass

            doc["match_reasons"] = reasons if reasons else ["✅ 프로필 조건 충족"]
            doc["reasons"] = doc["match_reasons"]

        if criteria == "deadline":
            def deadline_key(doc):
                deadline = doc.get("metadata", {}).get("deadline") or doc.get("deadline", "99991231")
                if deadline and deadline.replace("-", "").isdigit():
                    return deadline.replace("-", "")
                return "99991231"
            unique_docs = sorted(unique_docs, key=deadline_key)

        elif criteria == "relevance":
            user_region = user_profile.get("region", "")
            def relevance_key(doc):
                score = doc.get("score", 0)
                doc_region = doc.get("metadata", {}).get("region") or doc.get("region", "")
                if user_region and doc_region:
                    if doc_region == "전국" or user_region in doc_region:
                        score += 0.2
                return -score
            unique_docs = sorted(unique_docs, key=relevance_key)

        else:
            unique_docs = sorted(unique_docs, key=lambda d: -d.get("score", 0))

        for idx, doc in enumerate(unique_docs, 1):
            doc["rank"] = idx

        print(f"[RANK] 순위 매기기 완료: {len(unique_docs)}개")
        if unique_docs:
            for i, doc in enumerate(unique_docs[:3], 1):
                print(f"  {i}. {doc.get('title', '제목없음')[:50]}")

        return unique_docs

    return Tool(
        name="rank_results",
        description="검색 결과에 순위를 매기고 상세한 매칭 이유를 추가합니다.",
        parameters={
            "documents": {"type": "array"},
            "user_profile": {"type": "object"},
            "criteria": {
                "type": "string",
                "enum": ["score", "deadline", "relevance"],
                "default": "relevance",
            },
        },
        function=rank_documents,
    )


def create_analysis_tool(llm_client) -> Tool:
    """LLM 분석 도구"""

    def analyze_match(document: Dict, user_profile: Dict) -> str:
        if not llm_client:
            return ""

        title = document.get("title", "")
        content = document.get("content", "")[:300]
        meta = document.get("metadata", {})

        prompt = f"""
다음 지원사업이 사용자에게 왜 적합한지 한 문장으로 설명하세요.

[사용자]
- 지역: {user_profile.get('region')}
- 분야: {user_profile.get('business_field')}
- 단계: {user_profile.get('business_stage')}

[지원사업]
제목: {title}
내용: {content}
지역: {meta.get('region', '')}
대상: {meta.get('apply_target', '')}

한 문장으로 "이 사업은 ~하기 때문에 적합합니다" 형식으로:
"""

        try:
            result = llm_client.generate(prompt, "간결하게", max_tokens=80)
            return str(result).strip()
        except Exception:
            return ""

    return Tool(
        name="analyze_match",
        description="LLM으로 매칭 적합성을 분석합니다.",
        parameters={
            "document": {"type": "object"},
            "user_profile": {"type": "object"},
        },
        function=analyze_match,
    )


def create_summary_tool(llm_client) -> Tool:
    """요약 도구"""

    def summarize_results(documents: List[Dict], user_profile: Dict) -> str:
        if not llm_client or not documents:
            return ""

        top_titles = [d.get("title", "") for d in documents[:3]]
        titles_text = "\n".join(f"{i+1}. {t}" for i, t in enumerate(top_titles))

        prompt = f"""
{user_profile.get('name', '사용자')}님께 다음 지원사업을 추천합니다:

{titles_text}

2-3문장으로 따뜻하게 요약하고 격려해주세요.
"""

        try:
            result = llm_client.generate(prompt, "친절한 상담사", max_tokens=150)
            return str(result).strip()
        except Exception:
            return f"{len(documents)}개의 적합한 지원사업을 찾았습니다."

    return Tool(
        name="summarize_recommendations",
        description="추천 결과를 요약합니다.",
        parameters={
            "documents": {"type": "array"},
            "user_profile": {"type": "object"},
        },
        function=summarize_results,
    )