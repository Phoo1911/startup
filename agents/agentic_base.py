# agents/agentic_base.py
"""
Agentic AI 기반 에이전트 모음
- RecommendationAgenticAgent: 도구를 써서 검색 → 필터 → 랭킹
- ChatbotAgenticAgent       : 검색 결과를 참고해서 상담 답변
"""

from typing import Dict, List, Any, Optional
import json
import re

from agents.base import AgenticAgent
from core.tools import ToolRegistry


# ───────────────────────────────────────────────
# 🔥 HTML 태그 제거 유틸
# ───────────────────────────────────────────────
def clean_html(text: Any) -> Any:
    """간단한 HTML 제거: <br> → 줄바꿈, 나머지 태그 삭제"""
    if not isinstance(text, str):
        return text

    # <br>, <br/> → 줄바꿈
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    # 모든 HTML 태그 제거
    text = re.sub(r"<.*?>", "", text)
    # 줄바꿈 정리
    text = re.sub(r"\n+", "\n", text)

    return text.strip()


# ───────────────────────────────────────────────
# 🔥 추천용 Agentic 에이전트
# ───────────────────────────────────────────────
class RecommendationAgenticAgent(AgenticAgent):
    """
    추천을 위한 Agentic AI 에이전트
    - LLM이 search / filter / rank 도구를 단계적으로 호출
    - 마지막에 ranked_results를 recommendations로 반환
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry):
        super().__init__("AgenticRecommender", llm_client)
        self.tools = tool_registry
        self.max_iterations = 10

    # 메인 엔트리
    def recommend(self, user_profile: Dict[str, Any], top_n: int = 10) -> Dict[str, Any]:
        self.think(
            "추천 작업 시작",
            action=f"사용자: {user_profile.get('name', '사용자')}",
            confidence=0.9,
        )

        # 에이전트 내부 상태
        context = {
            "user_profile": user_profile,
            "top_n": top_n,
            "search_results": [],
            "filtered_results": [],
            "ranked_results": [],
        }

        iterations_used = 0

        # Agentic 루프
        for iteration in range(self.max_iterations):
            iterations_used = iteration + 1

            self.think(
                f"반복 {iteration + 1}/{self.max_iterations}",
                action="다음 행동 결정 중",
                confidence=0.8,
            )

            next_action = self._decide_next_action(context, iteration)

            # FINISH면 루프 종료
            if next_action.get("action") == "FINISH":
                self.think(
                    "작업 완료",
                    result=f"{len(context.get('ranked_results', []))}개 추천",
                    confidence=1.0,
                )
                break

            tool_result = self._execute_tool(next_action, context)

            if tool_result is not None:
                context = self._update_context(context, next_action, tool_result)
                
                # 🔥 디버깅: 각 단계별 결과 출력
                print(f"\n[DEBUG] {next_action.get('action')} 후:")
                print(f"  - search_results: {len(context.get('search_results', []))}개")
                print(f"  - filtered_results: {len(context.get('filtered_results', []))}개")
                print(f"  - ranked_results: {len(context.get('ranked_results', []))}개")

        # 🔥 최종 결과 결정 로직 개선
        ranked = context.get("ranked_results", [])
        
        # ranked가 비어있거나 부족하면 폴백
        if len(ranked) < top_n:
            print(f"\n[WARNING] ranked_results가 부족함 ({len(ranked)}개). 폴백 시도...")
            
            # 폴백 순서: filtered_results → search_results
            fallback_docs = context.get("filtered_results") or context.get("search_results") or []
            print(f"[DEBUG] 폴백 문서 수: {len(fallback_docs)}개")
            
            if fallback_docs and len(fallback_docs) > len(ranked):
                rank_tool = self.tools.get_tool("rank_results")
                if rank_tool:
                    try:
                        # 🔥 중복 제거: 이미 ranked에 있는 문서는 제외
                        ranked_titles = {doc.get('title') for doc in ranked if doc.get('title')}
                        new_docs = [
                            doc for doc in fallback_docs 
                            if doc.get('title') not in ranked_titles
                        ]
                        
                        print(f"[DEBUG] 폴백 대상 문서: {len(new_docs)}개")
                        
                        if new_docs:
                            fallback_ranked = rank_tool.function(
                                documents=new_docs,
                                user_profile=context["user_profile"],
                                criteria="relevance",
                            )
                            
                            # 기존 ranked와 합치기
                            ranked = ranked + fallback_ranked
                            print(f"[DEBUG] 폴백 후 총 개수: {len(ranked)}개")
                            
                            self.think(
                                "폴백으로 rank_results 실행",
                                result=f"총 {len(ranked)}개",
                                confidence=0.9,
                            )
                    except Exception as e:
                        print(f"[ERROR] 폴백 rank 실패: {e}")
                        self.think(
                            "폴백 rank_results 실패",
                            result=str(e),
                            confidence=0.2,
                        )
        
        # 🔥 그래도 결과가 없으면 최후의 수단: 필터되지 않은 검색 결과 사용
        if not ranked:
            print("[EMERGENCY] 모든 단계 실패. 원본 검색 결과 사용...")
            ranked = context.get("search_results", [])[:top_n]

        final_results = ranked[:top_n]
        print(f"\n[FINAL] 최종 추천 개수: {len(final_results)}개")
        
        # 🔥 최종 결과 타이틀 출력 (디버깅용)
        for i, doc in enumerate(final_results[:5], 1):
            print(f"  {i}. {doc.get('title', '제목없음')}")
        
        return {
            "recommendations": final_results,
            "iterations": iterations_used,
            "agent_thoughts": self.get_thoughts(),
        }

    # ───────────────────────────────────────────────
    # 1) 다음 액션 선택 (LLM)
    # ───────────────────────────────────────────────
    def _decide_next_action(self, context: Dict[str, Any], iteration: int) -> Dict[str, Any]:
        tools_desc = self.tools.get_tools_description()

        has_search = len(context.get("search_results", [])) > 0
        has_filtered = len(context.get("filtered_results", [])) > 0
        has_ranked = len(context.get("ranked_results", [])) > 0

        prompt = f"""
당신은 창업지원 추천 시스템의 AI 에이전트입니다.
현재 상태를 보고 다음에 사용할 도구를 결정하세요.

[사용자 프로필]
{json.dumps(context['user_profile'], ensure_ascii=False, indent=2)}

[상태]
- 검색 완료: {has_search} ({len(context.get('search_results', []))}개)
- 필터 완료: {has_filtered} ({len(context.get('filtered_results', []))}개)
- 순위 완료: {has_ranked} ({len(context.get('ranked_results', []))}개)

[사용 가능한 도구]
{tools_desc}

다음 JSON 형식만 출력하세요.
{{
  "action": "search_database | filter_results | rank_results | FINISH",
  "reason": "이 행동을 선택한 이유",
  "parameters": {{
    // 각 도구에 넘길 파라미터 (없으면 비워두기)
  }}
}}
"""

        try:
            response = self.llm.generate(
                prompt,
                "JSON만 출력하세요.",
                max_tokens=800,
            )

            text = str(response).strip()
            # ```json 코드 블록 제거
            text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()

            decision = json.loads(text)

            self.think(
                f"결정: {decision.get('action')}",
                action=decision.get("reason", ""),
                confidence=0.85,
            )

            return decision

        except Exception as e:
            # 파싱 실패 시 기본 플로우
            self.think("결정 실패 → 기본 플로우 사용", result=str(e), confidence=0.3)

            if not has_search:
                return {"action": "search_database", "parameters": {}}
            elif not has_filtered:
                return {"action": "filter_results", "parameters": {}}
            elif not has_ranked:
                return {"action": "rank_results", "parameters": {}}
            else:
                return {"action": "FINISH", "parameters": {}}

    # ───────────────────────────────────────────────
    # 2) 도구 실행
    # ───────────────────────────────────────────────
    def _execute_tool(self, action: Dict[str, Any], context: Dict[str, Any]) -> Any:
        tool_name = action.get("action")

        if tool_name == "FINISH":
            return None

        tool = self.tools.get_tool(tool_name)
        if not tool:
            self.think(f"도구 없음: {tool_name}", confidence=0.1)
            return None

        try:
            params = action.get("parameters", {}) or {}

            # 자동 파라미터 구성
            if tool_name == "search_database":
                profile = context["user_profile"]
                default_query = f"{profile.get('region', '')} {profile.get('business_field', '')} {profile.get('business_stage', '')}".strip()

                params.setdefault("query", default_query or "창업 지원사업")
                params.setdefault("top_k", max(context["top_n"] * 5, 50))
                params.setdefault("data_types", profile.get("desired_data_types"))

            elif tool_name == "filter_results":
                params.setdefault("documents", context.get("search_results", []))
                params.setdefault("region", context["user_profile"].get("region"))
                params.setdefault("age", context["user_profile"].get("age"))
                params.setdefault("is_disabled", context["user_profile"].get("is_disabled"))

            elif tool_name == "rank_results":
                params.setdefault("documents", context.get("filtered_results", []))
                params.setdefault("user_profile", context["user_profile"])

            # 도구 실행
            result = tool.function(**params)

            # 🔥 search_database 결과는 즉시 중복 제거
            if tool_name == "search_database" and isinstance(result, list):
                seen_titles = set()
                deduplicated = []
                for doc in result:
                    title = doc.get("title", "").strip()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        deduplicated.append(doc)
                
                print(f"[SEARCH 중복제거] {len(result)}개 → {len(deduplicated)}개")
                result = deduplicated

            # 결과 개수 표시
            count_str = (
                f"{len(result)}개" if isinstance(result, list) else "1개"
            )
            self.think(
                f"도구 실행: {tool_name}",
                result=count_str,
                confidence=0.9,
            )

            return result

        except Exception as e:
            self.think(
                f"도구 실행 실패: {tool_name}",
                result=str(e),
                confidence=0.2,
            )
            return None


    # 3) 컨텍스트 업데이트 + HTML 정리

    def _update_context(self, context: Dict[str, Any], action: Dict[str, Any], result: Any) -> Dict[str, Any]:
        tool_name = action.get("action")

        def clean_docs(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            cleaned: List[Dict[str, Any]] = []
            seen_keys = set()

            for d in docs:
                if not isinstance(d, dict):
                    continue

                # 🔥 강화된 중복 체크: ID > (title + host_org) > title
                doc_id = d.get("id")
                title = (d.get("title") or "").strip()
                host_org = (d.get("host_org") or d.get("metadata", {}).get("host_org") or "").strip()
                
                if doc_id:
                    key = f"id:{doc_id}"
                elif title and host_org:
                    key = f"title_org:{title}_{host_org}"
                elif title:
                    key = f"title:{title}"
                else:
                    continue
                
                if key in seen_keys:
                    print(f"[DUPLICATE SKIP] {title[:30]}...")
                    continue
                seen_keys.add(key)

                d = d.copy()

                # 🔹 1) metadata 평탄화
                meta = d.get("metadata") or {}
                if isinstance(meta, dict):
                    important_fields = [
                        "type", "data_type", "title", "region", "field", 
                        "deadline", "apply_period", "status", "detail_url", 
                        "apply_url", "guide_url", "apply_target", "host_org",
                        "startup_period", "age_limit", "reg_date", "view_cnt",
                        "play_time", "address", "center_name", "rent", "seat_count"
                    ]
                    
                    for field in important_fields:
                        if field in meta and field not in d:
                            d[field] = meta[field]

                    if "data_type" not in d:
                        d["data_type"] = meta.get("data_type") or meta.get("type", "")

                # 🔹 2) extra 딕셔너리 구성
                if "extra" not in d:
                    d["extra"] = {}
                
                if "apply_period" in d and d["apply_period"]:
                    d["extra"]["apply_period"] = d["apply_period"]
                
                if "deadline" in d and d["deadline"]:
                    d["extra"]["deadline"] = d["deadline"]

                # 🔹 3) HTML 정리
                for key in ["summary", "description", "content", "title"]:
                    if key in d and d[key] is not None:
                        d[key] = clean_html(d[key])

                cleaned.append(d)

            return cleaned

        if tool_name == "search_database" and isinstance(result, list):
            cleaned = clean_docs(result)
            context["search_results"] = cleaned
            print(f"[CONTEXT UPDATE] search_results: {len(cleaned)}개 저장")

        elif tool_name == "filter_results" and isinstance(result, list):
            cleaned = clean_docs(result)
            context["filtered_results"] = cleaned
            print(f"[CONTEXT UPDATE] filtered_results: {len(cleaned)}개 저장")

        elif tool_name == "rank_results" and isinstance(result, list):
            cleaned = clean_docs(result)
            context["ranked_results"] = cleaned
            print(f"[CONTEXT UPDATE] ranked_results: {len(cleaned)}개 저장")

        return context



# ───────────────────────────────────────────────
# 🔥 챗봇용 Agentic 에이전트
# ───────────────────────────────────────────────
class ChatbotAgenticAgent(AgenticAgent):
    """
    상담 챗봇용 Agentic 에이전트
    - 모든 질문에 자동으로 RAG 검색
    - 검색 결과만 사용해서 답변 생성
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry):
        super().__init__("AgenticChatbot", llm_client)
        self.tools = tool_registry
        self.max_iterations = 5

    def chat(
        self,
        user_profile: Dict[str, Any],
        question: str,
        category: Optional[str] = None,
    ) -> str:
        self.think(
            "챗봇 질문 처리",
            action=f"질문: {question[:50]}...",
            confidence=0.9,
        )

        # 🔥 무조건 RAG 검색 실행 (의미 유사도 기반)
        context_docs: List[Dict[str, Any]] = []
        search_tool = self.tools.get_tool("search_database")
        if search_tool:
            try:
                docs = search_tool.function(
                    query=question,
                    top_k=5,
                    data_types=user_profile.get("desired_data_types"),
                )

                # 🔥 HTML 정리
                context_docs = []
                for d in docs:
                    if not isinstance(d, dict):
                        continue
                    d = d.copy()
                    d["summary"] = clean_html(d.get("summary", ""))
                    d["description"] = clean_html(d.get("description", ""))
                    d["content"] = clean_html(d.get("content", ""))
                    context_docs.append(d)

                self.think(
                    "RAG 검색 완료",
                    result=f"{len(context_docs)}개 문서",
                    confidence=0.9,
                )
            except Exception as e:
                self.think("검색 실패", result=str(e), confidence=0.3)

        # 🔥 검색 결과가 없거나 유사도가 너무 낮으면 알림
        if not context_docs:
            self.think("검색 결과 없음", confidence=0.5)
            return "죄송합니다. 해당 질문에 대한 관련 정보를 찾지 못했습니다. 질문을 더 구체적으로 해주시거나, 창업지원과 관련된 다른 내용을 물어봐 주세요."

        # 🔥 유사도 필터링 (너무 낮은 결과 제외)
        high_quality_docs = [
            doc for doc in context_docs 
            if doc.get("score", 0) > 0.3  # 유사도 0.3 이상만
        ]
        
        if not high_quality_docs:
            self.think("유사도 낮음", confidence=0.4)
            return "질문과 관련된 정확한 정보를 찾기 어렵습니다. 창업지원, 교육, 공간, 지원사업 등 구체적인 주제로 질문해주시면 더 정확한 답변을 드릴 수 있습니다."

        return self._generate_answer(user_profile, question, high_quality_docs, category)

    # 최종 답변 생성
    def _generate_answer(
        self,
        user_profile: Dict[str, Any],
        question: str,
        context_docs: List[Dict[str, Any]],
        category: Optional[str],
    ) -> str:
        # 🔥 context 구성 강화 - 더 많은 정보 포함
        context_parts = []
        for i, doc in enumerate(context_docs[:5], 1):
            title = doc.get("title", "제목없음")
            content = doc.get("content", "")[:400]
            summary = doc.get("summary", "")[:200]
            
            meta = doc.get("metadata", {})
            region = meta.get("region") or doc.get("region") or ""
            host_org = meta.get("host_org") or doc.get("host_org") or ""
            apply_target = meta.get("apply_target") or doc.get("apply_target") or ""
            
            doc_text = f"""
[{i}] {title}
- 지역: {region}
- 주최: {host_org}
- 대상: {apply_target}
- 요약: {summary if summary else content[:200]}
- 상세: {content}
"""
            context_parts.append(doc_text.strip())
        
        context_text = "\n\n".join(context_parts)
        
        print(f"\n[CONTEXT] LLM에게 전달할 문서 수: {len(context_docs)}")
        print(f"[CONTEXT] 총 길이: {len(context_text)} 글자")

        # 🔥 사용자 정보 포맷팅
        user_info_parts = []
        if user_profile.get('age'):
            user_info_parts.append(f"{user_profile.get('age')}세")
        if user_profile.get('region'):
            user_info_parts.append(user_profile.get('region'))
        if user_profile.get('business_stage'):
            user_info_parts.append(user_profile.get('business_stage'))
        if user_profile.get('business_field'):
            user_info_parts.append(user_profile.get('business_field'))
        
        background_section = ""
        if user_info_parts or category:
            background_section = "배경 정보 (참고용):\n"
            if user_info_parts:
                background_section += f"- 사용자: {', '.join(user_info_parts)}\n"
            if category:
                background_section += f"- 카테고리: {category}\n"
            background_section += "\n"

        # 🔥 강화된 프롬프트
        prompt = f"""
당신은 창업지원 상담 AI입니다. 아래 데이터베이스 검색 결과를 바탕으로 답변하세요.

{background_section}질문: {question}

=== 📚 데이터베이스 검색 결과 ({len(context_docs)}개) ===
{context_text}
=== 검색 결과 끝 ===

답변 규칙:
1. **위 검색 결과만 사용**하여 답변하세요
2. 검색 결과에 있는 프로그램/사업을 구체적으로 언급하세요
3. 지역, 주최기관, 대상 등 세부 정보를 포함하세요
4. 3-5문장으로 자연스럽게 작성하세요
5. 외부 지식(Udemy, Coursera 등) 절대 사용 금지
6. 검색 결과가 질문과 맞지 않으면 솔직하게 "검색된 정보가 질문과 정확히 일치하지 않습니다"라고 말하세요

답변:"""

        try:
            answer = self.llm.generate(
                prompt,
                "검색 결과를 반드시 활용하여 구체적으로 답변하세요.",
                max_tokens=1000,
            )

            result = str(answer).strip()
            
            print(f"\n[LLM 답변] {result[:200]}...")
            
            self.think("답변 생성 완료", confidence=0.95)
            return result

        except Exception as e:
            self.think("답변 생성 실패", result=str(e), confidence=0.2)

            return f"다음 {len(context_docs)}개의 관련 정보를 찾았습니다:\n\n" + "\n".join(
                [f"• {doc.get('title', '제목없음')}" for doc in context_docs[:3]]
            )