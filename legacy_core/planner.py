# core/agentic_planner.py
"""
Agentic AI 플래너 - ReAct 패턴 구현
- Thought: 현재 상황 분석
- Action: 도구 선택 및 실행
- Observation: 결과 관찰
- 반복적으로 목표 달성
"""

from typing import Dict, List, Any, Optional
import json
import re
from dataclasses import dataclass

from legacy_core.tools import ToolRegistry
from models.data import UserProfile


@dataclass
class AgentStep:
    """Agent의 한 단계 실행 결과"""
    thought: str
    action: str
    action_input: Dict[str, Any]
    observation: Any
    iteration: int


class AgenticPlanner:
    """
    ReAct 패턴 기반 Agentic AI 플래너
    LLM이 스스로 계획하고 도구를 선택해서 실행
    """

    def __init__(self, llm_client, tool_registry: ToolRegistry):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_iterations = 10
        self.steps: List[AgentStep] = []

    def plan_and_execute(
        self, 
        user_profile: Dict[str, Any], 
        task: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        주어진 작업(task)을 Agentic 방식으로 해결
        
        Args:
            user_profile: 사용자 프로필
            task: 수행할 작업 (예: "추천", "검색", "필터링")
            context: 추가 컨텍스트
        
        Returns:
            최종 결과 딕셔너리
        """
        self.steps = []
        context = context or {}
        
        # 초기 상태
        state = {
            "user_profile": user_profile,
            "task": task,
            "search_results": [],
            "filtered_results": [],
            "ranked_results": [],
            "final_answer": None,
            **context
        }

        print(f"\n{'='*80}")
        print(f"🤖 Agentic AI 플래닝 시작")
        print(f"  Task: {task}")
        print(f"{'='*80}\n")

        for iteration in range(self.max_iterations):
            print(f"\n🔄 Iteration {iteration + 1}/{self.max_iterations}")
            print("-" * 80)

            # 1️⃣ THOUGHT: 현재 상황 분석
            thought = self._generate_thought(state, iteration)
            print(f"💭 Thought: {thought}")

            # 2️⃣ ACTION: 다음 행동 결정
            action_plan = self._decide_action(state, thought, iteration)
            
            if action_plan.get("action") == "FINISH":
                print("✅ 작업 완료!")
                break

            action_name = action_plan.get("action")
            action_input = action_plan.get("parameters", {})
            print(f"🎬 Action: {action_name}")
            print(f"📥 Input: {json.dumps(action_input, ensure_ascii=False)[:100]}...")

            # 3️⃣ OBSERVATION: 도구 실행 및 결과 관찰
            observation = self._execute_action(action_name, action_input, state)
            print(f"👁 Observation: {self._summarize_observation(observation)}")

            # 4️⃣ 단계 기록
            step = AgentStep(
                thought=thought,
                action=action_name,
                action_input=action_input,
                observation=observation,
                iteration=iteration
            )
            self.steps.append(step)

            # 5️⃣ 상태 업데이트
            state = self._update_state(state, action_name, observation)

        # 최종 결과 생성
        return self._create_final_result(state)

    def _generate_thought(self, state: Dict, iteration: int) -> str:
        """현재 상황을 분석하는 Thought 생성"""
        
        search_count = len(state.get("search_results", []))
        filtered_count = len(state.get("filtered_results", []))
        ranked_count = len(state.get("ranked_results", []))

        prompt = f"""
당신은 창업지원 추천 시스템의 AI 에이전트입니다.
현재 상황을 분석하고 다음에 무엇을 해야 할지 생각하세요.

[작업 목표]
{state['task']}

[사용자 프로필]
- 지역: {state['user_profile'].get('region')}
- 분야: {state['user_profile'].get('business_field')}
- 단계: {state['user_profile'].get('business_stage')}
- 나이: {state['user_profile'].get('age')}

[현재 상태]
- 반복: {iteration + 1}/{self.max_iterations}
- 검색 결과: {search_count}개
- 필터링 결과: {filtered_count}개
- 순위 매긴 결과: {ranked_count}개

[지금까지 수행한 작업]
{self._format_history()}

다음에 무엇을 해야 할지 한 문장으로 생각을 적어주세요.
(예: "검색을 아직 안 했으니 먼저 검색해야겠다", "필터링 후 순위를 매겨야겠다")
"""

        try:
            response = self.llm.generate(
                prompt,
                system_prompt="간결하게 한 문장으로 생각을 표현하세요.",
                max_tokens=150
            )
            return str(response).strip()
        except Exception as e:
            return f"오류 발생: {e}"

    def _decide_action(
        self, 
        state: Dict, 
        thought: str, 
        iteration: int
    ) -> Dict[str, Any]:
        """다음 행동 결정 (LLM 기반)"""

        tools_desc = self.tools.get_tools_description()
        
        search_count = len(state.get("search_results", []))
        filtered_count = len(state.get("filtered_results", []))
        ranked_count = len(state.get("ranked_results", []))

        prompt = f"""
당신은 창업지원 추천 시스템의 AI 에이전트입니다.
현재 생각(Thought)을 바탕으로 다음 행동(Action)을 결정하세요.

[생각]
{thought}

[현재 상태]
- 검색 결과: {search_count}개
- 필터링 결과: {filtered_count}개
- 순위 매긴 결과: {ranked_count}개

[사용 가능한 도구]
{tools_desc}

[행동 선택 규칙]
1. 검색을 안 했으면 → search_database
2. 검색 결과가 있고 필터링을 안 했으면 → filter_results
3. 필터링 결과가 있고 순위를 안 매겼으면 → rank_results
4. 모든 단계를 완료했으면 → FINISH

다음 JSON 형식만 출력하세요:
{{
  "action": "search_database | filter_results | rank_results | FINISH",
  "reason": "이 행동을 선택한 이유",
  "parameters": {{
    // 도구에 넘길 파라미터
  }}
}}
"""

        try:
            response = self.llm.generate(
                prompt,
                system_prompt="JSON만 출력하세요. 다른 텍스트는 넣지 마세요.",
                max_tokens=500
            )

            # JSON 파싱
            text = str(response).strip()
            text = re.sub(r"```(?:json)?", "", text).strip()
            decision = json.loads(text)

            return decision

        except Exception as e:
            print(f"⚠️ Action 결정 실패: {e}")
            
            # 폴백: 규칙 기반
            if search_count == 0:
                return {"action": "search_database", "parameters": {}}
            elif filtered_count == 0:
                return {"action": "filter_results", "parameters": {}}
            elif ranked_count == 0:
                return {"action": "rank_results", "parameters": {}}
            else:
                return {"action": "FINISH", "parameters": {}}

    def _execute_action(
        self, 
        action_name: str, 
        action_input: Dict, 
        state: Dict
    ) -> Any:
        """도구 실행"""
        
        if action_name == "FINISH":
            return "작업 완료"

        tool = self.tools.get_tool(action_name)
        if not tool:
            return f"오류: {action_name} 도구를 찾을 수 없습니다"

        try:
            # 파라미터 자동 채우기
            params = self._prepare_parameters(action_name, action_input, state)
            
            # 실행
            result = tool.function(**params)
            return result

        except Exception as e:
            return f"도구 실행 오류: {e}"

    def _prepare_parameters(
        self, 
        action_name: str, 
        action_input: Dict, 
        state: Dict
    ) -> Dict[str, Any]:
        """도구 파라미터 자동 준비"""
        
        params = action_input.copy()

        if action_name == "search_database":
            profile = state["user_profile"]
            
            # 쿼리 자동 생성
            if "query" not in params or not params["query"]:
                query_parts = [
                    profile.get("region", ""),
                    profile.get("business_field", ""),
                    profile.get("business_stage", "")
                ]
                params["query"] = " ".join([p for p in query_parts if p])
            
            # top_k 자동 설정
            params.setdefault("top_k", 30)
            
            # data_types 자동 설정
            params.setdefault("data_types", profile.get("desired_data_types"))

        elif action_name == "filter_results":
            # 검색 결과 자동 전달
            params.setdefault("documents", state.get("search_results", []))
            
            # 프로필 조건 자동 전달
            profile = state["user_profile"]
            params.setdefault("region", profile.get("region"))
            params.setdefault("age", profile.get("age"))
            params.setdefault("is_disabled", profile.get("is_disabled"))
            params.setdefault("exclude_closed", True)

        elif action_name == "rank_results":
            # 필터링 결과 자동 전달
            params.setdefault("documents", state.get("filtered_results", []))
            params.setdefault("user_profile", state["user_profile"])
            params.setdefault("criteria", "relevance")

        return params

    def _update_state(
        self, 
        state: Dict, 
        action_name: str, 
        observation: Any
    ) -> Dict:
        """상태 업데이트"""
        
        if action_name == "search_database" and isinstance(observation, list):
            state["search_results"] = observation
            
        elif action_name == "filter_results" and isinstance(observation, list):
            state["filtered_results"] = observation
            
        elif action_name == "rank_results" and isinstance(observation, list):
            state["ranked_results"] = observation

        return state

    def _create_final_result(self, state: Dict) -> Dict[str, Any]:
        """최종 결과 생성"""
        
        ranked = state.get("ranked_results", [])
        
        return {
            "status": "SUCCESS" if ranked else "NO_RESULTS",
            "recommendations": ranked,
            "total_steps": len(self.steps),
            "agent_steps": [
                {
                    "iteration": s.iteration,
                    "thought": s.thought,
                    "action": s.action,
                    "observation_summary": self._summarize_observation(s.observation)
                }
                for s in self.steps
            ],
            "final_state": {
                "search_count": len(state.get("search_results", [])),
                "filtered_count": len(state.get("filtered_results", [])),
                "ranked_count": len(ranked)
            }
        }

    def _format_history(self) -> str:
        """실행 히스토리 포맷팅"""
        if not self.steps:
            return "아직 아무 작업도 수행하지 않았습니다."
        
        lines = []
        for i, step in enumerate(self.steps, 1):
            lines.append(f"{i}. {step.action} → {self._summarize_observation(step.observation)}")
        
        return "\n".join(lines)

    def _summarize_observation(self, observation: Any) -> str:
        """Observation 요약"""
        if isinstance(observation, list):
            return f"{len(observation)}개 결과"
        elif isinstance(observation, dict):
            return f"딕셔너리 ({len(observation)} keys)"
        elif isinstance(observation, str):
            return observation[:100]
        else:
            return str(type(observation))