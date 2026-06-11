"""
에이전트 기본 클래스
"""
from typing import List, Optional
from datetime import datetime

from models.data import AgentThought
from config.settings import Config


class AgenticAgent:
    """에이전트 기본 클래스"""

    def __init__(self, name: str, llm_client: Optional[object] = None):
        # 에이전트 이름 (ex. "RAGBuilder", "Recommender")
        self.name = name
        # 사용할 LLM 클라이언트 (없으면 None)
        self.llm = llm_client
        # 사고 과정(로그) 저장 리스트
        self.thoughts: List[AgentThought] = []
        # 간단한 메모용 리스트 (필요하면 나중에 확장)
        self.memory: List[str] = []

    def think(
        self,
        thought: str,
        action: str = "",
        result: str = "",
        confidence: float = 0.0,
    ):
        """에이전트가 어떤 생각/행동을 했는지 한 줄 기록"""

        thought_obj = AgentThought(
            agent_name=self.name,
            # 예: "2025-11-25T14:03:12" 이런 식으로 저장
            timestamp=datetime.now().isoformat(timespec="seconds"),
            thought=thought,
            action=action,
            result=result,
            confidence=confidence,
        )
        self.thoughts.append(thought_obj)

        # 디버깅용: 설정에서 AGENT_VERBOSE=True 이면 콘솔에 찍어줌
        if Config.AGENT_VERBOSE:
            print(f"🧠 [{self.name}] {thought}")
            if result:
                print(f"   ✓ {result}")

    def get_thoughts(self) -> List[AgentThought]:
        """지금까지 기록한 사고 로그 전부 가져오기"""
        return self.thoughts

