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
        self.name = name
        self.llm = llm_client
        self.thoughts: List[AgentThought] = []
        self.memory: List[str] = []
    
    def think(self, thought: str, action: str = "", result: str = "", confidence: float = 0.0):
        """사고 과정 기록"""
        thought_obj = AgentThought(
            agent_name=self.name,
            timestamp=datetime.now().strftime("%H:%M:%S"),
            thought=thought,
            action=action,
            result=result,
            confidence=confidence
        )
        self.thoughts.append(thought_obj)
        
        if Config.AGENT_VERBOSE:
            print(f"🧠 [{self.name}] {thought}")
            if result:
                print(f"   ✓ {result}")
    
    def get_thoughts(self) -> List[AgentThought]:
        """사고 과정 반환"""
        return self.thoughts
