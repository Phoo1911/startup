"""
데이터 모델 정의
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import numpy as np

@dataclass
class UserProfile:
    """사용자 프로필"""
    name: str
    age: int
    region: str
    business_stage: str
    business_field: str
    target_type: str
    is_veteran: bool = False
    is_disabled: bool = False
    company_name: Optional[str] = None
    additional_context: str = ""
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'age': self.age,
            'region': self.region,
            'business_stage': self.business_stage,
            'business_field': self.business_field,
            'target_type': self.target_type,
            'is_veteran': self.is_veteran,
            'is_disabled': self.is_disabled,
            'company_name': self.company_name,
            'additional_context': self.additional_context
        }

@dataclass
class Document:
    """RAG 문서"""
    id: str
    text: str
    metadata: Dict
    embedding: Optional[np.ndarray] = None

@dataclass
class AgentThought:
    """에이전트 사고 과정"""
    agent_name: str
    timestamp: str
    thought: str
    action: str
    result: str
    confidence: float

@dataclass
class MatchResult:
    """매칭 결과"""
    id: str
    title: str
    data_type: str
    match_score: float
    match_reasons: List[str]
    warnings: List[str]
    deadline: str
    detail_url: str
    priority: str
    summary: str
    raw_data: Dict = field(default_factory=dict)
    agent_reasoning: List[AgentThought] = field(default_factory=list)
