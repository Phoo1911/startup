"""
데이터 모델 정의
"""
from dataclasses import dataclass, field as dc_field
from typing import List, Dict, Optional, Any
import numpy as np


from dataclasses import dataclass, field
from typing import List

 


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
    # ✅ 여기 한 줄
    desired_data_types: List[str] = field(
        default_factory=lambda: ["announcement", "business"]
    )

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
            'additional_context': self.additional_context,
            'desired_data_types': self.desired_data_types,  # ✅ 잘 넣었음
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
    id: str
    title: str
    data_type: str
    region: str
    field: str
    deadline: str
    status: str
    apply_period: str
    priority: str
    match_score: float
    score: float = 0.0
    reasons: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)




@dataclass
class QuestionPlan:
    """질문 분석 결과 (프롬프트 플래너용)"""
    intents: List[str] = field(default_factory=list)       # ["RECOMMEND", "REQUIRED_DOCS", ...]
    data_types: List[str] = field(default_factory=list)    # ["announcement", "business", ...]
    answer_style: str = "auto"     