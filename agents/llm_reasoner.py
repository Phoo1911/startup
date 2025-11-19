"""
LLM 추론 에이전트
"""
from typing import List
from agents.base import AgenticAgent
from models.data import UserProfile, MatchResult

class LLMReasoningAgent(AgenticAgent):
    """LLM 기반 추론 에이전트"""
    
    def __init__(self, llm_client: object):
        super().__init__("LLMReasoner", llm_client)
    
    def enhance_matches(self, matches: List[MatchResult], profile: UserProfile) -> List[MatchResult]:
        """매칭 결과에 LLM 분석 추가"""
        if not self.llm or not matches:
            return matches
        
        self.think("LLM 분석 시작", action="상위 5개 분석", confidence=0.85)
        
        for match in matches[:5]:
            enhanced = self._analyze_with_llm(match, profile)
            if enhanced and enhanced != "[LLM 비활성화]":
                match.match_reasons.append(f"🧠 {enhanced}")
        
        self.think("LLM 분석 완료", result="인사이트 추가", confidence=0.9)
        return matches
    
    def _analyze_with_llm(self, match: MatchResult, profile: UserProfile) -> str:
        """개별 매칭에 대한 LLM 분석"""
        prompt = f"""
사용자({profile.age}세, {profile.region}, {profile.business_field})에게 
'{match.title}'이 적합한 이유를 한 문장으로 설명해주세요.
        """.strip()
        
        try:
            return self.llm.generate(prompt, "간결하게 답변하세요.", max_tokens=100).strip()
        except:
            return ""
    
    def generate_summary(self, matches: List[MatchResult], profile: UserProfile) -> str:
        """전체 매칭 결과 요약"""
        if not self.llm or not matches:
            return f"{profile.name}님께 {len(matches)}개 지원사업 추천"
        
        top = matches[:3]
        matches_text = "\n".join([f"{i+1}. {m.title}" for i, m in enumerate(top)])
        
        prompt = f"""
{profile.name}님({profile.age}세, {profile.region}, {profile.business_field})에게 
다음을 추천합니다:

{matches_text}

2-3문장으로 격려 메시지를 작성해주세요.
        """.strip()
        
        try:
            return self.llm.generate(prompt, "친근한 상담사처럼", max_tokens=200)
        except:
            return f"{profile.name}님께 적합한 지원사업을 찾았습니다."
