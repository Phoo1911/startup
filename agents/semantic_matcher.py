"""
의미 기반 매칭 에이전트
"""
from typing import List, Optional
from agents.base import AgenticAgent
from models.data import UserProfile, MatchResult
from utils.text import clean_text
from config.settings import Config

class SemanticMatchingAgent(AgenticAgent):
    """의미 기반 매칭 에이전트"""
    
    def __init__(self, rag_system: object, llm_client: Optional[object] = None):
        super().__init__("SemanticMatcher", llm_client)
        self.rag = rag_system
    
    def match(self, profile: UserProfile, top_k: int = 20) -> List[MatchResult]:
        """프로필 기반 매칭"""
        query = self._profile_to_query(profile)
        self.think("프로필 분석 및 쿼리 생성", action=f"Query: {query[:50]}...", confidence=0.9)
        
        # RAG 검색
        retrieved = self.rag.retrieve(query, top_k=top_k)
        self.think("RAG 검색 완료", result=f"{len(retrieved)}개 검색", confidence=0.95)
        
        matches: List[MatchResult] = []
        
        for doc, similarity in retrieved:
            if similarity < Config.SIMILARITY_THRESHOLD:
                continue
            
            meta = doc.metadata
            doc_region = clean_text(meta.get("region", "")).replace(" ", "")
            user_region = clean_text(profile.region).replace(" ", "")
            
            # 지역 필터링
            if user_region and doc_region:
                if doc_region not in ["전국", "전국단위"] and user_region not in doc_region:
                    continue
            
            match_score = similarity * 100.0
            reasons = self._generate_reasons(doc, profile, similarity)
            
            match = MatchResult(
                id=doc.id,
                title=meta.get('title', '제목없음'),
                data_type=meta.get('type', 'unknown'),
                match_score=match_score,
                match_reasons=reasons,
                warnings=[],
                deadline=meta.get('deadline', ''),
                detail_url=meta.get('detail_url', ''),
                priority=self._get_priority(match_score),
                summary=doc.text[:150],
                raw_data=meta.get('raw', {}),
                agent_reasoning=self.thoughts.copy()
            )
            
            matches.append(match)
        
        self.think("매칭 완료", result=f"{len(matches)}개", confidence=1.0)
        return matches
    
    def _profile_to_query(self, profile: UserProfile) -> str:
        """프로필을 검색 쿼리로 변환"""
        parts = [
            f"지역: {profile.region}",
            f"창업단계: {profile.business_stage}",
            f"사업분야: {profile.business_field}",
            f"대상: {profile.target_type}",
        ]
        
        if profile.is_veteran:
            parts.append("참전유공자")
        if profile.is_disabled:
            parts.append("장애인")
        if profile.additional_context:
            parts.append(profile.additional_context)
        
        return " ".join(parts)
    
    def _generate_reasons(self, doc: object, profile: UserProfile, similarity: float) -> List[str]:
        """매칭 이유 생성"""
        reasons = [f"✓ 의미 유사도: {similarity:.1%}"]
        
        meta = doc.metadata
        doc_region = clean_text(meta.get("region", "")).replace(" ", "")
        user_region = clean_text(profile.region).replace(" ", "")
        
        if user_region and doc_region and user_region in doc_region:
            reasons.append(f"✓ 지역 일치: {profile.region}")
        
        text_lower = doc.text.lower()
        
        if profile.business_field.lower() in text_lower:
            reasons.append(f"✓ 분야 일치: {profile.business_field}")
        
        if profile.target_type.lower() in text_lower:
            reasons.append(f"✓ 대상유형 일치: {profile.target_type}")
        
        return reasons
    
    def _get_priority(self, score: float) -> str:
        """우선순위 계산"""
        if score >= 85:
            return "HIGH"
        elif score >= 70:
            return "MEDIUM"
        else:
            return "LOW"
