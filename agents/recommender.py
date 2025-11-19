"""
추천 에이전트
"""
from typing import List, Dict, Optional
from collections import defaultdict
from datetime import datetime
from agents.base import AgenticAgent
from models.data import UserProfile, MatchResult
from utils.date import format_deadline

class RecommendationAgent(AgenticAgent):
    """추천 생성 에이전트"""
    
    def __init__(self, llm_client: Optional[object] = None):
        super().__init__("Recommender", llm_client)
    
    def create_report(
        self, 
        matches: List[MatchResult], 
        profile: UserProfile, 
        llm_summary: str = None, 
        top_n: int = 10
    ) -> Dict:
        """최종 추천 리포트 생성"""
        self.think("최종 리포트 생성", action=f"상위 {top_n}개", confidence=1.0)
        
        if not matches:
            return {
                'status': 'NO_MATCH',
                'message': '매칭 결과 없음',
                'recommendations': []
            }
        
        top_matches = matches[:top_n]
        recommendations = []
        
        for idx, match in enumerate(top_matches, 1):
            rec = {
                'rank': idx,
                'id': match.id,
                'title': match.title,
                'data_type': match.data_type,
                'match_score': round(match.match_score, 1),
                'priority': match.priority,
                'reasons': match.match_reasons,
                'deadline': format_deadline(match.deadline),
                'detail_url': match.detail_url,
                'summary': match.summary
            }
            recommendations.append(rec)
        
        # 통계
        by_type = defaultdict(int)
        by_priority = defaultdict(int)
        for m in matches:
            by_type[m.data_type] += 1
            by_priority[m.priority] += 1
        
        report = {
            'status': 'SUCCESS',
            'profile': profile.to_dict(),
            'total_matches': len(matches),
            'by_type': dict(by_type),
            'by_priority': dict(by_priority),
            'recommendations': recommendations,
            'llm_summary': llm_summary,
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self.think("리포트 생성 완료", result=f"{len(recommendations)}개", confidence=1.0)
        return report
