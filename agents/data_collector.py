"""
데이터 수집 에이전트
"""
import requests
from typing import List, Dict, Optional
from agents.base import AgenticAgent
from models.enums import APIEndpoint
from config.settings import Config

class DataCollectionAgent(AgenticAgent):
    """공공API 데이터 수집 에이전트"""
    
    def __init__(self, service_key: str, llm_client: Optional[object] = None):
        super().__init__("DataCollector", llm_client)
        self.service_key = service_key
        self.base_url = Config.BASE_URL
    
    def collect_all(self, max_pages: int = 3) -> Dict[str, List[Dict]]:
        """모든 엔드포인트 데이터 수집"""
        self.think("API 데이터 수집 시작", action="여러 엔드포인트 호출", confidence=0.9)
        
        data = {
            # K-Startup 공고/사업/콘텐츠/통계
            'announcements': self._fetch_endpoint(APIEndpoint.ANNOUNCEMENT, max_pages),
            'business': self._fetch_endpoint(APIEndpoint.BUSINESS, max_pages),
            'content': self._fetch_endpoint(APIEndpoint.CONTENT, max_pages),
            'statistical': self._fetch_endpoint(APIEndpoint.STATISTICAL, max_pages),
            
            # 창업에듀 강좌
            'edu_lectures': self._fetch_endpoint(APIEndpoint.EDU_LECTURE, max_pages),
            
            # 창업공간/센터
            'spaces': self._fetch_endpoint(APIEndpoint.SLP_SPACE, max_pages),
            'centers': self._fetch_endpoint(APIEndpoint.SLP_CENTER, max_pages),
            
            # 창업기업 확인서 (제품/기업)
            'products': self._fetch_endpoint(APIEndpoint.CERT_PRODUCT, max_pages),
            'corporates': self._fetch_endpoint(APIEndpoint.CERT_CORPORATE, max_pages),
            
            # 주관기관 정보
            'institutions': self._fetch_endpoint(APIEndpoint.INSTITUTION, max_pages),
        }
        
        total = sum(len(v) for v in data.values())
        self.think(f"데이터 수집 완료", result=f"총 {total}개", confidence=1.0)
        
        return data
    
    def _fetch_endpoint(self, endpoint: APIEndpoint, max_pages: int) -> List[Dict]:
        """특정 엔드포인트 데이터 수집"""
        items = []
        print(f"   📡 {endpoint.name} 수집 중...", end=" ")
        
        for page in range(1, max_pages + 1):
            url = f"{self.base_url}{endpoint.value}"
            params = {
                "serviceKey": self.service_key,
                "page": page,
                "perPage": 100,
                "returnType": "json"
            }
            
            try:
                response = requests.get(url, params=params, timeout=Config.TIMEOUT)
                
                if response.status_code != 200:
                    print(f"\n   ⚠ HTTP {response.status_code}: {response.text[:200]}")
                    break
                
                data = response.json()
                
                if "data" not in data:
                    break
                
                page_items = data.get("data", [])
                if not page_items:
                    break
                
                items.extend(page_items)
                
                # 마지막 페이지 확인
                total_count = data.get("totalCount", None)
                if total_count and len(items) >= total_count:
                    break
            
            except Exception as e:
                print(f"\n   ⚠ 오류: {e}")
                break
        
        print(f"{len(items)}개 ✅")
        return items
