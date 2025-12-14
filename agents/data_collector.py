# agents/data_collector.py

from datetime import datetime, timedelta
from typing import List, Dict, Optional
import requests

from agents.base import AgenticAgent
from models.enums import APIEndpoint
from config.settings import Config


class DataCollectionAgent(AgenticAgent):
    """공공API 데이터 수집 에이전트"""

    def __init__(self, service_key: str, llm_client: Optional[object] = None):
        super().__init__("DataCollector", llm_client)
        self.service_key = service_key
        self.base_url = Config.BASE_URL
        print(f"[DEBUG] SERVICE_KEY 설정됨? {bool(self.service_key)}")

    def collect_all(self, max_pages: int = 3, days_range: int = 180) -> Dict[str, List[Dict]]:
        """
        모든 엔드포인트 데이터 수집
        
        Args:
            max_pages: 페이지당 최대 수집 수
            days_range: 공고 기간 범위 (기본: 180일 = 6개월)
        """
        self.think("API 데이터 수집 시작", action="여러 엔드포인트 호출", confidence=0.9)

        data = {
            # K-Startup 공고/사업/콘텐츠/통계
            "announcements": self._fetch_endpoint(
                APIEndpoint.ANNOUNCEMENT, 
                max_pages
            ),
            "business": self._fetch_endpoint(APIEndpoint.BUSINESS, max_pages),
            "content": self._fetch_endpoint(APIEndpoint.CONTENT, max_pages),
            "statistical": self._fetch_endpoint(APIEndpoint.STATISTICAL, max_pages),
            # 창업에듀 강좌
            "edu_lectures": self._fetch_endpoint(APIEndpoint.EDU_LECTURE, max_pages),
            # 창업공간/센터
            "spaces": self._fetch_endpoint(APIEndpoint.SLP_SPACE, max_pages),
            "centers": self._fetch_endpoint(APIEndpoint.SLP_CENTER, max_pages),
            # 창업기업 확인서 (제품/기업)
            "products": self._fetch_endpoint(APIEndpoint.CERT_PRODUCT, max_pages),
            "corporates": self._fetch_endpoint(APIEndpoint.CERT_CORPORATE, max_pages),
            # 주관기관 정보
            "institutions": self._fetch_endpoint(APIEndpoint.INSTITUTION, max_pages),
        }

        total = sum(len(v) for v in data.values())
        self.think("데이터 수집 완료", result=f"총 {total}개", confidence=1.0)

        return data

    def _fetch_endpoint(
        self, 
        endpoint: APIEndpoint, 
        max_pages: int,
        days_range: int = 180
    ) -> List[Dict]:
        """
        특정 엔드포인트 데이터 수집
        
        Args:
            endpoint: API 엔드포인트
            max_pages: 최대 페이지 수
            days_range: 공고 날짜 필터 범위 (일 단위)
        """
        items: List[Dict] = []
        print(f"   📡 {endpoint.name} 수집 중...", end=" ")

        

        for page in range(1, max_pages + 1):
            url = f"{self.base_url}{endpoint.value}"
            params = {
                "serviceKey": self.service_key,
                "page": page,
                "perPage": 100,
                "returnType": "json",
            }

           

            try:
                response = requests.get(url, params=params, timeout=Config.TIMEOUT)

                if response.status_code != 200:
                    print(f"\n   ⚠ HTTP {response.status_code}: {response.text[:200]}")
                    break

                json_data = response.json()
                top_data = json_data.get("data", None)
                page_items: List[Dict] = []

                if isinstance(top_data, dict):
                    inner = top_data.get("data", [])
                    if isinstance(inner, list):
                        page_items = inner
                elif isinstance(top_data, list):
                    page_items = top_data

                if not page_items:
                    break

                items.extend(page_items)

                total_count = json_data.get("totalCount", None)
                if isinstance(total_count, int) and total_count > 0:
                    if len(items) >= total_count:
                        break

            except Exception as e:
                print(f"\n   ⚠ 오류: {e}")
                break

        print(f"{len(items)}개 ✅")


        # 타입 태깅
        TYPE_MAP = {
            APIEndpoint.ANNOUNCEMENT: "announcement",
            APIEndpoint.BUSINESS: "business",
            APIEndpoint.CONTENT: "content",
            APIEndpoint.EDU_LECTURE: "lecture",
            APIEndpoint.SLP_SPACE: "space",
            APIEndpoint.SLP_CENTER: "center",
            APIEndpoint.CERT_PRODUCT: "product",
            APIEndpoint.CERT_CORPORATE: "corporate",
            APIEndpoint.STATISTICAL: "statistical",
            APIEndpoint.INSTITUTION: "institution",
        }

        human_type = TYPE_MAP.get(endpoint, "기타")
        for item in items:
            item.setdefault("type", human_type)

        return items