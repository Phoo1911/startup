# agents/data_collector.py

from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

from legacy_agents.base import AgenticAgent
from config.settings import Config
from models.enums import APIEndpoint


class DataCollectionAgent(AgenticAgent):
    """Public API data collection agent."""

    def __init__(self, service_key: str, llm_client: Optional[object] = None):
        super().__init__("DataCollector", llm_client)
        self.service_key = service_key
        self.base_url = Config.BASE_URL
        print(f"[DEBUG] SERVICE_KEY configured? {bool(self.service_key)}")

    def collect_all(self, max_pages: int = 3, days_range: int = 90) -> Dict[str, List[Dict]]:
        """Collect data from all configured endpoints."""
        self.think("API data collection start", action="call multiple endpoints", confidence=0.9)

        data = {
            "announcements": self._fetch_endpoint(APIEndpoint.ANNOUNCEMENT, max_pages, days_range=days_range),
            "business": self._fetch_endpoint(APIEndpoint.BUSINESS, max_pages, days_range=days_range),
            "content": self._fetch_endpoint(APIEndpoint.CONTENT, max_pages, days_range=days_range),
            "statistical": self._fetch_endpoint(APIEndpoint.STATISTICAL, max_pages, days_range=days_range),
            "edu_lectures": self._fetch_endpoint(APIEndpoint.EDU_LECTURE, max_pages, days_range=days_range),
            "spaces": self._fetch_endpoint(APIEndpoint.SLP_SPACE, max_pages, days_range=days_range),
            "centers": self._fetch_endpoint(APIEndpoint.SLP_CENTER, max_pages, days_range=days_range),
            "products": self._fetch_endpoint(APIEndpoint.CERT_PRODUCT, max_pages, days_range=days_range),
            "corporates": self._fetch_endpoint(APIEndpoint.CERT_CORPORATE, max_pages, days_range=days_range),
            "institutions": self._fetch_endpoint(APIEndpoint.INSTITUTION, max_pages, days_range=days_range),
        }

        total = sum(len(v) for v in data.values())
        self.think("data collection done", result=f"total {total}", confidence=1.0)
        return data

    def _fetch_endpoint(
        self,
        endpoint: APIEndpoint,
        max_pages: int,
        days_range: int = 90,
    ) -> List[Dict]:
        """Collect data from a specific endpoint."""
        items: List[Dict] = []
        print(f"   collecting {endpoint.name}...", end=" ")

        extra_params = self._build_endpoint_params(endpoint, days_range)

        for page in range(1, max_pages + 1):
            url = f"{self.base_url}{endpoint.value}"
            params = {
                "serviceKey": self.service_key,
                "page": page,
                "perPage": 100,
                "returnType": "json",
            }
            params.update(extra_params)

            try:
                response = requests.get(url, params=params, timeout=Config.TIMEOUT)
                if response.status_code != 200:
                    print(f"\n   warning HTTP {response.status_code}: {response.text[:200]}")
                    break

                json_data = response.json()
                top_data = json_data.get("data")
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

                total_count = json_data.get("totalCount")
                if isinstance(total_count, int) and total_count > 0 and len(items) >= total_count:
                    break

            except Exception as exc:
                print(f"\n   warning: {exc}")
                break

        filtered_items = self._filter_recent_items(endpoint, items, days_range)
        if len(filtered_items) != len(items):
            print(f"{len(filtered_items)} items (filtered from {len(items)})")
        else:
            print(f"{len(filtered_items)} items")

        type_map = {
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
        human_type = type_map.get(endpoint, "other")
        for item in filtered_items:
            item.setdefault("type", human_type)

        return filtered_items

    @staticmethod
    def _ymd(days_ago: int = 0) -> str:
        return (datetime.now() - timedelta(days=days_ago)).strftime("%Y%m%d")

    @staticmethod
    def _parse_date_like(raw: object) -> Optional[date]:
        text = str(raw or "").strip()
        if not text:
            return None
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) < 8:
            return None
        try:
            return datetime.strptime(digits[:8], "%Y%m%d").date()
        except ValueError:
            return None

    @staticmethod
    def _parse_year(raw: object) -> Optional[int]:
        text = str(raw or "").strip()
        if not text:
            return None
        digits = "".join(ch for ch in text if ch.isdigit())
        if len(digits) < 4:
            return None
        year = int(digits[:4])
        if 2000 <= year <= 2100:
            return year
        return None

    def _filter_recent_items(
        self,
        endpoint: APIEndpoint,
        items: List[Dict],
        days_range: int,
    ) -> List[Dict]:
        if not items:
            return items

        today = datetime.now().date()
        cutoff = today - timedelta(days=max(days_range, 0))

        if endpoint == APIEndpoint.ANNOUNCEMENT:
            filtered: List[Dict] = []
            for item in items:
                start_d = self._parse_date_like(item.get("pbanc_rcpt_bgng_dt"))
                end_d = self._parse_date_like(item.get("pbanc_rcpt_end_dt"))
                status = str(item.get("rcrt_prgs_yn") or "").strip().upper()

                if status in {"N", "마감", "종료", "CLOSED"}:
                    continue
                if end_d is not None and end_d < today:
                    continue

                anchor_d = end_d or start_d
                if anchor_d is None:
                    continue
                if anchor_d >= cutoff:
                    filtered.append(item)
            return filtered

        if endpoint == APIEndpoint.BUSINESS:
            current_year = today.year
            filtered = []
            for item in items:
                biz_year = self._parse_year(item.get("biz_yr"))
                if biz_year is None or biz_year >= current_year:
                    filtered.append(item)
            return filtered

        if endpoint in {APIEndpoint.CONTENT, APIEndpoint.STATISTICAL}:
            filtered = []
            for item in items:
                reg_d = self._parse_date_like(item.get("fstm_reg_dt"))
                mod_d = self._parse_date_like(item.get("last_mdfcn_dt"))
                anchor_d = mod_d or reg_d
                if anchor_d is not None and anchor_d >= cutoff:
                    filtered.append(item)
            return filtered

        if endpoint == APIEndpoint.EDU_LECTURE:
            filtered = []
            for item in items:
                start_d = self._parse_date_like(item.get("lctr_bgng_dt"))
                end_d = self._parse_date_like(item.get("lctr_end_dt"))
                anchor_d = end_d or start_d
                if anchor_d is None or anchor_d >= cutoff:
                    filtered.append(item)
            return filtered

        return items

    def _build_endpoint_params(self, endpoint: APIEndpoint, days_range: int) -> Dict[str, object]:
        """
        Build endpoint-specific collection filters.

        We do not use every available cond[...] parameter. Most of them are for
        targeted search, not corpus-wide collection. For collection, only high-value
        recency/validity filters are applied.
        """
        params: Dict[str, object] = {}

        # Announcement recency is handled later in the serving pipeline. Applying
        # a collection-time deadline filter can drop the entire corpus when the
        # API field is sparse or stale, so we collect first and filter later.
        if endpoint in {APIEndpoint.CERT_PRODUCT, APIEndpoint.CERT_CORPORATE}:
            params["cond[confmdoc_expr_dt::GTE]"] = self._ymd(0)

        return params
