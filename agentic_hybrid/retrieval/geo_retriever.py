"""
retrieval/geo_retriever.py — Haversine 거리 기반 공간/센터 검색

외부 의존성 없이 순수 수학으로 가장 가까운 창업공간·센터를 찾는다.

사용 예:
    gr = GeoRetriever(documents)          # VectorRetriever.documents 전달
    results = gr.search_nearby(
        lat=37.5172, lng=127.0473,        # 강남구청 좌표
        radius_km=3.0,
        doc_types=["space", "center"],
        top_k=5,
    )
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

_EARTH_RADIUS_KM = 6371.0

# 한국 주요 도시 좌표 (주소 → 위경도 변환 폴백용)
_CITY_COORDS: Dict[str, tuple[float, float]] = {
    "서울": (37.5665, 126.9780),
    "강남": (37.5172, 127.0473),
    "홍대": (37.5563, 126.9236),
    "판교": (37.3943, 127.1111),
    "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526),
    "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114),
    "세종": (36.4800, 127.2890),
    "수원": (37.2636, 127.0286),
    "성남": (37.4449, 127.1389),
    "제주": (33.4996, 126.5312),
}


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """두 위경도 좌표 사이의 Haversine 거리(km)를 반환한다."""
    r = _EARTH_RADIUS_KM
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def city_to_coords(city_name: str) -> Optional[tuple[float, float]]:
    """도시/지역명을 위경도로 변환한다. 미등록 지역이면 None 반환."""
    for key, coords in _CITY_COORDS.items():
        if key in city_name or city_name in key:
            return coords
    return None


class GeoRetriever:
    """
    위경도가 있는 문서(공간·센터)에 대한 거리 기반 검색기.

    Parameters
    ----------
    documents : VectorRetriever.documents 또는 동일 포맷의 dict 리스트
    """

    def __init__(self, documents: List[Dict[str, Any]]) -> None:
        # latitude/longitude가 있는 문서만 인덱싱
        self._geo_docs: List[Dict[str, Any]] = [
            doc for doc in documents
            if self._has_valid_coords(doc)
        ]

    @staticmethod
    def _has_valid_coords(doc: Dict[str, Any]) -> bool:
        meta = doc.get("metadata", {}) or {}
        try:
            lat = float(meta.get("latitude") or 0)
            lng = float(meta.get("longitude") or 0)
            return lat != 0.0 and lng != 0.0
        except (ValueError, TypeError):
            return False

    def search_nearby(
        self,
        lat: float,
        lng: float,
        radius_km: float = 5.0,
        doc_types: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        주어진 좌표에서 radius_km 이내의 문서를 거리 오름차순으로 반환.

        Parameters
        ----------
        lat, lng    : 기준 위경도
        radius_km   : 검색 반경 (기본 5km)
        doc_types   : 필터할 문서 타입 목록. None이면 전체 검색
                      예: ["space", "center"]
        top_k       : 반환 최대 개수
        """
        results: List[Dict[str, Any]] = []

        for doc in self._geo_docs:
            meta = doc.get("metadata", {}) or {}

            # 타입 필터
            if doc_types and meta.get("type") not in doc_types:
                continue

            try:
                doc_lat = float(meta["latitude"])
                doc_lng = float(meta["longitude"])
            except (KeyError, ValueError, TypeError):
                continue

            dist = haversine_km(lat, lng, doc_lat, doc_lng)
            if dist <= radius_km:
                item = dict(doc)
                item["distance_km"] = round(dist, 3)
                item["score"] = max(0.0, 1.0 - dist / radius_km)  # 거리 기반 점수
                item["retrieval_source"] = "geo"
                results.append(item)

        results.sort(key=lambda x: x["distance_km"])
        return results[:top_k]

    def search_by_city(
        self,
        city_name: str,
        radius_km: float = 5.0,
        doc_types: Optional[List[str]] = None,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        도시/지역명으로 검색. 내부적으로 city_to_coords() 변환 후 search_nearby() 호출.
        미등록 지역이면 빈 리스트 반환.
        """
        coords = city_to_coords(city_name)
        if coords is None:
            return []
        return self.search_nearby(*coords, radius_km=radius_km, doc_types=doc_types, top_k=top_k)

    @property
    def indexed_count(self) -> int:
        """위경도 인덱싱된 문서 수."""
        return len(self._geo_docs)
