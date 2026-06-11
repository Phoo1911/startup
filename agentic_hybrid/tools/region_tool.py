from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from agentic_hybrid.tools.semantic_similarity import cosine_similarity

_REGION_ALIASES: Dict[str, str] = {
    "서울특별시": "서울",
    "부산광역시": "부산",
    "대구광역시": "대구",
    "인천광역시": "인천",
    "광주광역시": "광주",
    "대전광역시": "대전",
    "울산광역시": "울산",
    "세종특별자치시": "세종",
    "세종시": "세종",
    "경기도": "경기",
    "강원특별자치도": "강원",
    "강원도": "강원",
    "충청북도": "충북",
    "충청남도": "충남",
    "전라북도": "전북",
    "전북특별자치도": "전북",
    "전라남도": "전남",
    "경상북도": "경북",
    "경상남도": "경남",
    "제주특별자치도": "제주",
    "제주도": "제주",
}

_NATIONWIDE_KEYWORDS = {"전국", "전체", "제한없음", "전 지역", "전지역"}
_REGION_TOKENS: List[str] = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]


def _normalize(region: str) -> str:
    region = str(region or "").strip()
    if not region:
        return ""
    return _REGION_ALIASES.get(region, region)


def passes_region_constraint(
    user_region: Optional[str],
    metadata: Dict[str, Any],
    embedding_model_name: Optional[str] = None,
    similarity_threshold: float = 0.38,
) -> bool:
    if not user_region:
        return True

    raw_region = str(
        metadata.get("region")
        or metadata.get("supt_regin")
        or metadata.get("suptRegin")
        or metadata.get("regin_clss")
        or metadata.get("area")
        or metadata.get("location")
        or ""
    ).strip()

    title_text = str(metadata.get("title") or "")
    body_text = str(metadata.get("text") or metadata.get("content") or "")
    addr_text = str(metadata.get("addr") or metadata.get("address") or "")
    local_gov = str(metadata.get("lwdg_nm") or "")
    blended_text = f"{raw_region} {title_text} {body_text} {addr_text} {local_gov}".strip()

    if any(kw in blended_text for kw in _NATIONWIDE_KEYWORDS):
        return True

    user_norm = _normalize(user_region)

    mentioned_regions: List[str] = []
    for token in _REGION_TOKENS:
        if re.search(re.escape(token), blended_text):
            mentioned_regions.append(_normalize(token))
    if mentioned_regions:
        return user_norm in mentioned_regions

    parts = re.split(r"[,/·\s]+", raw_region)
    for part in parts:
        part_norm = _normalize(part)
        if part_norm and (user_norm in part_norm or part_norm in user_norm):
            return True

    if not raw_region and not addr_text and not title_text and not body_text:
        return True

    model_name = embedding_model_name or "BM-K/KoSimCSE-roberta"
    try:
        similarity = cosine_similarity(user_norm, blended_text, model_name)
        return similarity >= similarity_threshold
    except Exception:
        return False
