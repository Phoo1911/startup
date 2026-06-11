"""
tools/field_tool.py — 지원 분야(업종) 필터

메타데이터의 field 필드(supt_biz_clsfc)와 사용자 업종/산업을 비교한다.

특징:
  - 키워드 동의어 테이블로 "AI" → ["AI", "인공지능", "소프트웨어"] 확장 매칭
  - "업종 무관", "전 분야" 등은 항상 통과
  - 사용자 업종 미입력 → 항상 통과
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_hybrid.tools.semantic_similarity import cosine_similarity

# 업종 동의어 테이블 — 키: 정규화 레이블, 값: 매칭 키워드 목록
_FIELD_SYNONYMS: Dict[str, List[str]] = {
    "AI": ["AI", "인공지능", "머신러닝", "딥러닝", "machine learning"],
    "바이오": ["바이오", "생명공학", "헬스케어", "의료", "의약", "bio"],
    "핀테크": ["핀테크", "금융", "fintech", "블록체인"],
    "에듀테크": ["에듀테크", "교육", "edutech", "edtech", "학습"],
    "제조": ["제조", "manufacturing", "하드웨어", "생산"],
    "콘텐츠": ["콘텐츠", "미디어", "게임", "엔터테인먼트", "문화"],
    "IT": ["IT", "소프트웨어", "플랫폼", "SaaS", "앱", "서비스"],
    "농업": ["농업", "농식품", "스마트팜", "agtech"],
    "환경": ["환경", "그린", "ESG", "클린테크", "탄소"],
    "물류": ["물류", "유통", "SCM", "공급망"],
    "패션": ["패션", "의류", "뷰티"],
    "부동산": ["부동산", "프롭테크", "건설"],
    "여행": ["여행", "관광", "숙박"],
    "푸드": ["푸드", "식품", "외식", "F&B"],
}

_UNRESTRICTED_KEYWORDS = {
    "전 분야", "전분야", "업종무관", "업종 무관", "제한없음", "전체", "모든 분야"
}


def _expand_keywords(industry: str) -> List[str]:
    """사용자 입력 업종을 동의어 포함 키워드 목록으로 확장."""
    industry_lower = industry.lower()
    keywords = [industry, industry_lower]
    for label, synonyms in _FIELD_SYNONYMS.items():
        if any(s.lower() in industry_lower or industry_lower in s.lower() for s in synonyms):
            keywords.extend(synonyms)
    return list(set(keywords))


def passes_field_constraint(
    user_industry: Optional[str],
    metadata: Dict[str, Any],
    embedding_model_name: Optional[str] = None,
    similarity_threshold: float = 0.45,
) -> bool:
    """
    사용자 업종이 정책 지원 분야에 해당하는지 확인한다.

    Parameters
    ----------
    user_industry : 사용자가 입력한 업종 문자열 (None이면 항상 True)
    metadata      : 문서 메타데이터. ``field`` 키를 참조한다.

    Returns
    -------
    bool — True이면 이 문서는 분야 조건을 통과한다.
    """
    if not user_industry:
        return True

    raw_field_parts = [
        str(metadata.get("field") or "").strip(),
        str(metadata.get("supt_biz_clsfc") or "").strip(),
        str(metadata.get("category_cd") or "").strip(),
        str(metadata.get("biz_category_cd") or "").strip(),
        str(metadata.get("character") or "").strip(),
        str(metadata.get("supt_biz_chrct") or "").strip(),
        str(metadata.get("intro") or "").strip(),
        str(metadata.get("supt_biz_intrd_info") or "").strip(),
        str(metadata.get("content") or "").strip(),
        str(metadata.get("cntn") or "").strip(),
        str(metadata.get("keywords") or "").strip(),
        str(metadata.get("kywrd") or "").strip(),
        str(metadata.get("manu_category") or "").strip(),
        str(metadata.get("manu_lclss") or "").strip(),
        str(metadata.get("manu_mclss") or "").strip(),
        str(metadata.get("manu_sclss") or "").strip(),
        str(metadata.get("inds_clsf_clss_cd") or "").strip(),
        str(metadata.get("inds_clsfc_clss_cd") or "").strip(),
        str(metadata.get("title") or "").strip(),
    ]
    raw_field = " ".join(part for part in raw_field_parts if part).strip()

    # 분야 정보 없음 → 제한 없는 것으로 간주
    if not raw_field:
        return True

    # "전 분야" 등 → 항상 통과
    if any(kw in raw_field for kw in _UNRESTRICTED_KEYWORDS):
        return True

    # 동의어 확장 후 포함 여부 검사
    keywords = _expand_keywords(user_industry)
    raw_lower = raw_field.lower()
    if any(kw.lower() in raw_lower for kw in keywords):
        return True

    model_name = embedding_model_name or "BM-K/KoSimCSE-roberta"
    try:
        similarity = cosine_similarity(user_industry, raw_field, model_name)
        return similarity >= similarity_threshold
    except Exception:
        return False
