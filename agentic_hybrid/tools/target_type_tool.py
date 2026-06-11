from __future__ import annotations

from typing import Any, Dict, List, Optional

from agentic_hybrid.tools.semantic_similarity import cosine_similarity

_UI_TARGET_TYPE_SYNONYMS: Dict[str, List[str]] = {
    "청소년": ["청소년", "청년", "10대", "고등학생"],
    "대학생": ["대학생", "대학(원)생", "대학원생", "재학생", "휴학생", "졸업예정자"],
    "일반인": ["일반인", "일반", "누구나", "전체", "무관", "제한없음", "제한 없음"],
}

_UNRESTRICTED_KEYWORDS = {
    "누구나",
    "전체",
    "제한없음",
    "제한 없음",
    "무관",
    "전 국민",
    "전국민",
}


def _normalize_target_type(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    for label, synonyms in _UI_TARGET_TYPE_SYNONYMS.items():
        if text == label or any(token in text for token in synonyms):
            return label
    return text


def passes_target_type_constraint(
    user_target_type: Optional[str],
    metadata: Dict[str, Any],
    embedding_model_name: Optional[str] = None,
    similarity_threshold: float = 0.42,
) -> bool:
    if not user_target_type:
        return True

    normalized_target = _normalize_target_type(str(user_target_type))
    if not normalized_target:
        return True

    aply_trgt = str(
        metadata.get("aply_trgt")
        or metadata.get("apply_target")
        or metadata.get("apply_target_desc")
        or metadata.get("biz_supt_trgt_info")
        or metadata.get("target")
        or metadata.get("cond[aply_trgt::LIKE]")
        or ""
    ).strip()
    if aply_trgt:
        if normalized_target == "일반인":
            return any(token in aply_trgt for token in _UI_TARGET_TYPE_SYNONYMS["일반인"]) or not any(
                token in aply_trgt for token in ["청소년", "대학생"]
            )
        return any(token in aply_trgt for token in _UI_TARGET_TYPE_SYNONYMS.get(normalized_target, []))

    blob = " ".join(
        [
            str(metadata.get("target_type") or ""),
            str(metadata.get("support_target") or ""),
            str(metadata.get("target") or ""),
            str(metadata.get("biz_supt_trgt_info") or ""),
            str(metadata.get("apply_target") or ""),
            str(metadata.get("apply_target_desc") or ""),
            str(metadata.get("qualification") or ""),
            str(metadata.get("eligibility") or ""),
            str(metadata.get("title") or ""),
            str(metadata.get("text") or metadata.get("content") or ""),
        ]
    ).strip()
    if not blob:
        return True

    if any(keyword in blob for keyword in _UNRESTRICTED_KEYWORDS):
        return True

    wanted_tokens = _UI_TARGET_TYPE_SYNONYMS.get(normalized_target, [normalized_target])
    if any(token in blob for token in wanted_tokens):
        return True

    if normalized_target == "일반인":
        return True

    model_name = embedding_model_name or "BM-K/KoSimCSE-roberta"
    try:
        similarity = cosine_similarity(normalized_target, blob, model_name)
        return similarity >= similarity_threshold
    except Exception:
        return False
