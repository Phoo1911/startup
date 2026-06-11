from __future__ import annotations

import re
from typing import Any, Dict, List


ALL_DOC_TYPES: List[str] = [
    "announcement",
    "business",
    "content",
    "statistical",
    "lecture",
    "space",
    "center",
    "product",
    "corporate",
    "institution",
]


_SPACE_HINTS = (
    "공간",
    "센터",
    "입주",
    "사무실",
    "창업공간",
    "보육센터",
    "메이커스페이스",
)
_STATUS_HINTS = ("마감", "접수", "신청", "상태", "언제까지", "기간")
_COMPARE_HINTS = ("비교", "차이", "장단점", "어떤 게", "vs")

_KOREAN_CITIES = (
    "서울",
    "부산",
    "대구",
    "인천",
    "광주",
    "대전",
    "울산",
    "세종",
    "경기",
    "강원",
    "충북",
    "충남",
    "전북",
    "전남",
    "경북",
    "경남",
    "제주",
)


def _contains_any(text: str, hints: tuple[str, ...]) -> bool:
    return any(hint in text for hint in hints)


def _extract_geo(text: str) -> str | None:
    for city in _KOREAN_CITIES:
        if city in text:
            return city
    return None


def _infer_intent_label(question: str, selected_doc_types: List[str]) -> str:
    if selected_doc_types and all(doc_type in {"space", "center"} for doc_type in selected_doc_types):
        return "find_space"
    if _contains_any(question, _SPACE_HINTS):
        return "find_space"
    if _contains_any(question, _COMPARE_HINTS):
        return "compare"
    if _contains_any(question, _STATUS_HINTS):
        return "check_status"
    if "지원" in question or "공고" in question or "사업" in question:
        return "find_policy"
    return "general"


def _infer_doc_types(intent_label: str, selected_doc_types: List[str]) -> List[str]:
    if selected_doc_types:
        return [doc_type for doc_type in selected_doc_types if doc_type in ALL_DOC_TYPES]
    if intent_label == "find_space":
        return ["space", "center"]
    if intent_label in {"find_policy", "check_status", "compare"}:
        return ["announcement", "business"]
    return []


def intent_classifier_node(state: Dict[str, Any], llm: Any = None, cfg: Any = None) -> Dict[str, Any]:
    question = str(state.get("question") or "").strip()
    selected_doc_types = list(state.get("selected_doc_types") or [])
    reasoning_trace = list(state.get("reasoning_trace") or [])

    if state.get("skip_intent_classifier"):
        intent = {
            "label": "general",
            "doc_types": [doc_type for doc_type in selected_doc_types if doc_type in ALL_DOC_TYPES],
            "geo": None,
            "source": "skipped",
        }
        reasoning_trace.append("intent_classifier: skipped by state flag")
    else:
        intent_label = _infer_intent_label(question, selected_doc_types)
        intent = {
            "label": intent_label,
            "doc_types": _infer_doc_types(intent_label, selected_doc_types),
            "geo": _extract_geo(question),
            "source": "rule_based",
        }
        reasoning_trace.append(
            "intent_classifier: "
            f"label={intent['label']}, doc_types={intent['doc_types']}, geo={intent['geo'] or '-'}"
        )

    out = dict(state)
    out["intent"] = intent
    out["reasoning_trace"] = reasoning_trace
    return out
