"""
state.py — AgentState 업데이트 버전

변경사항:
  - intent: Dict[str, Any]  추가 (intent_classifier_node 결과)
  - parsed_query에 region, industry 필드가 planner_node에서 채워짐 (문서 변경 없음)
"""

from __future__ import annotations

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    question: str
    user_question: str
    chat_history: List[Dict[str, Any]]
    expanded_query: str
    plan: List[Dict[str, Any]]
    reasoning_trace: List[str]
    retrieved_docs: List[Dict[str, Any]]
    reranked_docs: List[Dict[str, Any]]
    filtered_docs: List[Dict[str, Any]]
    final_docs: List[Dict[str, Any]]
    doc_answer_triplets: List[Dict[str, Any]]
    completeness_check: Dict[str, Any]
    followup_docs: List[Dict[str, Any]]
    answer: str
    parsed_query: Dict[str, Any]
    intent: Dict[str, Any]          # NEW: intent_classifier_node 결과
    selected_doc_types: List[str]   # Optional UI override
    profile_constraints: Dict[str, Any]
    skip_intent_classifier: bool


def init_state(
    question: str,
    selected_doc_types: List[str] | None = None,
    profile_constraints: Dict[str, Any] | None = None,
    chat_history: List[Dict[str, Any]] | None = None,
    skip_intent_classifier: bool = False,
) -> AgentState:
    return AgentState(
        question=question,
        user_question=question,
        chat_history=list(chat_history or []),
        expanded_query=question,
        plan=[],
        reasoning_trace=[],
        retrieved_docs=[],
        reranked_docs=[],
        filtered_docs=[],
        final_docs=[],
        doc_answer_triplets=[],
        completeness_check={},
        followup_docs=[],
        answer="",
        parsed_query={},
        intent={},                  # NEW
        selected_doc_types=list(selected_doc_types or []),
        profile_constraints=dict(profile_constraints or {}),
        skip_intent_classifier=bool(skip_intent_classifier),
    )
