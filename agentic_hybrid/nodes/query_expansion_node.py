from __future__ import annotations

from typing import Any, Dict, Optional

def _llm_complete(llm: Any, prompt: str, system_prompt: Optional[str] = None) -> str:
    if llm is None:
        return ""
    if hasattr(llm, "complete"):
        return str(llm.complete(prompt, system_prompt=system_prompt)).strip()
    if hasattr(llm, "generate"):
        return str(llm.generate(prompt, system_prompt or "", max_tokens=500)).strip()
    return ""


def query_expansion_node(state: Dict[str, Any], llm: Any, cfg: Any = None) -> Dict[str, Any]:
    use_hyde = True if cfg is None else bool(getattr(cfg, "USE_HYDE", True))
    question = str(state.get("question", "")).strip()
    reasoning_trace = list(state.get("reasoning_trace", []))

    expanded = question
    if use_hyde and llm is not None:
        prompt = f"질문에 대한 이상적인 정책 추천 답변을 4~6문장으로 작성하세요. 검색 확장용입니다.\n질문: {question}"
        hyde = _llm_complete(llm, prompt, "Write a retrieval-oriented hypothetical answer.")
        if hyde:
            expanded = f"{question}\n\n[HyDE]\n{hyde}"
            reasoning_trace.append("query_expansion: HyDE applied")
        else:
            reasoning_trace.append("query_expansion: HyDE failed, fallback to original query")
    else:
        reasoning_trace.append("query_expansion: HyDE disabled")

    out = dict(state)
    out["expanded_query"] = expanded
    out["reasoning_trace"] = reasoning_trace
    return out
