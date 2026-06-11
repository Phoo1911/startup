from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def _llm_complete(llm: Any, prompt: str, system_prompt: Optional[str] = None) -> str:
    if llm is None:
        return ""
    if hasattr(llm, "complete"):
        return str(llm.complete(prompt, system_prompt=system_prompt)).strip()
    if hasattr(llm, "generate"):
        return str(llm.generate(prompt, system_prompt or "", max_tokens=400)).strip()
    if callable(llm):
        return str(llm(prompt)).strip()
    return ""


def _safe_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def _clean_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "unknown", "n/a"}:
        return None
    return text


def _dedupe_strings(values: List[Any]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        text = _clean_text(value)
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out



def _llm_parse(question: str, llm: Any) -> Dict[str, Any]:
    """
    LLM-only query parsing. Extracts structured slots needed for startup-support retrieval and filtering.
    No rule-based fallback.
    """
    prompt = (
        "Extract only structured slots needed for startup-support retrieval and filtering. Output JSON only.\n"
        f"Question: {question}\n"
        'schema={"age":int|null,"startup_years":number|null,"industry":string|null,'
        '"special_conditions":[string],"region":string|null,"target_type":string|null,'
        '"time_preference":"open_now|include_upcoming"|null}\n'
        "Rules:\n"
        "- age must be a literal age number only.\n"
        "- startup_years may be 0 for pre-startup users.\n"
        "- region must be null if no region is stated.\n"
        "- target_type should be filled only when explicitly stated.\n"
        "- time_preference is open_now for currently open items, include_upcoming for upcoming items too, otherwise null.\n"
        "- Do not classify intent labels or doc_types here.\n"
        "- Do not guess eligibility conditions that are not stated in the question."
    )
    raw = _llm_complete(llm, prompt, "JSON only.")
    if not raw:
        return {}
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _merge_parsed_values(
    *,
    llm_payload: Dict[str, Any],
    profile_constraints: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, str]]:
    """
    Merge LLM extracted values with profile constraints.
    Profile constraints take highest priority, then LLM results.
    """
    parsed: Dict[str, Any] = {
        "age": None,
        "startup_years": None,
        "industry": None,
        "special_conditions": [],
        "region": None,
        "target_type": None,
        "time_preference": "open_now",
    }
    source: Dict[str, str] = {}

    def assign(key: str, value: Any, src: str) -> None:
        parsed[key] = value
        source[key] = src

    # Profile constraints have highest priority
    for key in ("age", "startup_years", "industry", "region", "target_type", "time_preference"):
        value = profile_constraints.get(key)
        cleaned = _clean_text(value) if key not in {"age", "startup_years"} else value
        if cleaned not in (None, "", []):
            assign(key, cleaned, "profile")

    profile_specials = profile_constraints.get("special_conditions")
    if isinstance(profile_specials, list) and profile_specials:
        assign("special_conditions", _dedupe_strings(profile_specials), "profile")

    # LLM results fill remaining slots
    llm_age = _safe_int(llm_payload.get("age"))
    if llm_age is not None and "age" not in source:
        assign("age", llm_age, "llm")

    llm_sy = _safe_float(llm_payload.get("startup_years"))
    if llm_sy is not None and "startup_years" not in source:
        assign("startup_years", llm_sy, "llm")

    for key in ("industry", "region", "target_type"):
        value = _clean_text(llm_payload.get(key))
        if value is not None and key not in source:
            assign(key, value, "llm")


    tp = _clean_text(llm_payload.get("time_preference"))
    if tp in {"open_now", "include_upcoming"} and "time_preference" not in source:
        assign("time_preference", tp, "llm")

    llm_specials = llm_payload.get("special_conditions")
    if isinstance(llm_specials, list) and llm_specials and "special_conditions" not in source:
        assign("special_conditions", _dedupe_strings(llm_specials), "llm")

    parsed["_extraction_source"] = source
    return parsed, source


def planner_node(state: Dict[str, Any], llm: Any, cfg: Any = None) -> Dict[str, Any]:
    """
    LLM-only query planner. No rule-based extraction fallback.
    Extracts structured query attributes (age, startup_years, industry, region, etc.)
    using LLM with profile constraints override.
    """
    question = str(state.get("question", "")).strip()
    reasoning_trace = list(state.get("reasoning_trace", []))
    profile_constraints = dict(state.get("profile_constraints") or {})
    intent = state.get("intent") or {}

    use_planner = True if cfg is None else bool(getattr(cfg, "USE_AGENTIC_PLANNER", True))
    
    llm_payload: Dict[str, Any] = {}
    if llm is not None and use_planner:
        llm_payload = _llm_parse(question, llm)

    parsed, source = _merge_parsed_values(
        llm_payload=llm_payload,
        profile_constraints=profile_constraints,
    )

    plan = [
        {"step": "retrieve", "status": "pending"},
        {"step": "filter_age", "status": "pending"},
        {"step": "filter_deadline", "status": "pending"},
        {"step": "filter_startup", "status": "pending"},
        {"step": "filter_region", "status": "pending"},
        {"step": "filter_field", "status": "pending"},
        {"step": "filter_target", "status": "pending"},
        {"step": "generate", "status": "pending"},
    ]

    reasoning_trace.append(
        "planner: parsed (llm-only) "
        f"age={parsed.get('age')}[{source.get('age', 'none')}], "
        f"startup_years={parsed.get('startup_years')}[{source.get('startup_years', 'none')}], "
        f"industry={parsed.get('industry')}[{source.get('industry', 'none')}], "
        f"region={parsed.get('region')}[{source.get('region', 'none')}], "
        f"target_type={parsed.get('target_type')}[{source.get('target_type', 'none')}], "
        f"special={parsed.get('special_conditions', [])}[{source.get('special_conditions', 'none')}], "
        f"time_preference={parsed.get('time_preference')}[{source.get('time_preference', 'none')}]"
    )

    out = dict(state)
    out["parsed_query"] = parsed
    out["plan"] = plan
    out["reasoning_trace"] = reasoning_trace
    return out
