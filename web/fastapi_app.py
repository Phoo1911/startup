"""
FastAPI backend connected to agentic_hybrid + React SPA static hosting
Run: uvicorn web.fastapi_app:app --reload
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import sys
from dataclasses import replace
from datetime import datetime
import threading
import json
import pickle
import re

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent))
load_dotenv()

from agentic_hybrid.config import apply_experiment_mode, load_config
from agentic_hybrid.graph import build_agentic_graph
from agentic_hybrid.main_agentic import (
    SimpleLLM,
    _detect_text_language,
    _llm_thoughts,
    _to_natural_thoughts_for_lang,
)
from agentic_hybrid.nodes.intent_classifier_node import ALL_DOC_TYPES
from agentic_hybrid.state import init_state
from agentic_hybrid.tools.deadline_tool import _parse_date, passes_deadline_constraint
from agentic_hybrid.tools.semantic_similarity import cosine_similarity
from legacy_core.orchestrator import AgenticOrchestrator as LegacyAgenticOrchestrator
from config.settings import Config


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
ASSETS_DIR = FRONTEND_DIR / "assets"
INDEX_FILE = FRONTEND_DIR / "index.html"
DEFAULT_WEB_MODE = "no_intent_dense"


app = FastAPI(
    title="Agentic Hybrid API",
    description="React + FastAPI + agentic_hybrid",
    version="1.0.0",
)

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


class ProfileRequest(BaseModel):
    name: str = "창업자"
    age: int = Field(..., ge=18, le=100)
    region: str
    business_stage: str
    business_field: str
    target_type: str
    is_veteran: bool = False
    is_disabled: bool = False
    additional_context: str = ""
    desired_data_types: List[str] = Field(default_factory=lambda: ["announcement", "business"])


class MatchRequest(BaseModel):
    profile: Optional[ProfileRequest] = None
    question: Optional[str] = None
    top_n: int = Field(default=10, ge=1, le=50)
    mode: str = DEFAULT_WEB_MODE
    use_cache: bool = True


class ChatRequest(BaseModel):
    profile: Optional[ProfileRequest] = None
    question: str
    history: List[dict] = Field(default_factory=list)
    mode: str = DEFAULT_WEB_MODE
    chat_mode: str = "general"


class LLMConfigRequest(BaseModel):
    provider: str
    model: str


cfg: Any = None
llm: Any = None
graph_app: Any = None
announcement_lookup: Dict[str, List[Dict[str, Any]]] = {}
rebuild_lock = threading.Lock()
rebuild_status: Dict[str, Any] = {
    "running": False,
    "status": "idle",
    "started_at": None,
    "finished_at": None,
    "error": None,
    "message": "",
}

_FOLLOW_UP_KEYWORDS = [
    "링크", "url", "주소", "홈페이지", "신청", "안내", "접수", "마감", "기한",
    "대상", "조건", "자격", "기간", "언제", "어디", "얼마", "비교", "차이",
    "그거", "그 사업", "이거", "이 사업", "위 사업", "첫번째", "두번째",
]


def _profile_context(profile: ProfileRequest) -> str:
    flags = []
    if profile.is_veteran:
        flags.append("참전유공자")
    if profile.is_disabled:
        flags.append("장애인")
    extra_flags = f", 특이사항: {', '.join(flags)}" if flags else ""
    extra = f", 추가설명: {profile.additional_context}" if profile.additional_context else ""
    return (
        f"이용자 프로필: 이름 {profile.name}, 나이 {profile.age}세, 지역 {profile.region}, "
        f"창업단계 {profile.business_stage}, 사업분야 {profile.business_field}, 대상유형 {profile.target_type}"
        f"{extra_flags}{extra}."
    )


def _match_prompt_for_doc_types(selected_doc_types: List[str]) -> str:
    wanted = set(_safe_doc_types(selected_doc_types))
    if not wanted or wanted == set(ALL_DOC_TYPES):
        return (
            "위 프로필에 맞는 창업관련 정보를 찾아줘. "
            "나이, 지역, 창업단계, 대상유형은 구조화 조건으로 매칭하고, 사업분야는 의미적으로 유사한 표현까지 포함해줘."
        )
    if wanted.issubset({"announcement", "business"}):
        return (
            "위 프로필에 맞는 지원사업과 공고를 찾아줘. "
            "나이, 지역, 창업단계, 대상유형은 구조화 조건으로 매칭하고, 사업분야는 의미적으로 유사한 표현까지 포함해줘. 마감된 공고는 제외하고 진행 중이거나 예정된 것을 우선해줘."
        )
    if wanted.issubset({"content", "statistical"}):
        return (
            "위 프로필과 관련된 자료실 콘텐츠와 통계자료를 찾아줘. "
            "나이, 지역, 창업단계, 대상유형은 구조화 조건으로 참고하고, 사업분야는 의미적 유사도로 확장해서 관련 자료를 찾아줘. 마감이나 접수기간은 필수 조건으로 보지 마."
        )
    if wanted.issubset({"lecture"}):
        return (
            "위 프로필과 관련된 창업교육과 강좌를 찾아줘. "
            "나이, 지역, 창업단계, 대상유형은 구조화 조건으로 참고하고, 사업분야는 의미적으로 유사한 교육까지 포함해줘."
        )
    if wanted.issubset({"space", "center"}):
        return (
            "위 프로필과 관련된 창업공간과 센터를 찾아줘. "
            "지역은 구조화 조건으로 적용하고, 사업분야는 의미적으로 관련 있는 공간과 센터까지 포함해줘. 마감이나 접수기간은 필수 조건으로 보지 마."
        )
    if wanted.issubset({"product", "corporate"}):
        return (
            "위 프로필과 관련된 창업기업 확인제품 또는 확인기업 정보를 찾아줘. "
            "사업분야는 의미적으로 확장하고, 나머지 프로필 조건은 구조화 조건으로 참고해줘."
        )
    if wanted.issubset({"institution"}):
        return (
            "위 프로필과 관련된 창업지원기관 정보를 찾아줘. "
            "지역은 구조화 조건으로 적용하고, 사업분야는 의미적으로 유사한 기관까지 포함해줘."
        )
    return (
        "위 프로필과 선택된 데이터 유형에 맞는 정보를 찾아줘. "
        "나이, 지역, 창업단계, 대상유형은 구조화 조건으로 매칭하고, 사업분야는 의미적으로 유사한 표현까지 포함해줘."
    )


def _build_match_question(profile: Optional[ProfileRequest], selected_doc_types: List[str]) -> str:
    task = _match_prompt_for_doc_types(selected_doc_types)
    if profile is None:
        return task
    return f"{_profile_context(profile)} {task}"


def _resolve_match_question(req: MatchRequest, selected_doc_types: List[str]) -> str:
    custom = str(req.question or "").strip()
    if custom:
        if req.profile is not None:
            return f"{_profile_context(req.profile)} 추가 질문: {custom}"
        return custom
    return _build_match_question(req.profile, selected_doc_types)


def _doc_score(doc: Dict[str, Any]) -> float:
    for key in ("cross_encoder_score", "combined_score", "rrf_score", "score"):
        value = doc.get(key)
        if value is not None:
            try:
                return float(value)
            except Exception:
                return 0.0
    return 0.0


def _normalize_url(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text:
        return ""
    if text.startswith(("http://", "https://")):
        return text
    if text.startswith("//"):
        return f"https:{text}"
    if text.startswith("www."):
        return f"https://{text}"
    if re.match(r"^[A-Za-z0-9.-]+\.[A-Za-z]{2,}(/.*)?$", text):
        return f"https://{text}"
    return ""


def _recommendation_links(md: Dict[str, Any]) -> Dict[str, str]:
    apply_url = _normalize_url(md.get("apply_url", "") or md.get("biz_aply_url", ""))
    guide_url = _normalize_url(md.get("guide_url", "") or md.get("biz_gdnc_url", ""))
    detail_url = _normalize_url(
        md.get("detail_url", "") or md.get("detl_pg_url", "") or md.get("lctr_pg_url", "") or md.get("hmpg", "")
    )
    if not detail_url:
        detail_url = apply_url or guide_url
    return {
        "detail_url": detail_url,
        "guide_url": guide_url,
        "apply_url": apply_url,
    }


def _normalize_biz_name(text: Any) -> str:
    value = str(text or "").strip().lower()
    if not value:
        return ""
    value = value.replace("&apos;", " ")
    value = re.sub(r"\[[^\]]+\]", " ", value)
    value = re.sub(r"[(){}\[\]<>~!@#$%^&*_+=|\\/:;\"'`,.?-]+", " ", value)
    value = re.sub(r"\bg\s+", "g", value)
    value = re.sub(
        r"(모집공고|참여기업모집|기업모집|참가자모집|입주기업모집|모집|공고|지원사업|지원 프로그램|프로그램|사업|지원)$",
        " ",
        value,
    )
    value = re.sub(r"(예비창업)\s+(지원)", r"\1\2", value)
    value = re.sub(r"(초기창업)\s+(패키지)", r"\1\2", value)
    value = re.sub(r"(창업도약)\s+(패키지)", r"\1\2", value)
    value = re.sub(r"(창업중심대학)\s+(지원)", r"\1\2", value)
    value = re.sub(r"\s+", "", value)
    return value


def _announcement_lookup_keys(md: Dict[str, Any]) -> List[str]:
    candidates = [
        md.get("intg_pbanc_biz_nm"),
        md.get("intg_biz_name"),
        md.get("biz_pbanc_nm"),
        md.get("title"),
    ]
    out: List[str] = []
    seen = set()
    for item in candidates:
        key = _normalize_biz_name(item)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _load_announcement_lookup(docs_pickle_path: Any) -> Dict[str, List[Dict[str, Any]]]:
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    path = Path(str(docs_pickle_path or "")).expanduser()
    if not path.exists():
        return lookup

    with path.open("rb") as f:
        docs = pickle.load(f)

    for doc in docs:
        md = getattr(doc, "metadata", {}) or {}
        if str(md.get("type") or "").strip().lower() != "announcement":
            continue
        keys = _announcement_lookup_keys(md)
        if not keys:
            continue
        links = _recommendation_links(md)
        entry = {
            "id": getattr(doc, "id", ""),
            "title": str(md.get("title") or "").strip(),
            "text": getattr(doc, "page_content", None) or getattr(doc, "text", None) or "",
            "deadline": str(md.get("deadline") or md.get("pbanc_rcpt_end_dt") or "").strip(),
            "status": str(md.get("status") or md.get("rcrt_prgs_yn") or "").strip(),
            "detail_url": links["detail_url"],
            "guide_url": links["guide_url"],
            "apply_url": links["apply_url"],
            "region": str(md.get("region") or md.get("supt_regin") or "").strip(),
            "metadata": md,
            "is_open": bool(passes_deadline_constraint(md)),
        }
        for key in keys:
            lookup.setdefault(key, []).append(entry)

    for key, items in lookup.items():
        items.sort(
            key=lambda item: (
                0 if item.get("is_open") else 1,
                str(item.get("deadline") or "9999-99-99"),
                str(item.get("title") or ""),
            )
        )
    return lookup


def _linked_announcements_for_business(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    md = rec.get("metadata", {}) or {}
    keys = _announcement_lookup_keys(md)
    seen_titles = set()
    matches: List[Dict[str, Any]] = []
    for key in keys:
        for item in announcement_lookup.get(key, []):
            sig = (item.get("title"), item.get("deadline"), item.get("detail_url"))
            if sig in seen_titles:
                continue
            seen_titles.add(sig)
            matches.append(item)
    matches.sort(
        key=lambda item: (
            0 if item.get("is_open") else 1,
            str(item.get("deadline") or "9999-99-99"),
            str(item.get("title") or ""),
        )
    )
    return matches


def _year_value(rec: Dict[str, Any]) -> int:
    md = rec.get("metadata", {}) or {}
    for key in ("year", "biz_yr"):
        value = str(md.get(key) or "").strip()
        if value.isdigit():
            return int(value)
    title = str(rec.get("title") or "")
    m = re.search(r"(20\d{2})", title)
    return int(m.group(1)) if m else 0


def _explicit_policy_group_key(rec: Dict[str, Any]) -> str:
    md = rec.get("metadata", {}) or {}
    candidates = [
        md.get("intg_pbanc_biz_nm"),
        md.get("intg_biz_name"),
    ]
    for item in candidates:
        key = _normalize_biz_name(item)
        if key:
            return f"intg:{key}"
    return ""


def _title_policy_group_key(rec: Dict[str, Any]) -> str:
    md = rec.get("metadata", {}) or {}
    candidates = [
        md.get("biz_pbanc_nm"),
        md.get("title"),
        rec.get("title"),
    ]
    for item in candidates:
        key = _normalize_biz_name(item)
        if key:
            return f"title:{key}"
    return f"id:{str(rec.get('id') or '')}"


def _policy_group_key(rec: Dict[str, Any]) -> str:
    return _explicit_policy_group_key(rec) or _title_policy_group_key(rec)


def _policy_semantic_text(rec: Dict[str, Any]) -> str:
    md = rec.get("metadata", {}) or {}
    parts = [
        md.get("intg_pbanc_biz_nm"),
        md.get("intg_biz_name"),
        md.get("biz_pbanc_nm"),
        md.get("title"),
        rec.get("title"),
        md.get("field"),
        md.get("supt_biz_clsfc"),
    ]
    return " ".join(str(x or "").strip() for x in parts if str(x or "").strip())


def _merge_grouped_recommendations_semantically(
    grouped: Dict[str, List[Dict[str, Any]]],
    *,
    similarity_threshold: float = 0.83,
) -> Dict[str, List[Dict[str, Any]]]:
    if len(grouped) <= 1:
        return grouped

    model_name = str(getattr(cfg, "EMBEDDING_MODEL_NAME", "") or "").strip()
    if not model_name:
        return grouped

    keys = list(grouped.keys())
    consumed = set()
    merged: Dict[str, List[Dict[str, Any]]] = {}

    for key in keys:
        if key in consumed:
            continue
        base_items = list(grouped.get(key) or [])
        consumed.add(key)
        base_text = _policy_semantic_text(base_items[0]) if base_items else ""
        if not base_text:
            merged[key] = base_items
            continue

        for other_key in keys:
            if other_key in consumed or other_key == key:
                continue
            if key.startswith("intg:") and other_key.startswith("intg:") and key != other_key:
                continue
            other_items = list(grouped.get(other_key) or [])
            other_text = _policy_semantic_text(other_items[0]) if other_items else ""
            if not other_text:
                continue
            try:
                sim = cosine_similarity(base_text, other_text, model_name)
            except Exception:
                sim = 0.0
            if sim >= similarity_threshold:
                base_items.extend(other_items)
                consumed.add(other_key)
        merged[key] = base_items

    return merged


def _is_stale_business_year(rec: Dict[str, Any]) -> bool:
    year = _year_value(rec)
    if not year:
        return False
    return year < datetime.now().year


def _dedupe_recommendations_by_policy_group(recs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return _collapse_recommendations_by_policy_group(recs, [])


def _bridge_business_recommendations_with_announcements(
    recs: List[Dict[str, Any]],
    selected_doc_types: List[str],
) -> List[Dict[str, Any]]:
    return _collapse_recommendations_by_policy_group(recs, selected_doc_types)


def _announcement_rec_from_linked(
    source_rec: Dict[str, Any],
    linked_ann: Dict[str, Any],
    group_size: int,
) -> Dict[str, Any]:
    metadata = dict(linked_ann.get("metadata") or {})
    metadata["bridged_from_business"] = True
    metadata["source_business_title"] = str(source_rec.get("title") or "")
    metadata["linked_open_announcement_title"] = str(linked_ann.get("title") or "")
    metadata["linked_open_announcement_deadline"] = str(linked_ann.get("deadline") or "")
    metadata["linked_open_announcement_count"] = 1
    metadata["dedup_group_size"] = group_size
    return {
        **source_rec,
        "id": linked_ann.get("id") or source_rec.get("id"),
        "title": linked_ann.get("title") or source_rec.get("title"),
        "summary": str(linked_ann.get("text") or source_rec.get("summary") or "").strip(),
        "data_type": "announcement",
        "deadline": linked_ann.get("deadline") or source_rec.get("deadline", ""),
        "detail_url": linked_ann.get("detail_url") or source_rec.get("detail_url", ""),
        "guide_url": linked_ann.get("guide_url") or source_rec.get("guide_url", ""),
        "apply_url": linked_ann.get("apply_url") or source_rec.get("apply_url", ""),
        "region": linked_ann.get("region") or source_rec.get("region", ""),
        "status": "OPEN",
        "metadata": metadata,
    }


def _collapse_recommendations_by_policy_group(
    recs: List[Dict[str, Any]],
    selected_doc_types: List[str],
) -> List[Dict[str, Any]]:
    if not recs:
        return []

    wanted = _safe_doc_types(selected_doc_types)
    business_only = wanted == ["business"]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in recs:
        grouped.setdefault(_policy_group_key(rec), []).append(rec)
    grouped = _merge_grouped_recommendations_semantically(grouped)

    collapsed: List[Dict[str, Any]] = []
    for key, items in grouped.items():
        lookup_candidates = []
        bare_key = key.split(":", 1)[1] if ":" in key else key
        lookup_candidates.append(bare_key)
        for rec in items:
            md = rec.get("metadata", {}) or {}
            for raw_key in _announcement_lookup_keys(md):
                if raw_key not in lookup_candidates:
                    lookup_candidates.append(raw_key)

        linked_map: Dict[tuple, Dict[str, Any]] = {}
        for lookup_key in lookup_candidates:
            for item in announcement_lookup.get(lookup_key, []):
                sig = (item.get("title"), item.get("deadline"), item.get("detail_url"))
                linked_map[sig] = item
        linked = list(linked_map.values())
        open_linked = [item for item in linked if item.get("is_open")]

        if open_linked:
            open_linked.sort(
                key=lambda item: (
                    str(item.get("deadline") or "9999-99-99"),
                    str(item.get("title") or ""),
                )
            )
            source_rec = sorted(
                items,
                key=lambda rec: (
                    0 if str(rec.get("data_type") or "").strip().lower() == "announcement" else 1,
                    -float(rec.get("match_score") or 0.0),
                    -float(rec.get("relevance_score") or 0.0),
                ),
            )[0]
            collapsed.append(_announcement_rec_from_linked(source_rec, open_linked[0], len(items)))
            continue

        if business_only and linked:
            continue

        business_items = [
            rec for rec in items
            if str(rec.get("data_type") or "").strip().lower() == "business"
        ]
        if business_items:
            business_items.sort(
                key=lambda rec: (
                    -_year_value(rec),
                    -float(rec.get("match_score") or 0.0),
                    -float(rec.get("relevance_score") or 0.0),
                )
            )
            best = business_items[0]
            if _is_stale_business_year(best):
                continue
        else:
            items.sort(
                key=lambda rec: (
                    0 if str(rec.get("data_type") or "").strip().lower() == "announcement" else 1,
                    -float(rec.get("match_score") or 0.0),
                    -float(rec.get("relevance_score") or 0.0),
                )
            )
            best = items[0]

        metadata = best.get("metadata", {}) or {}
        metadata["dedup_group_size"] = len(items)
        best["metadata"] = metadata
        collapsed.append(best)

    collapsed.sort(
        key=lambda rec: (
            0 if str(rec.get("data_type") or "").strip().lower() == "announcement" else 1,
            -float(rec.get("match_score") or 0.0),
            -float(rec.get("relevance_score") or 0.0),
        )
    )
    for idx, rec in enumerate(collapsed, start=1):
        rec["rank"] = idx
        rec["priority"] = "HIGH" if idx <= 3 else ("MEDIUM" if idx <= 7 else "LOW")
    return collapsed


def _normalized_relevance_scores(docs: List[Dict[str, Any]]) -> List[float]:
    raw_scores = [_doc_score(doc) for doc in docs]
    if not raw_scores:
        return []
    lo = min(raw_scores)
    hi = max(raw_scores)
    if hi <= lo:
        return [100.0 for _ in raw_scores]
    return [round(((score - lo) / (hi - lo)) * 100.0, 1) for score in raw_scores]


def _announcement_has_timing_signal(metadata: Dict[str, Any]) -> bool:
    return any(
        str(metadata.get(key) or "").strip()
        for key in (
            "status",
            "rcrt_prgs_yn",
            "deadline",
            "pbanc_rcpt_end_dt",
            "apply_period",
        )
    )


def _non_policy_recency_date(metadata: Dict[str, Any], doc_type: str) -> Optional[datetime]:
    date_keys_by_type = {
        "content": ("fstm_reg_dt", "reg_date", "reg_dt"),
        "statistical": ("last_mdfcn_dt", "fstm_reg_dt", "first_reg_dt"),
        "lecture": ("mdfcn_dt", "reg_dt"),
    }
    for key in date_keys_by_type.get(doc_type, ()):
        parsed = _parse_date(metadata.get(key))
        if parsed is not None:
            return datetime.combine(parsed, datetime.min.time())
    return None


def _is_stale_non_policy_doc(metadata: Dict[str, Any], doc_type: str) -> bool:
    recency_dt = _non_policy_recency_date(metadata, doc_type)
    if recency_dt is None:
        return False
    age_days = (datetime.now() - recency_dt).days
    max_age_days_by_type = {
        "content": 365 * 3,
        "statistical": 365 * 3,
        "lecture": 365 * 2,
    }
    max_age_days = max_age_days_by_type.get(doc_type)
    if max_age_days is None:
        return False
    return age_days > max_age_days


def _keep_recommendation_doc(doc: Dict[str, Any]) -> bool:
    metadata = doc.get("metadata", {}) or {}
    doc_type = str(metadata.get("type") or "").strip().lower()
    if doc_type == "announcement":
        if not _announcement_has_timing_signal(metadata):
            return False
        return passes_deadline_constraint(metadata)
    if doc_type == "business":
        return passes_deadline_constraint(metadata)
    if doc_type in {"content", "statistical", "lecture"}:
        return not _is_stale_non_policy_doc(metadata, doc_type)
    return True


def _select_match_docs(result: Dict[str, Any], top_n: int, selected_doc_types: List[str]) -> List[Dict[str, Any]]:
    docs = list(result.get("final_docs") or result.get("filtered_docs") or result.get("reranked_docs") or [])
    if not docs:
        return []

    wanted_types = _safe_doc_types(selected_doc_types) or list(ALL_DOC_TYPES)
    if len(wanted_types) <= 1:
        return docs[:top_n]

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for doc in docs:
        doc_type = str((doc.get("metadata", {}) or {}).get("type") or "").strip().lower()
        grouped.setdefault(doc_type, []).append(doc)

    for doc_type in grouped:
        grouped[doc_type].sort(key=_doc_score, reverse=True)

    selected: List[Dict[str, Any]] = []
    seen_ids = set()
    exhausted = False
    while len(selected) < top_n and not exhausted:
        exhausted = True
        for doc_type in wanted_types:
            bucket = grouped.get(doc_type) or []
            while bucket and bucket[0].get("id") in seen_ids:
                bucket.pop(0)
            if not bucket:
                continue
            exhausted = False
            doc = bucket.pop(0)
            doc_id = doc.get("id")
            if doc_id in seen_ids:
                continue
            seen_ids.add(doc_id)
            selected.append(doc)
            if len(selected) >= top_n:
                break

    if selected:
        return selected
    return docs[:top_n]


def _merge_match_docs_by_type(
    req: MatchRequest,
    selected_doc_types: List[str],
) -> tuple[str, List[Dict[str, Any]], List[str]]:
    """
    When multiple doc types are selected, run profile matching per type and
    merge the resulting candidates. This makes "전체" behave like a union of
    type-specific results rather than a single broad query where some types can
    disappear during retrieval/reranking.
    """
    if len(selected_doc_types) <= 1:
        question = _resolve_match_question(req, selected_doc_types)
        result = _invoke_agentic(
            question=question,
            selected_doc_types=selected_doc_types,
            profile=req.profile,
            skip_intent_classifier=True,
        )
        docs = _select_match_docs(result, req.top_n, selected_doc_types)
        docs = [doc for doc in docs if _keep_recommendation_doc(doc)]
        return (
            question,
            docs,
            list(result.get("reasoning_trace", [])),
        )

    per_type_results: Dict[str, List[Dict[str, Any]]] = {}
    merged_trace: List[str] = [
        f"profile_match: merged per-type search for {len(selected_doc_types)} selected data types"
    ]

    for doc_type in selected_doc_types:
        question = _resolve_match_question(req, [doc_type])
        result = _invoke_agentic(
            question=question,
            selected_doc_types=[doc_type],
            profile=req.profile,
            skip_intent_classifier=True,
        )
        docs = _select_match_docs(result, req.top_n, [doc_type])
        docs = [doc for doc in docs if _keep_recommendation_doc(doc)]
        docs.sort(key=_doc_score, reverse=True)
        per_type_results[doc_type] = docs
        merged_trace.append(
            f"profile_match: type={doc_type} produced {len(docs)} docs"
        )

    # Diversity without forcing equal counts:
    # 1) take at most one top document per type when relevant
    # 2) fill the remaining slots globally by relevance score
    seed_docs: List[Dict[str, Any]] = []
    seen_ids = set()
    for doc_type in selected_doc_types:
        docs = per_type_results.get(doc_type) or []
        if not docs:
            continue
        doc = docs[0]
        doc_id = doc.get("id")
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        seed_docs.append(doc)
        if len(seed_docs) >= req.top_n:
            break

    global_candidates: List[Dict[str, Any]] = []
    for docs in per_type_results.values():
        global_candidates.extend(docs)
    global_candidates.sort(key=_doc_score, reverse=True)

    final_docs: List[Dict[str, Any]] = list(seed_docs)
    for doc in global_candidates:
        doc_id = doc.get("id")
        if doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        final_docs.append(doc)
        if len(final_docs) >= req.top_n:
            break

    merged_trace.append(
        f"profile_match: seeded {len(seed_docs)} type-diverse docs, then filled remaining slots by global relevance score"
    )
    final_docs.sort(key=_doc_score, reverse=True)
    merged_trace.append(
        f"profile_match: final {len(final_docs)} docs selected"
    )
    return (
        _resolve_match_question(req, selected_doc_types),
        final_docs,
        merged_trace,
    )


def _to_recommendations(docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    recs: List[Dict[str, Any]] = []
    relevance_scores = _normalized_relevance_scores(docs)
    for i, (doc, relevance_score) in enumerate(zip(docs, relevance_scores), start=1):
        md = doc.get("metadata", {}) or {}
        summary = str(doc.get("text") or "").strip()
        score = _doc_score(doc)
        links = _recommendation_links(md)
        reasons: List[str] = []
        budget_text = str(md.get("biz_supt_bdgt_info") or md.get("support_budget") or md.get("budget") or "").strip()
        region_text = str(md.get("region") or md.get("supt_regin") or md.get("regin_clss") or md.get("regin_clss_cd") or "").strip()
        target_text = str(md.get("apply_target") or md.get("aply_trgt") or md.get("apply_target_desc") or md.get("aply_trgt_ctnt") or "").strip()
        if budget_text:
            reasons.append(f"지원 규모 정보: {budget_text}")
        if target_text:
            reasons.append(f"지원 대상: {target_text}")
        if region_text:
            reasons.append(f"지역 조건: {region_text}")
        doc_type = str(md.get("type") or "").strip().lower()
        if doc_type in {"announcement", "business"} and passes_deadline_constraint(md):
            reasons.append("마감 상태 기준을 통과한 문서")

        recs.append(
            {
                "rank": i,
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "summary": summary,
                "data_type": md.get("type", ""),
                "deadline": md.get("deadline", "") or md.get("confmdoc_expr_dt", ""),
                "detail_url": links["detail_url"],
                "guide_url": links["guide_url"],
                "apply_url": links["apply_url"],
                "region": md.get("region", "") or md.get("supt_regin", "") or md.get("regin_clss", "") or md.get("regin_clss_cd", ""),
                "host_org": md.get("host_org", "") or md.get("pbanc_ntrp_nm", ""),
                "supervisor_org": md.get("supervisor_org", "") or md.get("sprv_inst", ""),
                "status": md.get("status", "") or md.get("rcrt_prgs_yn", ""),
                "apply_target": md.get("apply_target", "") or md.get("aply_trgt", ""),
                "apply_target_desc": md.get("apply_target_desc", "") or md.get("aply_trgt_ctnt", ""),
                "startup_period": md.get("startup_period", "") or md.get("biz_enyy", ""),
                "age_limit": md.get("age_limit", "") or md.get("biz_trgt_age", ""),
                "field": md.get("field", "") or md.get("supt_biz_clsfc", "") or md.get("biz_category_cd", ""),
                "match_score": score,
                "relevance_score": relevance_score,
                "priority": "HIGH" if i <= 3 else ("MEDIUM" if i <= 7 else "LOW"),
                "reasons": reasons[:3],
                "metadata": md,
            }
        )
    return recs


def _safe_doc_types(types: List[str]) -> List[str]:
    out = []
    for t in types:
        norm = str(t or "").strip().lower()
        if norm in ALL_DOC_TYPES:
            out.append(norm)
    return list(dict.fromkeys(out))


def _effective_doc_types(types: List[str]) -> List[str]:
    """
    UI에서 데이터 유형을 하나도 선택하지 않으면 "전체 선택"으로 간주한다.
    """
    normalized = _safe_doc_types(types)
    if not normalized:
        return list(ALL_DOC_TYPES)
    return normalized


_REGION_TOKENS = [
    "서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
    "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주",
]
_NATIONWIDE_TOKENS = ["전국", "전체", "제한없음", "제한 없음", "전 지역", "전지역"]


def _region_alias(region: str) -> str:
    m = {
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
        "강원도": "강원",
        "강원특별자치도": "강원",
        "충청북도": "충북",
        "충청남도": "충남",
        "전라북도": "전북",
        "전북특별자치도": "전북",
        "전라남도": "전남",
        "경상북도": "경북",
        "경상남도": "경남",
        "제주도": "제주",
        "제주특별자치도": "제주",
    }
    r = str(region or "").strip()
    return m.get(r, r)


def _post_filter_recs_by_region(
    recs: List[Dict[str, Any]],
    user_region: Optional[str],
) -> List[Dict[str, Any]]:
    """
    지역 정보(제목/요약/메타데이터)를 보고 타지역 문서를 제거한다.
    전국/전체/제한없음 문서는 항상 유지한다.
    """
    ur = _region_alias(str(user_region or "").strip())
    if not ur or ur in {"전국", "전체"}:
        return recs

    out: List[Dict[str, Any]] = []
    for rec in recs:
        data_type = str(rec.get("data_type") or rec.get("metadata", {}).get("type") or "").strip().lower()
        # Core graph filtering already applies region constraints.
        # Keep an extra conservative post-filter only for location-centric results.
        if data_type not in {"space", "center"}:
            out.append(rec)
            continue
        text = " ".join([
            str(rec.get("title") or ""),
            str(rec.get("summary") or ""),
            str(rec.get("region") or ""),
            str((rec.get("metadata") or {}).get("region") or ""),
        ])
        if any(token in text for token in _NATIONWIDE_TOKENS):
            out.append(rec)
            continue
        mentioned = [t for t in _REGION_TOKENS if re.search(re.escape(t), text)]
        if mentioned and ur not in mentioned:
            continue
        out.append(rec)
    for idx, rec in enumerate(out, start=1):
        rec["rank"] = idx
    return out


def _startup_years_from_stage(stage: str) -> Optional[float]:
    text = str(stage or "").strip()
    if not text:
        return None
    if "예비" in text:
        return 0.0
    match = re.search(r"(\d+(?:\.\d+)?)\s*?", text)
    if match:
        return float(match.group(1))
    if any(token in text for token in ["초기", "1년 미만", "1년미만"]):
        return 1.0
    if any(token in text for token in ["도약", "성장", "3년"]):
        return 3.0
    return None


def _profile_constraints(profile: ProfileRequest) -> Dict[str, Any]:
    special_conditions: List[str] = []
    if profile.is_veteran:
        special_conditions.append("참전유공자")
    if profile.is_disabled:
        special_conditions.append("장애인")

    return {
        "age": profile.age,
        "startup_years": _startup_years_from_stage(profile.business_stage),
        "region": str(profile.region or "").strip() or None,
        "industry": str(profile.business_field or "").strip() or None,
        "target_type": str(profile.target_type or "").strip() or None,
        "special_conditions": special_conditions,
        "time_preference": "open_now",
    }


def _invoke_agentic(
    question: str,
    selected_doc_types: List[str],
    profile: Optional[ProfileRequest] = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    skip_intent_classifier: bool = False,
    raw_question: Optional[str] = None,
) -> Dict[str, Any]:
    if graph_app is None:
        raise RuntimeError("agentic_hybrid graph not initialized")
    state = init_state(
        question,
        selected_doc_types=selected_doc_types,
        profile_constraints=_profile_constraints(profile) if profile is not None else None,
        chat_history=chat_history or [],
        skip_intent_classifier=skip_intent_classifier,
    )
    state["user_question"] = str(raw_question or question or "").strip()
    return graph_app.invoke(state)


def _normalize_history(history: List[dict]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for item in history[-6:]:
        role = str(item.get("role") or "").strip().lower()
        content = str(item.get("content") or "").strip()
        if role in {"user", "assistant"} and content:
            out.append({"role": role, "content": content})
    return out


def _last_history_content(history: List[Dict[str, str]], role: str) -> str:
    for item in reversed(history):
        if item.get("role") == role and item.get("content"):
            return str(item["content"]).strip()
    return ""


def _looks_like_followup(question: str, history: List[Dict[str, str]]) -> bool:
    text = str(question or "").strip()
    if not text or not history:
        return False

    prev_user = _last_history_content(history[:-1], "user")
    prev_assistant = _last_history_content(history[:-1], "assistant")
    if not prev_user and not prev_assistant:
        return False

    if llm is not None and _llm_enabled(llm):
        prompt = (
            "다음 현재 질문이 이전 대화 맥락이 있어야만 해석되는 후속 질문인지 판단하세요.\n"
            "판단 기준:\n"
            "- 이전 대화가 없어도 독립적으로 검색/답변 가능하면 반드시 NO\n"
            "- 현재 질문이 이전 답변의 특정 대상, 순번, 링크, 마감, 조건을 직접 가리키면 YES\n"
            "- 주제가 비슷해 보여도 새로운 엔티티나 새로운 정보를 묻는 독립 질문이면 NO\n"
            "- 보수적으로 판단하세요. 확실하지 않으면 NO\n\n"
            f"이전 사용자 질문: {prev_user or '(없음)'}\n"
            f"이전 답변 요약: {(prev_assistant or '(없음)')[:400]}\n"
            f"현재 질문: {text}\n\n"
            "반드시 YES 또는 NO만 출력하세요."
        )
        decision = llm.complete(
            prompt,
            system_prompt="You classify whether the user's current question is context-dependent. Output only YES or NO.",
            max_tokens=5,
        ).strip().upper()
        if decision.startswith("YES"):
            return True
        if decision.startswith("NO"):
            return False

    lowered = text.lower()
    has_referential_cue = any(keyword in lowered for keyword in _FOLLOW_UP_KEYWORDS)
    return has_referential_cue


def _build_chat_question(
    profile: Optional[ProfileRequest],
    question: str,
    history: List[Dict[str, str]],
) -> str:
    base_question = str(question or "").strip()
    if profile is not None:
        base = f"{_profile_context(profile)} 추가 질문: {base_question}"
    else:
        base = base_question

    if not history or not _looks_like_followup(base_question, history):
        return base

    prev_user = _last_history_content(history[:-1], "user")
    prev_assistant = _last_history_content(history[:-1], "assistant")
    context_parts: List[str] = []
    if prev_user:
        context_parts.append(f"이전 사용자 질문: {prev_user}")
    if prev_assistant:
        context_parts.append(f"직전 답변 요약: {prev_assistant[:500]}")
    context_parts.append(f"현재 후속 질문: {base}")
    return "\n".join(context_parts)


def _llm_enabled(runtime_llm: Any) -> bool:
    if runtime_llm is None:
        return False
    if getattr(runtime_llm, "client", None) is not None:
        return True
    return getattr(runtime_llm, "backend", None) == "transformers_causal" and getattr(runtime_llm, "model", None) is not None


def _rebuild_runtime_with_cfg(new_cfg: Any) -> None:
    global cfg, llm, graph_app, announcement_lookup
    cfg = new_cfg
    llm = SimpleLLM(cfg)
    print(
        f"[LLM] provider={cfg.LLM_PROVIDER}, model={cfg.LLM_MODEL_NAME}, "
        f"base_url={cfg.OPENAI_BASE_URL}, backend={llm.backend}, enabled={_llm_enabled(llm)}, "
        f"init_error={getattr(llm, 'init_error', None)}"
    )
    graph_app = build_agentic_graph(cfg, llm)
    try:
        announcement_lookup = _load_announcement_lookup(getattr(cfg, "docs_pickle_path", ""))
        print(f"[AnnouncementLookup] loaded {len(announcement_lookup)} business-announcement keys")
    except Exception as exc:
        announcement_lookup = {}
        print(f"[AnnouncementLookup] failed to load: {exc}")


def _set_rebuild_status(**updates: Any) -> None:
    rebuild_status.update(updates)


def _perform_rebuild_index_job() -> None:
    _set_rebuild_status(
        running=True,
        status="running",
        started_at=datetime.now().isoformat(),
        finished_at=None,
        error=None,
        message="legacy pipeline로 원천 데이터 수집 및 인덱스 재생성을 시작했습니다.",
    )
    try:
        legacy = LegacyAgenticOrchestrator(
            service_key=Config.SERVICE_KEY,
            llm_api_key=Config.LLM_API_KEY,
        )
        legacy.ensure_index(use_cache=False)

        active_cfg = cfg if cfg is not None else _web_runtime_cfg()
        _rebuild_runtime_with_cfg(active_cfg)
        _set_rebuild_status(
            running=False,
            status="completed",
            finished_at=datetime.now().isoformat(),
            message="인덱스 재생성이 완료되었고 현재 FastAPI 런타임에 새 cache를 다시 연결했습니다.",
        )
    except Exception as exc:
        _set_rebuild_status(
            running=False,
            status="failed",
            finished_at=datetime.now().isoformat(),
            error=str(exc),
            message="인덱스 재생성 중 오류가 발생했습니다.",
        )
        print(f"[rebuild-index] failed: {exc}")


def _run_rebuild_index_job() -> None:
    if not rebuild_lock.acquire(blocking=False):
        return
    try:
        _perform_rebuild_index_job()
    finally:
        rebuild_lock.release()


def _clean_generated_text(text: str) -> str:
    cleaned = str(text or "").replace("<|im_end|>", "").strip()
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    return cleaned


def _build_thought(
    question: str,
    trace_lines: List[str],
    *,
    answer: str = "",
    final_docs: Optional[List[Dict[str, Any]]] = None,
) -> List[str]:
    if not trace_lines:
        return []
    thought_lines = _llm_thoughts(llm, question, trace_lines)
    if not thought_lines:
        thought_lines = _to_natural_thoughts_for_lang(trace_lines, "ko")
    cleaned_lines = [_clean_generated_text(line) for line in thought_lines if _clean_generated_text(line)]
    final_lines = list(cleaned_lines[:5])

    final_doc_count = len(final_docs or [])
    if final_doc_count:
        cited_indices = {
            int(match)
            for match in re.findall(r"\[(\d+)\]", str(answer or ""))
            if str(match).isdigit()
        }
        cited_doc_count = sum(1 for idx in cited_indices if 1 <= idx <= final_doc_count)
        if cited_doc_count > 0:
            final_lines.append(
                f"최종 답변에는 근거 문서 {final_doc_count}건이 사용되었고, 이 중 {cited_doc_count}건이 답변 본문에서 직접 인용되었습니다."
            )
        else:
            final_lines.append(
                f"최종 답변에는 근거 문서 {final_doc_count}건이 사용되었습니다."
            )

    deduped: List[str] = []
    seen = set()
    for line in final_lines:
        if line and line not in seen:
            seen.add(line)
            deduped.append(line)
    return deduped[:6]


def _finalize_match_thought(
    *,
    thought_lines: List[str],
    recommendation_docs: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    profile: Optional[ProfileRequest],
) -> List[str]:
    internal_count = len(recommendation_docs)
    shown_count = len(recommendations)
    region_name = str(profile.region).strip() if profile is not None else "선택한 지역"

    doc_types = [
        str((doc.get("metadata", {}) or {}).get("type") or "").strip().lower()
        for doc in recommendation_docs
    ]
    policy_count = sum(1 for t in doc_types if t in {"announcement", "business"})
    loc_count = sum(1 for t in doc_types if t in {"space", "center"})

    summary_lines: List[str] = []
    if shown_count == internal_count:
        summary_lines.append(f"최종 화면 기준으로 추천 {shown_count}건이 그대로 반영되었습니다.")
    elif shown_count > 0:
        summary_lines.append(
            f"내부 후보는 {internal_count}건이었고, 최종 화면에는 후처리를 거친 {shown_count}건이 노출되었습니다."
        )
    else:
        summary_lines.append(
            f"내부 후보는 {internal_count}건이었지만, 최종 화면에는 {region_name} 조건까지 다시 반영한 결과 남은 추천이 없었습니다."
        )

    filter_notes: List[str] = []
    if policy_count > 0:
        filter_notes.append("공고/사업 문서는 마감 상태와 접수 가능 여부를 한 번 더 확인했습니다.")
    if loc_count > 0:
        filter_notes.append("공간·센터 문서는 지역 조건을 다시 확인했습니다.")

    bridged_count = 0
    grouped_total = 0
    stale_business_left = 0
    for rec in recommendations:
        metadata = rec.get("metadata", {}) or {}
        if metadata.get("bridged_from_business"):
            bridged_count += 1
        grouped_total += max(int(metadata.get("dedup_group_size") or 1), 1)
        if str(rec.get("data_type") or "").strip().lower() == "business":
            year = str(metadata.get("year") or metadata.get("biz_yr") or "").strip()
            if year.isdigit() and int(year) < datetime.now().year:
                stale_business_left += 1

    dedup_removed = max(grouped_total - shown_count, 0)
    if dedup_removed > 0:
        filter_notes.append(f"같은 사업군으로 보이는 후보를 묶어 중복 후보 {dedup_removed}건을 정리했습니다.")
    if bridged_count > 0:
        filter_notes.append(f"연결된 공고가 열려 있는 사업은 business 대신 announcement 기준으로 {bridged_count}건을 대표 표시했습니다.")
    if stale_business_left == 0 and shown_count > 0:
        filter_notes.append("현재 연도보다 이전인 사업 정보는 추천 결과에서 제외했습니다.")

    if shown_count != internal_count and filter_notes:
        summary_lines.extend(filter_notes)
    elif filter_notes:
        summary_lines.extend(filter_notes[:2])

    seen = set()
    merged: List[str] = []
    for line in summary_lines + list(thought_lines):
        line = _clean_generated_text(line)
        if not line or line in seen:
            continue
        seen.add(line)
        merged.append(line)
    return merged[:6]


def _build_match_thought(
    *,
    trace_lines: List[str],
    recommendation_docs: List[Dict[str, Any]],
    recommendations: List[Dict[str, Any]],
    profile: Optional[ProfileRequest],
) -> List[str]:
    # Profile matching UI should remain consistent, so keep this path
    # deterministic instead of mixing LLM-written thoughts with templated lines.
    natural_lines = _to_natural_thoughts_for_lang(trace_lines, "ko") if trace_lines else []
    natural_lines = [_clean_generated_text(line) for line in natural_lines if _clean_generated_text(line)]
    return _finalize_match_thought(
        thought_lines=natural_lines,
        recommendation_docs=recommendation_docs,
        recommendations=recommendations,
        profile=profile,
    )


def _web_runtime_cfg() -> Any:
    base_cfg = apply_experiment_mode(load_config(), DEFAULT_WEB_MODE)
    # Web demo uses the no_intent_dense backbone, but keeps filtering and
    # deadline exclusion enabled so expired/ineligible policy docs do not leak
    # into final recommendations.
    return replace(base_cfg, USE_FILTER=True, USE_DEADLINE_GUARD=True)


_STREAM_STAGE_LABELS = {
    "intent_classifier": "의도/문서유형 분류",
    "query_expansion": "질의 확장",
    "retrieve": "1차 검색",
    "doc_type_router": "문서 유형 라우팅",
    "rerank": "재정렬",
    "planner": "질의 파싱",
    "inherit_deadline": "마감 정보 상속",
    "llm_deadline_review": "마감 상태 보조 판정",
    "filter": "조건 필터링",
    "freshness_rerank": "최신성 보정",
    "dedup": "중복 제거",
    "cross_doc_enrich": "교차 문서 보강",
    "final_policy_gate": "최종 정책 게이트",
    "generate": "답변 생성",
    "revise": "보완 검색/수정",
}


def _sse_frame(event: str, payload: Dict[str, Any]) -> bytes:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")


def _is_meaningful_progress(node_name: str, trace_lines: List[str]) -> bool:
    if not trace_lines:
        return False
    text = " ".join(str(line or "") for line in trace_lines).lower()
    if any(token in text for token in ["skipped", "disabled", "not used", "no docs"]):
        return False
    if str(node_name) in {"intent_classifier", "query_expansion"}:
        return False
    if "enriched 0/" in text:
        return False
    if "removed 0 duplicates" in text:
        return False
    if "kept 40/40" in text or "kept 20/20" in text or "kept 5/5" in text:
        return False
    return True


def _stream_agentic_chat(
    *,
    question: str,
    raw_question: str,
    selected_doc_types: List[str],
    profile: Optional[ProfileRequest],
    normalized_history: List[Dict[str, str]],
    skip_intent_classifier: bool,
):
    if graph_app is None:
        yield _sse_frame("error", {"message": "agentic_hybrid graph not initialized"})
        return

    state = init_state(
        question,
        selected_doc_types=selected_doc_types,
        profile_constraints=_profile_constraints(profile) if profile is not None else None,
        chat_history=normalized_history or [],
        skip_intent_classifier=skip_intent_classifier,
    )
    state["user_question"] = str(raw_question or question or "").strip()

    yield _sse_frame(
        "status",
        {
            "message": "질문을 처리하고 있습니다.",
            "selected_doc_types": selected_doc_types,
        },
    )

    latest_state: Dict[str, Any] = dict(state)
    last_trace_len = 0
    step_index = 0

    try:
        for chunk in graph_app.stream(state):
            if not isinstance(chunk, dict):
                continue
            for node_name, node_out in chunk.items():
                if not isinstance(node_out, dict):
                    continue
                latest_state.update(node_out)
                trace_lines = list(node_out.get("reasoning_trace", latest_state.get("reasoning_trace", [])) or [])
                new_lines = trace_lines[last_trace_len:]
                last_trace_len = len(trace_lines)
                if not _is_meaningful_progress(str(node_name), new_lines):
                    continue
                step_index += 1
                yield _sse_frame(
                    "progress",
                    {
                        "step": step_index,
                        "node": str(node_name),
                        "label": _STREAM_STAGE_LABELS.get(str(node_name), str(node_name)),
                        "trace_lines": new_lines,
                    },
                )

        final_trace = list(latest_state.get("reasoning_trace", []) or [])
        thought_lines = _build_thought(
            question,
            final_trace,
            answer=latest_state.get("answer", ""),
            final_docs=latest_state.get("final_docs", []),
        )
        yield _sse_frame(
            "final",
            {
                "answer": _clean_generated_text(latest_state.get("answer", "")),
                "reasoning_trace": final_trace,
                "thought": thought_lines,
                "final_docs": latest_state.get("final_docs", []),
                "structured_facts": latest_state.get("structured_facts", []),
            },
        )
    except Exception as e:
        yield _sse_frame("error", {"message": str(e)})


@app.on_event("startup")
async def startup_event():
    global cfg, llm, graph_app
    try:
        _rebuild_runtime_with_cfg(_web_runtime_cfg())
        print("✅ agentic_hybrid graph initialized")
    except Exception as e:
        print(f"❌ agentic_hybrid init failed: {e}")
        graph_app = None


@app.get("/")
async def home():
    if not INDEX_FILE.exists():
        raise HTTPException(status_code=500, detail="React frontend not found: web/frontend/index.html")
    return FileResponse(str(INDEX_FILE))


@app.post("/api/match")
async def match_profile(req: MatchRequest):
    if graph_app is None:
        raise HTTPException(status_code=500, detail="agentic_hybrid graph not initialized")

    try:
        rebuilt_fresh = False
        if not bool(req.use_cache):
            if not rebuild_lock.acquire(blocking=False):
                raise HTTPException(status_code=409, detail="인덱스 재생성 작업이 이미 진행 중입니다. 잠시 후 다시 시도해 주세요.")
            try:
                _perform_rebuild_index_job()
                if rebuild_status.get("status") != "completed":
                    raise HTTPException(
                        status_code=500,
                        detail=rebuild_status.get("error") or "인덱스 재생성에 실패했습니다.",
                    )
                rebuilt_fresh = True
            finally:
                rebuild_lock.release()

        raw_selected_doc_types = req.profile.desired_data_types if req.profile is not None else []
        selected_doc_types = _effective_doc_types(raw_selected_doc_types)
        if bool(_safe_doc_types(raw_selected_doc_types)):
            question, recommendation_docs, trace_lines = _merge_match_docs_by_type(
                req,
                selected_doc_types,
            )
            llm_summary = ""
        else:
            question = _resolve_match_question(req, selected_doc_types)
            result = _invoke_agentic(
                question=question,
                selected_doc_types=selected_doc_types,
                profile=req.profile,
                skip_intent_classifier=False,
            )
            recommendation_docs = _select_match_docs(result, req.top_n, selected_doc_types)
            trace_lines = list(result.get("reasoning_trace", []))
            llm_summary = result.get("answer", "")
        recommendations = _to_recommendations(recommendation_docs)
        recommendations = _bridge_business_recommendations_with_announcements(
            recommendations,
            selected_doc_types,
        )
        recommendations = _dedupe_recommendations_by_policy_group(recommendations)
        recommendations = _post_filter_recs_by_region(
            recommendations,
            req.profile.region if req.profile is not None else None,
        )
        by_priority = {
            "HIGH": sum(1 for r in recommendations if r["priority"] == "HIGH"),
            "MEDIUM": sum(1 for r in recommendations if r["priority"] == "MEDIUM"),
            "LOW": sum(1 for r in recommendations if r["priority"] == "LOW"),
        }
        thought_lines = _build_match_thought(
            trace_lines=trace_lines,
            recommendation_docs=recommendation_docs,
            recommendations=recommendations,
            profile=req.profile,
        )
        return JSONResponse(
            content={
                "status": "SUCCESS",
                "profile": req.profile.model_dump() if req.profile is not None else None,
                "question": question,
                "llm_summary": llm_summary,
                "recommendations": recommendations,
                "total_matches": len(recommendations),
                "by_priority": by_priority,
                "internal_match_count": len(recommendation_docs),
                "reasoning_trace": trace_lines,
                "thought": thought_lines,
                "fresh_rebuild": rebuilt_fresh,
                "rebuild_status": dict(rebuild_status) if rebuilt_fresh else None,
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(req: ChatRequest):
    if graph_app is None:
        raise HTTPException(status_code=500, detail="agentic_hybrid graph not initialized")

    try:
        active_profile = req.profile if str(req.chat_mode or "general").strip().lower() == "profile" else None
        print(f"[chat] mode={str(req.chat_mode or 'general').strip().lower()} req_profile={'yes' if req.profile is not None else 'no'} active_profile={'yes' if active_profile is not None else 'no'}")
        print(f"[chat] question={str(req.question or '').strip()[:160]}")
        selected_doc_types = _effective_doc_types(active_profile.desired_data_types if active_profile is not None else [])
        normalized_history = _normalize_history(req.history)
        question = _build_chat_question(active_profile, req.question, normalized_history)
        result = _invoke_agentic(
            question=question,
            selected_doc_types=selected_doc_types,
            profile=active_profile,
            chat_history=normalized_history,
            skip_intent_classifier=True,
            raw_question=req.question,
        )
        trace_lines = result.get("reasoning_trace", [])
        thought_lines = _build_thought(
            question,
            trace_lines,
            answer=result.get("answer", ""),
            final_docs=result.get("final_docs", []),
        )
        return {
            "answer": _clean_generated_text(result.get("answer", "")),
            "reasoning_trace": trace_lines,
            "thought": thought_lines,
            "final_docs": result.get("final_docs", []),
            "structured_facts": result.get("structured_facts", []),
            "chat_mode": str(req.chat_mode or "general").strip().lower(),
            "profile_used": active_profile is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    if graph_app is None:
        raise HTTPException(status_code=500, detail="agentic_hybrid graph not initialized")

    active_profile = req.profile if str(req.chat_mode or "general").strip().lower() == "profile" else None
    print(f"[chat] mode={str(req.chat_mode or 'general').strip().lower()} req_profile={'yes' if req.profile is not None else 'no'} active_profile={'yes' if active_profile is not None else 'no'}")
    print(f"[chat] question={str(req.question or '').strip()[:160]}")
    selected_doc_types = _effective_doc_types(active_profile.desired_data_types if active_profile is not None else [])
    normalized_history = _normalize_history(req.history)
    question = _build_chat_question(active_profile, req.question, normalized_history)

    return StreamingResponse(
        _stream_agentic_chat(
            question=question,
            raw_question=req.question,
            selected_doc_types=selected_doc_types,
            profile=active_profile,
            normalized_history=normalized_history,
            skip_intent_classifier=True,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/health")
async def health_check():
    return {
        "status": "ok",
        "agentic_hybrid": graph_app is not None,
        "runtime": {
            "web_mode": DEFAULT_WEB_MODE,
            "use_filter": getattr(cfg, "USE_FILTER", None),
            "use_deadline_guard": getattr(cfg, "USE_DEADLINE_GUARD", None),
            "top_k_retrieval": getattr(cfg, "TOP_K_RETRIEVAL", None),
            "top_k_final": getattr(cfg, "TOP_K_FINAL", None),
        },
        "llm": {
            "provider": getattr(cfg, "LLM_PROVIDER", None),
            "model": getattr(cfg, "LLM_MODEL_NAME", None),
            "backend": getattr(llm, "backend", None) if llm is not None else None,
            "enabled": _llm_enabled(llm),
            "init_error": getattr(llm, "init_error", None) if llm is not None else None,
        },
    }


@app.get("/api/llm/config")
async def get_llm_config():
    if cfg is None:
        raise HTTPException(status_code=500, detail="config not initialized")
    return {
        "provider": cfg.LLM_PROVIDER,
        "model": cfg.LLM_MODEL_NAME,
        "backend": getattr(llm, "backend", None) if llm is not None else None,
        "enabled": _llm_enabled(llm),
        "init_error": getattr(llm, "init_error", None) if llm is not None else None,
    }


@app.get("/api/admin/rebuild-index")
async def get_rebuild_index_status():
    return dict(rebuild_status)


@app.post("/api/admin/rebuild-index")
async def rebuild_index(background_tasks: BackgroundTasks):
    if rebuild_status.get("running"):
        return {
            "status": "already_running",
            "message": "이미 인덱스 재생성 작업이 진행 중입니다.",
            "job": dict(rebuild_status),
        }

    background_tasks.add_task(_run_rebuild_index_job)
    queued_at = datetime.now().isoformat()
    _set_rebuild_status(
        running=True,
        status="queued",
        started_at=queued_at,
        finished_at=None,
        error=None,
        message="인덱스 재생성 작업이 큐에 등록되었습니다.",
    )
    return {
        "status": "queued",
        "message": "인덱스 재생성 작업을 시작했습니다.",
        "job": dict(rebuild_status),
    }


@app.post("/api/llm/config")
async def set_llm_config(req: LLMConfigRequest):
    if cfg is None:
        raise HTTPException(status_code=500, detail="config not initialized")
    provider = str(req.provider or "").strip().lower()
    model = str(req.model or "").strip()
    if provider not in {"transformers", "openai", "huggingface", "hf", "vllm", "google"}:
        raise HTTPException(status_code=400, detail="provider must be transformers|openai|huggingface|hf|vllm|google")
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    try:
        new_cfg = replace(cfg, LLM_PROVIDER=provider, LLM_MODEL_NAME=model)
        _rebuild_runtime_with_cfg(new_cfg)
        return {
            "status": "ok",
            "provider": cfg.LLM_PROVIDER,
            "model": cfg.LLM_MODEL_NAME,
            "backend": getattr(llm, "backend", None),
            "enabled": _llm_enabled(llm),
            "init_error": getattr(llm, "init_error", None),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to apply llm config: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web.fastapi_app:app", host="0.0.0.0", port=8000, reload=True)
