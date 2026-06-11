"""AutoRAG를 사용해 data_type별 QA 데이터를 자동 생성.

✅ 이번 버전은 'corpus parquet에 data_type이 없는 문제'를 최대한 자동으로 복구합니다.
- metadata 안에 data_type이 있으면 바로 사용
- 없으면 corpus의 chunk id에서 원본 doc_id를 추정해서 parsed.parquet과 매칭

사용 예
  # (1) corpus 경로를 직접 주는 방법(네가 지금처럼)
  python scripts/03_generate_qa.py \
    --samples-per-type 5 \
    --llm-provider local \
    --local-model "Qwen/Qwen2.5-7B-Instruct" \
    --corpus-path autorag_workspace/corpus/0.parquet

  # (2) corpus 경로를 안 주면, autorag_workspace/corpus 아래 최신 parquet을 자동 탐지
  python scripts/03_generate_qa.py --samples-per-type 5 --llm-provider openai

필요 파일
- autorag_workspace/parsed/parsed.parquet
- autorag_workspace/corpus/*.parquet (chunk 결과)
"""

from __future__ import annotations

import argparse
import ast
import importlib
import pkgutil
import re
import shutil
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config
from autorag.data.qa.filter.dontknow import dontknow_filter_rule_based
from autorag.data.qa.sample import random_single_hop
from autorag.data.qa.schema import Corpus, Raw

AUTORAG_QA_PIPELINE_AVAILABLE = True
_AUTORAG_IMPORT_ERRORS: list[str] = []


POLICY_SAMPLE_MULTIPLIER = 3

CANONICAL_TYPES = {
    "announcement": "announcements",
    "announcements": "announcements",
    "business": "business",
    "biz": "business",
    "content": "content",
    "contents": "content",
    "statistical": "statistical",
    "statistics": "statistical",
    "stat": "statistical",
    "edu_lecture": "edu_lectures",
    "edu_lectures": "edu_lectures",
    "lecture": "edu_lectures",
    "lectures": "edu_lectures",
    "education": "edu_lectures",
    "slp_space": "spaces",
    "space": "spaces",
    "spaces": "spaces",
    "slp_spaces": "spaces",
    "slp_center": "centers",
    "center": "centers",
    "centers": "centers",
    "slp_centers": "centers",
    "cert_product": "products",
    "product": "products",
    "products": "products",
    "cert_products": "products",
    "cert_corporate": "corporates",
    "corporate": "corporates",
    "corporates": "corporates",
    "institution": "institutions",
    "institutions": "institutions",
}

POLICY_TYPES = {"announcements", "business"}
GENERIC_TYPE_IDS = set(CANONICAL_TYPES) | set(CANONICAL_TYPES.values())
DATA_TYPES = [
    "announcements",
    "business",
    "content",
    "statistical",
    "edu_lectures",
    "spaces",
    "centers",
    "products",
    "corporates",
    "institutions",
]


def _find_attr_in_package(package_name: str, attr_name: str):
    tried: list[str] = []

    # 1) package root first
    try:
        pkg = importlib.import_module(package_name)
        tried.append(package_name)
        if hasattr(pkg, attr_name):
            return getattr(pkg, attr_name), tried
    except Exception as exc:
        tried.append(f"{package_name} ({type(exc).__name__}: {exc})")
        return None, tried

    # 2) recursive submodule scan
    pkg_path = getattr(pkg, "__path__", None)
    if not pkg_path:
        return None, tried

    for mod in pkgutil.walk_packages(pkg_path, pkg.__name__ + "."):
        mod_name = mod.name
        tried.append(mod_name)
        try:
            module = importlib.import_module(mod_name)
        except Exception:
            continue
        if hasattr(module, attr_name):
            return getattr(module, attr_name), tried

    return None, tried


def _resolve_autorag_qa_functions():
    global AUTORAG_QA_PIPELINE_AVAILABLE, make_basic_gen_gt, make_concise_gen_gt, factoid_query_gen

    basic, basic_tried = _find_attr_in_package("autorag.data.qa.generation_gt", "make_basic_gen_gt")
    concise, concise_tried = _find_attr_in_package("autorag.data.qa.generation_gt", "make_concise_gen_gt")
    query, query_tried = _find_attr_in_package("autorag.data.qa.query", "factoid_query_gen")

    make_basic_gen_gt = basic  # type: ignore[assignment]
    make_concise_gen_gt = concise  # type: ignore[assignment]
    factoid_query_gen = query  # type: ignore[assignment]

    if make_basic_gen_gt is None:
        AUTORAG_QA_PIPELINE_AVAILABLE = False
        _AUTORAG_IMPORT_ERRORS.append(
            "generation_gt import failed: make_basic_gen_gt not found in autorag.data.qa.generation_gt package tree "
            f"(tried {len(basic_tried)} modules)"
        )
    if make_concise_gen_gt is None:
        AUTORAG_QA_PIPELINE_AVAILABLE = False
        _AUTORAG_IMPORT_ERRORS.append(
            "generation_gt import failed: make_concise_gen_gt not found in autorag.data.qa.generation_gt package tree "
            f"(tried {len(concise_tried)} modules)"
        )
    if factoid_query_gen is None:
        AUTORAG_QA_PIPELINE_AVAILABLE = False
        _AUTORAG_IMPORT_ERRORS.append(
            "query import failed: factoid_query_gen not found in autorag.data.qa.query package tree "
            f"(tried {len(query_tried)} modules)"
        )


_resolve_autorag_qa_functions()


def _override_autorag_prompts() -> None:
    """Override AutoRAG prompts for Korean startup-support QA generation.

    We keep AutoRAG's generation functions but replace the prompt content so that
    generated QA is closer to real user questions in this project.
    """
    try:
        query_prompt_mod = importlib.import_module("autorag.data.qa.query.prompt")
        gen_prompt_mod = importlib.import_module("autorag.data.qa.generation_gt.prompt")
        llm_types = importlib.import_module("llama_index.core.base.llms.types")
    except Exception as exc:
        _AUTORAG_IMPORT_ERRORS.append(f"prompt override skipped: {type(exc).__name__}: {exc}")
        return

    ChatMessage = getattr(llm_types, "ChatMessage", None)
    MessageRole = getattr(llm_types, "MessageRole", None)
    TextBlock = getattr(llm_types, "TextBlock", None)
    if ChatMessage is None or MessageRole is None or TextBlock is None:
        _AUTORAG_IMPORT_ERRORS.append("prompt override skipped: llama_index ChatMessage types not found")
        return

    query_ko = """당신은 한국 공공 창업지원 RAG 평가용 질문 생성기입니다.
주어진 Text를 바탕으로 실제 사용자가 물을 법한 한국어 질문 1개를 생성하세요.

질문 생성 기준:
1. 질문은 반드시 제공된 Text의 내용만 근거로 만들어야 합니다.
2. 한국의 창업지원 서비스 맥락에 맞는 질문을 우선 생성하세요.
3. 가능하면 아래 요소 중 문서에 실제로 있는 정보를 묻도록 하세요.
   - 지원 대상
   - 신청 자격
   - 연령 조건
   - 지역 조건
   - 창업 단계
   - 업종 또는 분야
   - 접수 기간 또는 마감 여부
   - 공간 위치 또는 이용 조건
4. 파일 이름, 파일 제목, 문서 출처 자체를 묻지 마세요.
5. 질문에 '주어진 문서에서', '제공된 텍스트에서' 같은 표현을 넣지 마세요.
6. 질문은 최대한 구체적으로, 실제 검색 질의처럼 자연스럽게 한국어로 작성하세요.
7. 문서에 없는 사실을 추정해서 질문하지 마세요.
8. 가능하면 예/아니오형보다 구체 정보형 질문을 우선하세요.
"""

    basic_ko = """당신은 한국 공공 창업지원 문서를 바탕으로 답변을 작성하는 AI입니다.
질문에 대한 답을 제공된 Text 안에서만 찾으세요.
문서에 근거가 있는 정보만 사용하고, 추정하거나 보완하지 마세요.
지원 대상, 자격, 연령, 지역, 업종, 창업단계, 마감, 접수기간, 공간 위치, 이용 조건이 있으면 그대로 답하세요.
질문에 대한 직접 근거가 없으면 '문서에 명시되지 않음'이라고 답하세요.
답변은 한국어로 작성하세요.
"""

    concise_ko = """당신은 한국 공공 창업지원 문서를 바탕으로 매우 간결한 답을 작성하는 AI입니다.
질문에 대한 답을 제공된 Text 안에서만 찾으세요.
문서에 근거가 있는 핵심 정보만 짧게 답하세요.
문서에 직접 근거가 없으면 '문서에 명시되지 않음'이라고만 답하세요.
완전한 문장이 아니어도 되며, 한국어로 작성하세요.
"""

    try:
        query_prompt_mod.QUERY_GEN_PROMPT["factoid_single_hop"]["ko"] = [
            ChatMessage(
                role=MessageRole.SYSTEM,
                blocks=[TextBlock(block_type="text", text=query_ko)],
                additional_kwargs={},
            )
        ]
    except Exception as exc:
        _AUTORAG_IMPORT_ERRORS.append(f"query prompt override failed: {type(exc).__name__}: {exc}")

    try:
        gen_prompt_mod.GEN_GT_SYSTEM_PROMPT["basic"]["ko"] = basic_ko
        gen_prompt_mod.GEN_GT_SYSTEM_PROMPT["concise"]["ko"] = concise_ko
    except Exception as exc:
        _AUTORAG_IMPORT_ERRORS.append(f"generation prompt override failed: {type(exc).__name__}: {exc}")


def _glob_latest_parquet(corpus_dir: Path) -> Path:
    candidates = [p for p in corpus_dir.rglob("*.parquet") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No parquet found under: {corpus_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _as_dict(x: Any) -> Optional[Dict[str, Any]]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, dict):
        return x
    if isinstance(x, str):
        s = x.strip()
        if len(s) < 2:
            return None
        if s[0] in "{[" and s[-1] in "}]":
            try:
                v = ast.literal_eval(s)
                if isinstance(v, dict):
                    return v
            except Exception:
                return None
    return None


def _strip_trailing_index(doc_id: str) -> List[str]:
    """chunk id에서 원본 id 후보들을 만든다.

    예:
      announcements_12_3 -> announcements_12
      announcements_12_3_1 -> announcements_12_3 -> announcements_12
    """
    out = []
    s = doc_id
    out.append(s)
    for _ in range(3):
        if "_" not in s:
            break
        head, tail = s.rsplit("_", 1)
        if tail.isdigit():
            s = head
            out.append(s)
        else:
            break
    # 중복 제거
    uniq = []
    for x in out:
        if x not in uniq:
            uniq.append(x)
    return uniq


def _extract_metadata_field(series: pd.Series, key: str) -> pd.Series:
    def _get(x: Any) -> Any:
        d = _as_dict(x)
        if not d:
            return None
        return d.get(key)

    return series.apply(_get)


def _first_nonempty(row: pd.Series, keys: List[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _recover_data_type(corpus_df: pd.DataFrame, parsed_df: pd.DataFrame) -> pd.DataFrame:
    """corpus_df에 data_type이 없을 때 최대한 복구한다."""
    if "data_type" in corpus_df.columns and not corpus_df["data_type"].isna().all():
        if not corpus_df["data_type"].isna().any():
            return corpus_df

    # 1) metadata에 data_type이 있으면 가장 먼저 사용
    if "metadata" in corpus_df.columns:
        dt_from_meta = _extract_metadata_field(corpus_df["metadata"], "data_type")
        if dt_from_meta.notna().any():
            corpus_df = corpus_df.copy()
            corpus_df["data_type"] = dt_from_meta
            return corpus_df

    # 2) parsed에서 (doc_id -> data_type) 매핑 만들기
    if "doc_id" in parsed_df.columns and "data_type" in parsed_df.columns:
        id_to_type = dict(zip(parsed_df["doc_id"].astype(str), parsed_df["data_type"].astype(str)))
    else:
        raise RuntimeError("parsed.parquet에 doc_id/data_type 컬럼이 없습니다.")

    # 3) corpus에서 원본 doc_id 추정
    #    - metadata에 doc_id 있으면 그걸 우선
    source_doc_id = None
    if "metadata" in corpus_df.columns:
        source_doc_id = _extract_metadata_field(corpus_df["metadata"], "doc_id")
        if source_doc_id is not None and source_doc_id.notna().any():
            source_doc_id = source_doc_id.astype(str)

    #    - 없으면 corpus의 id 컬럼에서 추정
    id_cols = [c for c in ("doc_id", "id", "document_id", "chunk_id") if c in corpus_df.columns]
    if source_doc_id is None or source_doc_id.isna().all():
        if not id_cols:
            raise RuntimeError("corpus parquet에 doc_id/id 컬럼이 없습니다.")
        base_col = id_cols[0]
        source_doc_id = corpus_df[base_col].astype(str)

    # 4) 매칭해서 data_type 채우기
    def _lookup(dt_id: str) -> Optional[str]:
        for cand in _strip_trailing_index(dt_id):
            if cand in id_to_type:
                return id_to_type[cand]
        return None

    recovered = source_doc_id.apply(_lookup)

    hit_rate = float(recovered.notna().mean())
    print(f"🧩 data_type 복구 매칭률: {hit_rate*100:.1f}%")

    if hit_rate < 0.2:
        # 너무 낮으면, 거의 못 맞춘 거라 안전하게 중단
        raise RuntimeError(
            "data_type 복구에 실패했습니다(매칭률이 너무 낮음).\n"
            "→ 해결: chunk 결과(corpus)에 metadata(doc_id/data_type) 유지되도록 하거나,\n"
            "  doc_id 네이밍(원본id_청크번호) 규칙을 확인하세요.\n"
            "  (빠른 해결) scripts/patch_corpus_datatype.py로 corpus를 먼저 패치하세요."
        )

    corpus_df = corpus_df.copy()
    corpus_df["data_type"] = recovered
    return corpus_df


def _pick_text_column(df: pd.DataFrame) -> str:
    for col in ("contents", "text", "texts"):
        if col in df.columns:
            return col
    raise RuntimeError("corpus parquet에 contents/text/texts 컬럼이 없습니다.")


def _get_row_metadata(row: pd.Series) -> Dict[str, Any]:
    for key in ("metadata",):
        value = row.get(key)
        meta = _as_dict(value)
        if isinstance(meta, dict):
            return meta
    return {}


def _normalize_title_key(title: str) -> str:
    return " ".join(str(title or "").split()).strip().lower()


def _build_title_docid_lookup(parsed_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    lookup: Dict[str, Dict[str, str]] = {}
    for _, row in parsed_df.iterrows():
        dtype = str(row.get("data_type") or "").strip()
        doc_id = str(row.get("doc_id") or "").strip()
        meta = _as_dict(row.get("metadata")) or {}
        title = str(meta.get("title") or "").strip()
        key = _normalize_title_key(title)
        if not (dtype and doc_id and key):
            continue
        lookup.setdefault(dtype, {})
        lookup[dtype].setdefault(key, doc_id)
    return lookup


def _is_valid_base_doc_id(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    return text.lower() not in GENERIC_TYPE_IDS


def _infer_base_doc_id(row: pd.Series, dtype: str, title_lookup: Dict[str, Dict[str, str]]) -> str:
    meta = _get_row_metadata(row)
    for key in ("doc_id", "policy_id", "id"):
        value = meta.get(key)
        if _is_valid_base_doc_id(value):
            return str(value).strip()

    candidate = _first_nonempty(row, ["doc_id", "id", "document_id", "chunk_id"])
    if _is_valid_base_doc_id(candidate):
        stripped = _strip_trailing_index(candidate)[-1]
        if _is_valid_base_doc_id(stripped):
            return stripped

    title = str(meta.get("title") or _first_nonempty(row, ["title", "name"])).strip()
    normalized_title = _normalize_title_key(title)
    if normalized_title and dtype in title_lookup:
        matched = title_lookup[dtype].get(normalized_title)
        if _is_valid_base_doc_id(matched):
            return matched

    return ""


def _canonicalize_dtype(value: Any) -> str:
    text = str(value or "").strip().lower()
    return CANONICAL_TYPES.get(text, text)


def _normalize_dtype_columns(df: pd.DataFrame) -> pd.DataFrame:
    if "data_type" not in df.columns:
        return df
    out = df.copy()
    out["data_type"] = out["data_type"].map(_canonicalize_dtype)
    return out


def _samples_for_dtype(dtype: str, samples_per_type: int) -> int:
    if dtype in POLICY_TYPES:
        return max(samples_per_type * POLICY_SAMPLE_MULTIPLIER, samples_per_type)
    return samples_per_type


def _parse_date_like(value: Any) -> Optional[date]:
    text = str(value or "").strip()
    if not text:
        return None
    text = text.replace(".", "-").replace("/", "-")
    text = re.sub(r"[^\d\-]", "", text)
    for fmt in ("%Y-%m-%d", "%Y%m%d", "%y-%m-%d", "%y%m%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except Exception:
            continue
    return None


def _extract_period_end_date_from_text(value: Any) -> Optional[date]:
    text = str(value or "")
    if not text:
        return None
    candidates = re.findall(r"(20\d{2}[.\-/]\d{1,2}[.\-/]\d{1,2}|20\d{6})", text)
    parsed = [_parse_date_like(item) for item in candidates]
    parsed = [item for item in parsed if item is not None]
    return max(parsed) if parsed else None


def _policy_doc_is_open(row: pd.Series, today: Optional[date] = None) -> bool:
    today = today or date.today()
    meta = _get_row_metadata(row)
    status_text = " ".join(
        str(x or "")
        for x in (
            meta.get("status"),
            meta.get("biz_pbanc_stts_nm"),
            meta.get("pbanc_stts_nm"),
        )
    ).strip()
    if status_text:
        lowered = status_text.lower()
        if any(token in lowered for token in ("마감", "종료", "closed", "expired")):
            return False
        if any(token in lowered for token in ("모집중", "진행중", "접수중", "open", "ongoing")):
            return True

    end_date = None
    for key in (
        "deadline",
        "confmdoc_expr_dt",
        "biz_pbanc_end_dt",
        "pbanc_end_dt",
        "end_date",
        "apply_end_date",
        "llm_deadline_end_date",
    ):
        end_date = _parse_date_like(meta.get(key) or row.get(key))
        if end_date is not None:
            break

    if end_date is None:
        for key in ("apply_period", "period", "recruit_period", "text"):
            end_date = _extract_period_end_date_from_text(meta.get(key) or row.get(key))
            if end_date is not None:
                break

    if end_date is None:
        # Unknown deadline should stay eligible for QA generation.
        return True
    return end_date >= today


def _exclude_expired_policy_rows(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    if dtype not in POLICY_TYPES or df.empty:
        return df
    today = date.today()
    mask = df.apply(lambda row: _policy_doc_is_open(row, today=today), axis=1)
    kept = df[mask].copy()
    dropped = int((~mask).sum())
    if dropped:
        print(f"🗓️ {dtype}: expired policy docs excluded {dropped}건 (kept={len(kept)})")
    return kept if not kept.empty else df


def _build_fallback_query(dtype: str, title: str, text: str) -> str:
    title = title or "해당 항목"
    prompts = {
        "business": f"{title} 사업의 지원 대상과 지원 내용은 무엇인가?",
        "announcements": f"{title} 공고의 지원 대상과 접수 정보는 무엇인가?",
        "content": f"{title} 자료의 핵심 내용은 무엇인가?",
        "statistical": f"{title} 통계자료의 핵심 내용은 무엇인가?",
        "edu_lectures": f"{title} 강좌의 주제와 학습 내용은 무엇인가?",
        "spaces": f"{title} 공간의 위치와 이용 정보는 무엇인가?",
        "centers": f"{title} 센터의 위치와 지원 정보는 무엇인가?",
        "products": f"{title} 제품 확인 정보의 핵심 내용은 무엇인가?",
        "corporates": f"{title} 기업 확인 정보의 핵심 내용은 무엇인가?",
        "institutions": f"{title} 기관의 역할과 기본 정보는 무엇인가?",
    }
    query = prompts.get(dtype, f"{title}의 핵심 내용은 무엇인가?")
    if len(query) < 10 and text:
        return f"{title}에 대해 설명해줘."
    return query


def _flatten_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    while isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
        if hasattr(value, "tolist"):
            try:
                value = value.tolist()
            except Exception:
                pass
    return value


def _normalize_query_text(value: Any) -> str:
    value = _flatten_scalar(value)
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def _normalize_generation_text(value: Any) -> str:
    value = _flatten_scalar(value)
    if value is None:
        return ""
    text = " ".join(str(value).split()).strip()
    prefixes = [
        "Based on the provided text, ",
        "Based on the information provided in the text, ",
        "According to the provided text, ",
    ]
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text


def _normalize_retrieval_gt(value: Any) -> list[str]:
    value = _flatten_scalar(value)
    if value is None:
        return []
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            out.extend(_normalize_retrieval_gt(item))
        uniq: list[str] = []
        for item in out:
            if item and item not in uniq:
                uniq.append(item)
        return uniq
    text = str(value).strip()
    text = re.sub(r"__chunk_\d+$", "", text)
    return [text] if text else []


def _postprocess_autorag_qa_df(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    out = df.copy()
    if "query" in out.columns:
        out["query"] = out["query"].map(_normalize_query_text)
    if "generation_gt" in out.columns:
        out["generation_gt"] = out["generation_gt"].map(_normalize_generation_text)
    if "concise_generation_gt" in out.columns:
        out["concise_generation_gt"] = out["concise_generation_gt"].map(_normalize_generation_text)
    if "retrieval_gt" in out.columns:
        out["retrieval_gt"] = out["retrieval_gt"].map(_normalize_retrieval_gt)
    out["data_type"] = dtype
    return out


def _build_fallback_answer(text: str, title: str) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return f"{title or '해당 항목'}에 대한 문서 내용이 충분하지 않습니다."
    if len(cleaned) <= 320:
        return cleaned
    cut = cleaned[:320].rsplit(" ", 1)[0].strip()
    return f"{cut}..."


def _generate_fallback_qa_frames(
    parsed_df: pd.DataFrame,
    samples_per_type: int,
    dtypes: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    target_dtypes = dtypes or DATA_TYPES

    for dtype in target_dtypes:
        subset = parsed_df[parsed_df["data_type"] == dtype].copy()
        subset = _exclude_expired_policy_rows(subset, dtype)
        if subset.empty:
            print(f"⚠️ {dtype} 데이터 없음 -> 건너뜀")
            continue

        n_samples = min(_samples_for_dtype(dtype, samples_per_type), len(subset))
        sample_df = subset.head(n_samples).reset_index(drop=True)
        rows: List[Dict[str, Any]] = []

        for idx, row in sample_df.iterrows():
            meta = _get_row_metadata(row)
            title = (
                str(meta.get("title") or "").strip()
                or _first_nonempty(row, ["title", "name"])
                or f"{dtype}_{idx}"
            )
            text = str(
                row.get("contents")
                or row.get("texts")
                or row.get("text")
                or ""
            ).strip()
            retrieval_gt = str(row.get("doc_id") or "").strip()
            rows.append(
                {
                    "qid": f"{dtype}_{idx}",
                    "query": _build_fallback_query(dtype, title, text),
                    "question": _build_fallback_query(dtype, title, text),
                    "generation_gt": _build_fallback_answer(text, title),
                    "concise_generation_gt": _build_fallback_answer(text, title),
                    "retrieval_gt": [retrieval_gt] if retrieval_gt else [],
                    "retrieval_gt_contents": text,
                    "data_type": dtype,
                }
            )

        frames.append(pd.DataFrame(rows))

    return frames


# -------------------------
# LLM builder
# -------------------------

def build_llm(provider: str, openai_model: str, local_model: str):
    """LLM provider에 따라 LLM 객체 생성."""
    if provider == "openai":
        if not Config.LLM_API_KEY:
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        from llama_index.llms.openai import OpenAI

        return OpenAI(model=openai_model, api_key=Config.LLM_API_KEY)

    if provider == "local":
        try:
            from llama_index.llms.huggingface import HuggingFaceLLM
        except ImportError as exc:
            raise RuntimeError("로컬 LLM 사용을 위해 llama-index[huggingface] 설치가 필요합니다.") from exc

        # ✅ tokenizer mismatch(StableLM 등) 경고 방지: tokenizer_name을 model과 동일하게 고정
        # llama-index 버전에 따라 HuggingFaceLLM 인자명이 조금씩 다를 수 있어서 fallback을 둡니다.
        try:
            return HuggingFaceLLM(
                model_name=local_model,
                tokenizer_name=local_model,
                device_map="auto",
                model_kwargs={"trust_remote_code": True},
                tokenizer_kwargs={"trust_remote_code": True},
            )
        except TypeError:
            # 구버전에서는 tokenizer_kwargs/model_kwargs를 안 받는 경우가 있음
            return HuggingFaceLLM(
                model_name=local_model,
                tokenizer_name=local_model,
                device_map="auto",
            )

    raise ValueError(f"지원하지 않는 LLM provider: {provider}")


# -------------------------
# main
# -------------------------

def generate_qa_dataset(
    samples_per_type: int = 15,
    llm_provider: str = "openai",
    openai_model: str = "gpt-4o-mini",
    local_model: str = "Qwen/Qwen2.5-3B-Instruct",
    corpus_path: str | None = None,
    use_prompt_override: bool = False,
    qa_output_name: str = "qa.parquet",
    corpus_output_name: str = "corpus.parquet",
) -> None:
    print(f"🤖 유형별 {samples_per_type}개 QA 생성 (provider={llm_provider})")
    if use_prompt_override:
        print("🧩 AutoRAG 기본 prompt 대신 domain-adapted prompt override를 사용합니다.")
        _override_autorag_prompts()
    else:
        print("📏 AutoRAG framework 기본 prompt를 그대로 사용합니다. (baseline)")
    llm = None
    if AUTORAG_QA_PIPELINE_AVAILABLE:
        llm = build_llm(llm_provider, openai_model, local_model)
    else:
        joined = "; ".join(_AUTORAG_IMPORT_ERRORS) or "unknown import error"
        print("⚠️ 설치된 AutoRAG 버전과 QA 생성 모듈 경로가 맞지 않아 fallback QA 생성으로 진행합니다.")
        print(f"   - 상세: {joined}")

    raw_path = PROJECT_ROOT / "autorag_workspace" / "parsed" / "parsed.parquet"
    corpus_dir = PROJECT_ROOT / "autorag_workspace" / "corpus"

    if corpus_path:
        corpus_path_p = (PROJECT_ROOT / corpus_path).resolve() if not Path(corpus_path).is_absolute() else Path(corpus_path)
    else:
        corpus_path_p = _glob_latest_parquet(corpus_dir)

    qa_dir = PROJECT_ROOT / "autorag_workspace" / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    final_qa_path = qa_dir / qa_output_name
    final_corpus_copy_path = qa_dir / corpus_output_name

    print(f"📂 parsed: {raw_path}")
    print(f"📂 corpus: {corpus_path_p}")

    print("📂 데이터 로딩 중...")
    raw_df = _normalize_dtype_columns(pd.read_parquet(raw_path))
    if "data_type" not in raw_df.columns:
        raise RuntimeError("parsed.parquet에 data_type 컬럼이 없습니다. export_to_autorag.py를 다시 실행하세요.")
    raw_instance = Raw(raw_df)

    corpus_df = _normalize_dtype_columns(pd.read_parquet(corpus_path_p))
    if "data_type" not in corpus_df.columns or corpus_df["data_type"].isna().all():
        print("⚠️ corpus parquet에 data_type이 없어 parsed 데이터로 복구합니다.")
        corpus_df = _recover_data_type(corpus_df, raw_df)
        corpus_df = _normalize_dtype_columns(corpus_df)

    qa_frames = []
    if AUTORAG_QA_PIPELINE_AVAILABLE and llm is not None:
        for dtype in DATA_TYPES:
            subset = corpus_df[corpus_df["data_type"] == dtype].copy()
            subset = _exclude_expired_policy_rows(subset, dtype)
            if subset.empty:
                if dtype in POLICY_TYPES:
                    print(f"?? {dtype} corpus ???? ?? parsed ?? fallback QA? ?????.")
                    qa_frames.extend(
                        _generate_fallback_qa_frames(
                            raw_df,
                            samples_per_type,
                            dtypes=[dtype],
                        )
                    )
                else:
                    print(f"?? {dtype} ??? ?? ? ???")
                continue

            n_samples = min(_samples_for_dtype(dtype, samples_per_type), len(subset))
            if n_samples <= 0:
                continue

            print(f"⚙️ {dtype}: {n_samples}개 샘플 생성 중...")
            try:
                subset_corpus = Corpus(subset, raw_instance)
                qa_pipeline = (
                    subset_corpus
                    .sample(random_single_hop, n=n_samples)
                    .map(lambda df: df.reset_index(drop=True))
                    .make_retrieval_gt_contents()
                    .batch_apply(factoid_query_gen, llm=llm, lang="ko")
                    .batch_apply(make_basic_gen_gt, llm=llm, lang="ko")
                    .batch_apply(make_concise_gen_gt, llm=llm, lang="ko")
                    .filter(dontknow_filter_rule_based, lang="ko")
                )

                tmp_qa_path = qa_dir / f"_tmp_{dtype}_qa.parquet"
                tmp_corpus_path = qa_dir / f"_tmp_{dtype}_corpus.parquet"
                qa_pipeline.to_parquet(str(tmp_qa_path), str(tmp_corpus_path))
                tmp_df = pd.read_parquet(tmp_qa_path)
                tmp_df = _postprocess_autorag_qa_df(tmp_df, dtype)
                qa_frames.append(tmp_df)
                tmp_qa_path.unlink(missing_ok=True)
                tmp_corpus_path.unlink(missing_ok=True)
            except Exception as exc:
                print(f"⚠️ {dtype} AutoRAG QA 생성 실패 -> fallback 사용 ({type(exc).__name__}: {exc})")
                qa_frames.extend(
                    _generate_fallback_qa_frames(
                        raw_df,
                        samples_per_type,
                        dtypes=[dtype],
                    )
                )
    else:
        qa_frames = _generate_fallback_qa_frames(raw_df, samples_per_type)

    if not qa_frames:
        raise RuntimeError("생성된 QA 데이터가 없습니다. 데이터 수집/청킹 단계를 확인하세요.")

    final_df = pd.concat(qa_frames, ignore_index=True)
    final_df.to_parquet(final_qa_path)

    # 평가용으로 corpus도 qa 폴더로 복사
    shutil.copyfile(corpus_path_p, final_corpus_copy_path)

    print("✅ QA 데이터 생성 완료!")
    print(f"  - QA: {final_qa_path}")
    print(f"  - Corpus copy: {final_corpus_copy_path}")

    print(f"\n📊 생성된 QA 샘플 ({len(final_df)}개 중 3개):")
    for idx, row in final_df.head(3).iterrows():
        print(f"\n[{row.get('data_type')}] 질문 {idx+1}: {row.get('query', 'N/A')}")
        answer = row.get("generation_gt", "N/A")
        print(f"답변: {str(answer)[:120]}...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoRAG QA 데이터 생성")
    parser.add_argument("--samples-per-type", type=int, default=5, help="데이터 타입별 생성할 QA 수")
    parser.add_argument("--llm-provider", choices=["openai", "local"], default="openai")
    parser.add_argument("--llm-model", default="gpt-4o-mini", help="OpenAI 모델명")
    parser.add_argument("--local-model", default="Qwen/Qwen2.5-3B-Instruct", help="로컬 HuggingFace 모델명")
    parser.add_argument(
        "--use-prompt-override",
        action="store_true",
        help="AutoRAG 기본 prompt 대신 한국 창업지원 도메인용 override prompt를 사용",
    )
    parser.add_argument(
        "--corpus-path",
        default=None,
        help="chunk 결과 corpus parquet 경로(없으면 autorag_workspace/corpus 아래 최신 parquet 자동 선택)",
    )
    parser.add_argument("--qa-output-name", default="qa.parquet", help="autorag_workspace/qa 아래 저장할 QA parquet 파일명")
    parser.add_argument("--corpus-output-name", default="corpus.parquet", help="autorag_workspace/qa 아래 저장할 corpus 복사본 파일명")
    args = parser.parse_args()

    generate_qa_dataset(
        samples_per_type=args.samples_per_type,
        llm_provider=args.llm_provider,
        openai_model=args.llm_model,
        local_model=args.local_model,
        corpus_path=args.corpus_path,
        use_prompt_override=args.use_prompt_override,
        qa_output_name=args.qa_output_name,
        corpus_output_name=args.corpus_output_name,
    )
