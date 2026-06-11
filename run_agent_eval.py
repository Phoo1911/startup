from __future__ import annotations

import argparse
import ast
import json
import re
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd
from dotenv import load_dotenv

from agentic_hybrid.config import apply_experiment_mode, load_config
from agentic_hybrid.graph import build_agentic_graph
from agentic_hybrid.main_agentic import SimpleLLM
from agentic_hybrid.state import init_state

DEFAULT_QA_PATH = Path("autorag_workspace/qa/qa.parquet")
DEFAULT_OUTPUT_PATH = Path("autorag_workspace/results_agent/predictions.parquet")
DEFAULT_RETRIEVAL_SAVE_K = 20
DEFAULT_MODES = (
    "baseline",
    "baseline_pure",
    "autorag",
    "full",
    "full_dense",
    "no_intent_dense",
    "no_planner_dense",
    "no_doc_type_router_dense",
    "no_revise_dense",
    "full_without_filter",
    "full_without_doc_type_router",
    "no_intent",
    "no_hyde",
    "no_reranker",
    "no_freshness",
    "no_deadline",
)


def _pick_question_column(df: pd.DataFrame) -> str:
    for col in ("question", "query"):
        if col in df.columns:
            return col
    raise ValueError("qa.parquet must contain either 'question' or 'query' column")


def _normalize_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if isinstance(value, (list, tuple, dict)):
        return value
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        if s[0] in "[{(" and s[-1] in "]})":
            for parser in (json.loads, ast.literal_eval):
                try:
                    return parser(s)
                except Exception:
                    pass
        return s
    return value


def _flatten_str_list(value: Any) -> List[str]:
    obj = _normalize_obj(value)
    if obj is None:
        return []
    if isinstance(obj, dict):
        out: List[str] = []
        for v in obj.values():
            out.extend(_flatten_str_list(v))
        return out
    if isinstance(obj, (list, tuple)):
        out: List[str] = []
        for item in obj:
            out.extend(_flatten_str_list(item))
        return [x for x in out if x]
    s = str(obj).strip()
    return [s] if s else []


def _coerce_text(value: Any) -> str:
    obj = _normalize_obj(value)
    if obj is None:
        return ""
    if isinstance(obj, dict):
        try:
            return json.dumps(obj, ensure_ascii=False)
        except Exception:
            return str(obj)
    if isinstance(obj, (list, tuple)):
        parts = [x for x in _flatten_str_list(obj) if x]
        return " | ".join(parts)
    text = str(obj).strip()
    return text


def _normalize_eval_doc_id(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return re.sub(r"__chunk_\d+$", "", text)


def _unique_keep_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _extract_doc_ids(doc: Dict[str, Any]) -> List[str]:
    md = doc.get("metadata", {}) or {}
    candidates: List[str] = []
    for key in (
        "policy_id",
        "doc_id",
        "id",
        "qid",
        "pbanc_sn",
        "confmdoc_isu_no",
        "spce_id",
        "cntr_id",
    ):
        value = doc.get(key)
        if value not in (None, ""):
            candidates.extend(_flatten_str_list(value))
        value = md.get(key)
        if value not in (None, ""):
            candidates.extend(_flatten_str_list(value))

    raw = md.get("raw")
    if isinstance(raw, dict):
        for key in ("id", "pbanc_sn", "confmdoc_isu_no", "spce_id", "cntr_id"):
            value = raw.get(key)
            if value not in (None, ""):
                candidates.extend(_flatten_str_list(value))

    return _unique_keep_order([c.strip() for c in candidates if str(c).strip()])


def _primary_doc_id(doc: Dict[str, Any]) -> str:
    md = doc.get("metadata", {}) or {}
    preferred = [
        md.get("doc_id"),
        doc.get("doc_id"),
        md.get("policy_id"),
        doc.get("policy_id"),
    ]
    for value in preferred:
        ids = [_normalize_eval_doc_id(x) for x in _flatten_str_list(value) if _normalize_eval_doc_id(x)]
        if ids:
            return ids[0]

    extracted = [_normalize_eval_doc_id(x) for x in _extract_doc_ids(doc) if _normalize_eval_doc_id(x)]
    return extracted[0] if extracted else ""


def _compact_docs(docs: list[Dict[str, Any]], max_docs: int = DEFAULT_RETRIEVAL_SAVE_K) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for d in docs[:max_docs]:
        md = d.get("metadata", {}) or {}
        out.append(
            {
                "id": d.get("id"),
                "title": d.get("title"),
                "text": str(d.get("text") or "")[:1000],
                "score": d.get("score") or d.get("combined_score") or d.get("cross_encoder_score"),
                "metadata": {
                    "type": md.get("type"),
                    "deadline": md.get("deadline"),
                    "region": md.get("region"),
                    "policy_id": md.get("policy_id"),
                    "doc_id": md.get("doc_id"),
                    "id": md.get("id"),
                },
            }
        )
    return out


def _compact_triplets(items: list[Dict[str, Any]], max_items: int = 20) -> list[Dict[str, Any]]:
    out: list[Dict[str, Any]] = []
    for item in items[:max_items]:
        out.append(
            {
                "doc_id": item.get("doc_id"),
                "title": item.get("title"),
                "candidate_answer": str(item.get("candidate_answer") or "")[:400],
                "judge_verdict": item.get("judge_verdict"),
                "judge_score": item.get("judge_score"),
                "judge_reason": item.get("judge_reason"),
            }
        )
    return out


def _build_runtime(mode: str):
    cfg = apply_experiment_mode(load_config(), mode)
    llm = SimpleLLM(cfg)
    app = build_agentic_graph(cfg, llm)
    return cfg, llm, app


def _run_single(app: Any, question: str, skip_intent_classifier: bool = False) -> tuple[Dict[str, Any], float]:
    t0 = time.perf_counter()
    result = app.invoke(
        init_state(
            question,
            selected_doc_types=[],
            skip_intent_classifier=skip_intent_classifier,
        )
    )
    latency = time.perf_counter() - t0
    return result, latency


def run_batch_eval(qa_path: Path, output_path: Path, modes: Iterable[str], limit: int | None) -> pd.DataFrame:
    if not qa_path.exists():
        raise FileNotFoundError(f"QA file not found: {qa_path}")

    load_dotenv()
    qa_df = pd.read_parquet(qa_path)
    question_col = _pick_question_column(qa_df)
    work_df = qa_df.copy()
    work_df[question_col] = work_df[question_col].astype(str).map(str.strip)
    work_df = work_df[work_df[question_col] != ""]
    if limit is not None:
        work_df = work_df.iloc[:limit]

    modes = [m.strip().lower() for m in modes if m and str(m).strip()]
    total = len(work_df) * len(modes)
    print(f"[INFO] loaded {len(work_df)} questions from {qa_path} (column={question_col})")
    print(f"[INFO] running {len(modes)} modes x {len(work_df)} questions = {total} jobs")
    print("[INFO] AutoRAG link is disabled in batch mode by direct graph invocation.")

    runtime_cache: dict[str, Any] = {}
    rows: list[dict] = []
    job_idx = 0

    for q_idx, (_, row) in enumerate(work_df.iterrows(), start=1):
        question = _coerce_text(row.get(question_col))
        gold_policy_ids = _unique_keep_order(
            [_normalize_eval_doc_id(x) for x in _flatten_str_list(row.get("retrieval_gt")) if _normalize_eval_doc_id(x)]
        )
        generation_gt = _coerce_text(row.get("generation_gt"))
        concise_generation_gt = _coerce_text(row.get("concise_generation_gt"))
        retrieval_gt_contents = _coerce_text(row.get("retrieval_gt_contents"))
        data_type = _coerce_text(row.get("data_type"))

        for mode in modes:
            job_idx += 1
            print(f"[RUN] {job_idx}/{total} q={q_idx} mode={mode}")
            try:
                if mode not in runtime_cache:
                    runtime_cache[mode] = _build_runtime(mode)
                _, _, app = runtime_cache[mode]
                result, latency = _run_single(
                    app,
                    question,
                    skip_intent_classifier=(
                        mode in {
                            "baseline",
                            "baseline_pure",
                            "no_intent",
                            "no_intent_dense",
                            "no_planner_dense",
                            "no_doc_type_router_dense",
                            "no_revise_dense",
                        }
                    ),
                )

                answer = str(result.get("answer") or "")
                final_docs = list(result.get("final_docs") or result.get("filtered_docs") or [])
                retrieved_docs = list(result.get("retrieved_docs") or final_docs)
                docs_for_prediction = final_docs or retrieved_docs
                predicted_ids = _unique_keep_order(
                    [doc_id for d in docs_for_prediction for doc_id in [_primary_doc_id(d)] if doc_id]
                )

                completeness_check = dict(result.get("completeness_check") or {})

                rows.append(
                    {
                        "qid": row.get("qid"),
                        "question": question,
                        "mode": mode,
                        "data_type": data_type,
                        "answer": answer,
                        "generation_gt": generation_gt,
                        "concise_generation_gt": concise_generation_gt,
                        "retrieval_gt_contents": retrieval_gt_contents,
                        "latency": latency,
                        "gold_policy_ids": gold_policy_ids,
                        "predicted_policy_ids": predicted_ids,
                        "retrieved_docs": _compact_docs(retrieved_docs, max_docs=DEFAULT_RETRIEVAL_SAVE_K),
                        "doc_answer_triplets": _compact_triplets(list(result.get("doc_answer_triplets") or [])),
                        "completeness_check": completeness_check if completeness_check else None,
                        "followup_docs": _compact_docs(list(result.get("followup_docs") or []), max_docs=10),
                    }
                )
                print(f"[OK ] {job_idx}/{total} mode={mode} latency={latency:.3f}s pred={len(predicted_ids)} gold={len(gold_policy_ids)}")
            except Exception as exc:
                error_text = "".join(
                    traceback.format_exception_only(type(exc), exc)
                ).strip()
                trace_text = traceback.format_exc().strip()
                rows.append(
                    {
                        "qid": row.get("qid"),
                        "question": question,
                        "mode": mode,
                        "data_type": data_type,
                        "answer": f"[ERROR] {error_text}",
                        "generation_gt": generation_gt,
                        "concise_generation_gt": concise_generation_gt,
                        "retrieval_gt_contents": retrieval_gt_contents,
                        "latency": float("nan"),
                        "gold_policy_ids": gold_policy_ids,
                        "predicted_policy_ids": [],
                        "retrieved_docs": [],
                        "doc_answer_triplets": [],
                        "completeness_check": {"traceback": trace_text},
                        "followup_docs": [],
                    }
                )
                print(f"[ERR] {job_idx}/{total} mode={mode} {error_text}")

    out_df = pd.DataFrame(rows, columns=[
        "qid", "question", "mode", "data_type", "answer", "generation_gt", "concise_generation_gt",
        "retrieval_gt_contents", "latency", "gold_policy_ids", "predicted_policy_ids", "retrieved_docs",
        "doc_answer_triplets", "completeness_check", "followup_docs",
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    print(f"[DONE] saved {len(out_df)} rows to {output_path}")
    return out_df


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-evaluate agentic_hybrid across experiment modes")
    parser.add_argument("--qa-path", type=Path, default=DEFAULT_QA_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of questions")
    args = parser.parse_args()

    modes = [str(m).strip().lower() for m in args.modes if str(m).strip()]
    allowed = {
        "baseline",
        "baseline_pure",
        "autorag",
        "full",
        "full_dense",
        "no_intent_dense",
        "no_planner_dense",
        "no_doc_type_router_dense",
        "no_revise_dense",
        "full_without_filter",
        "full_without_doc_type_router",
        "no_intent",
        "no_hyde",
        "no_reranker",
        "no_freshness",
        "no_deadline",
    }
    invalid = [m for m in modes if m not in allowed]
    if invalid:
        raise ValueError(f"Unsupported modes: {invalid}. Allowed: {sorted(allowed)}")

    run_batch_eval(qa_path=args.qa_path, output_path=args.output_path, modes=modes, limit=args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
