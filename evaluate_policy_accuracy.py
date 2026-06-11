from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

DEFAULT_INPUT = Path("autorag_workspace/results_agent/predictions.parquet")
DEFAULT_OUTPUT = Path("autorag_workspace/results_agent/policy_metrics.csv")


def _as_obj(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (list, tuple, set, dict)):
        return value
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
        if "," in s:
            return [part.strip() for part in s.split(",") if part.strip()]
        return s
    return value


def _flatten_str_list(value: Any) -> List[str]:
    obj = _as_obj(value)
    if obj is None:
        return []
    if isinstance(obj, dict):
        out: List[str] = []
        for item in obj.values():
            out.extend(_flatten_str_list(item))
        return out
    if isinstance(obj, (list, tuple, set)):
        out: List[str] = []
        for item in obj:
            out.extend(_flatten_str_list(item))
        return out
    s = str(obj).strip()
    return [s] if s else []


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


def _to_grouped_id_list(value: Any) -> List[List[str]]:
    obj = _as_obj(value)
    if obj is None:
        return []
    if isinstance(obj, dict):
        groups: List[List[str]] = []
        for item in obj.values():
            groups.extend(_to_grouped_id_list(item))
        return groups
    if isinstance(obj, (list, tuple, set)):
        seq = list(obj)
        if not seq:
            return []
        if any(isinstance(item, (list, tuple, set, dict)) for item in seq):
            groups: List[List[str]] = []
            for item in seq:
                group = _flatten_str_list(item)
                group = [_normalize_eval_doc_id(x) for x in group if _normalize_eval_doc_id(x)]
                if group:
                    groups.append(_unique_keep_order(group))
            return groups
        flat = _unique_keep_order(
            [_normalize_eval_doc_id(x) for x in _flatten_str_list(seq) if _normalize_eval_doc_id(x)]
        )
        return [[item] for item in flat]
    flat = [_normalize_eval_doc_id(x) for x in _flatten_str_list(obj) if _normalize_eval_doc_id(x)]
    return [[item] for item in flat]


def _safe_div(num: float, den: float) -> float:
    return num / den if den else 0.0


def _dcg(binary_relevance: List[int]) -> float:
    score = 0.0
    for idx, rel in enumerate(binary_relevance, start=1):
        if rel:
            score += rel / math.log2(idx + 1)
    return score


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _tokenize_text(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(str(text or ""))]


def _extract_gold_title(row: pd.Series) -> str:
    raw = str(row.get("retrieval_gt_contents") or "").strip()
    if not raw:
        return ""
    for line in raw.splitlines():
        text = str(line or "").strip()
        if text:
            return text
    return ""


def _extract_title_from_source_row(row: pd.Series) -> str:
    for key in ("title", "name", "titl_nm"):
        value = str(row.get(key) or "").strip()
        if value:
            return value
    md = _as_obj(row.get("metadata"))
    if isinstance(md, dict):
        for key in ("title", "name", "titl_nm"):
            value = str(md.get(key) or "").strip()
            if value:
                return value
    contents = str(row.get("contents") or "").strip()
    if contents:
        for line in contents.splitlines():
            text = str(line or "").strip()
            if text:
                return text[:200]
    return ""


def _load_gold_title_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Gold source file not found: {path}")

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    if "doc_id" not in df.columns:
        raise ValueError(f"Gold source file must contain 'doc_id'. Got: {list(df.columns)}")

    mapping: Dict[str, str] = {}
    for _, row in df.iterrows():
        doc_id = _normalize_eval_doc_id(str(row.get("doc_id") or "").strip())
        if not doc_id or doc_id in mapping:
            continue
        title = _extract_title_from_source_row(row)
        if title:
            mapping[doc_id] = title
    return mapping


def _normalize_title(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "").lower())


def _title_soft_match(gold_title: str, pred_title: str, threshold: float = 0.5) -> bool:
    gold = str(gold_title or "").strip()
    pred = str(pred_title or "").strip()
    if not gold or not pred:
        return False

    gold_norm = _normalize_title(gold)
    pred_norm = _normalize_title(pred)
    if gold_norm == pred_norm:
        return True
    if len(gold_norm) >= 6 and (gold_norm in pred_norm or pred_norm in gold_norm):
        return True

    gold_tokens = set(_tokenize_text(gold))
    pred_tokens = set(_tokenize_text(pred))
    if not gold_tokens or not pred_tokens:
        return False
    overlap = len(gold_tokens & pred_tokens) / max(len(gold_tokens), 1)
    return overlap >= threshold


def _extract_ranked_ids_from_docs(value: Any, top_k: int) -> List[str]:
    obj = _as_obj(value)
    if obj is None:
        return []

    docs = obj if isinstance(obj, list) else [obj]
    ranked: List[str] = []
    for item in docs:
        if not isinstance(item, dict):
            text = _normalize_eval_doc_id(str(item or "").strip())
            if text:
                ranked.append(text)
            continue

        md = item.get("metadata", {}) or {}
        candidates = [
            md.get("doc_id"),
            item.get("doc_id"),
            md.get("policy_id"),
            item.get("policy_id"),
            md.get("id"),
            item.get("id"),
        ]
        found = ""
        for candidate in candidates:
            ids = [_normalize_eval_doc_id(x) for x in _flatten_str_list(candidate) if _normalize_eval_doc_id(x)]
            if ids:
                found = ids[0]
                break
        if found:
            ranked.append(found)

    return _unique_keep_order(ranked)[:top_k]


def _extract_ranked_titles_from_docs(value: Any, top_k: int) -> List[str]:
    obj = _as_obj(value)
    if obj is None:
        return []
    docs = obj if isinstance(obj, list) else [obj]
    titles: List[str] = []
    for item in docs:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or "").strip()
        if title:
            titles.append(title)
    return titles[:top_k]


def _evaluate_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "gold_policy_ids" not in df.columns or "predicted_policy_ids" not in df.columns:
        raise ValueError("Input parquet must contain 'gold_policy_ids' and 'predicted_policy_ids'.")

    rows = []
    for _, row in df.iterrows():
        gold_groups = _to_grouped_id_list(row.get("gold_policy_ids"))
        pred_list = _unique_keep_order(_flatten_str_list(row.get("predicted_policy_ids")))

        union_gold = _unique_keep_order([item for group in gold_groups for item in group])
        gold_set = set(union_gold)

        tp = sum(1 for pred in pred_list if pred in gold_set)
        fp = max(len(pred_list) - tp, 0)
        hit_groups = sum(1 for group in gold_groups if any(pred in set(group) for pred in pred_list))
        fn = max(len(gold_groups) - hit_groups, 0)

        precision = _safe_div(tp, len(pred_list))
        recall = _safe_div(hit_groups, len(gold_groups))
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        top1 = 1.0 if pred_list and any(pred_list[0] in set(group) for group in gold_groups) else 0.0

        rr_values: List[float] = []
        ap_values: List[float] = []
        for group in gold_groups:
            group_set = set(group)
            positions = [i + 1 for i, pred in enumerate(pred_list) if pred in group_set]
            rr_values.append(1.0 / min(positions) if positions else 0.0)

            precision_at_hits: List[float] = []
            hits = 0
            for rank, pred in enumerate(pred_list, start=1):
                if pred in group_set:
                    hits += 1
                    precision_at_hits.append(hits / rank)
            ap_values.append(sum(precision_at_hits) / len(group) if group else 0.0)

        mrr = sum(rr_values) / len(rr_values) if rr_values else 0.0
        map_score = sum(ap_values) / len(ap_values) if ap_values else 0.0

        binary_rel = [1 if pred in gold_set else 0 for pred in pred_list]
        dcg = _dcg(binary_rel)
        ideal_rel = [1] * len(union_gold)
        idcg = _dcg(ideal_rel)
        ndcg = _safe_div(dcg, idcg)

        rows.append({
            "mode": row.get("mode", "all"),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision_row": precision,
            "recall_row": recall,
            "f1_row": f1,
            "top1": top1,
            "mrr_row": mrr,
            "map_row": map_score,
            "ndcg_row": ndcg,
        })

    return pd.DataFrame(rows)


def _evaluate_retrieval_rows(df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if "gold_policy_ids" not in df.columns or "retrieved_docs" not in df.columns:
        raise ValueError("Input parquet must contain 'gold_policy_ids' and 'retrieved_docs'.")

    rows = []
    for _, row in df.iterrows():
        gold_groups = _to_grouped_id_list(row.get("gold_policy_ids"))
        ranked_ids = _extract_ranked_ids_from_docs(row.get("retrieved_docs"), top_k=top_k)

        hit_groups = 0
        rr_values: List[float] = []
        for group in gold_groups:
            group_set = set(group)
            positions = [i + 1 for i, pred in enumerate(ranked_ids) if pred in group_set]
            if positions:
                hit_groups += 1
                rr_values.append(1.0 / min(positions))
            else:
                rr_values.append(0.0)

        recall = _safe_div(hit_groups, len(gold_groups))
        mrr = sum(rr_values) / len(rr_values) if rr_values else 0.0

        rows.append({
            "mode": row.get("mode", "all"),
            "recall_row": recall,
            "mrr_row": mrr,
        })

    return pd.DataFrame(rows)


def _evaluate_retrieval_soft_rows(df: pd.DataFrame, top_k: int, gold_title_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    if "retrieved_docs" not in df.columns:
        raise ValueError("Input parquet must contain 'retrieved_docs'.")

    rows = []
    for _, row in df.iterrows():
        gold_title = ""
        if gold_title_map:
            gold_groups = _to_grouped_id_list(row.get("gold_policy_ids"))
            for group in gold_groups:
                for doc_id in group:
                    mapped = str(gold_title_map.get(doc_id) or "").strip()
                    if mapped:
                        gold_title = mapped
                        break
                if gold_title:
                    break
        if not gold_title:
            gold_title = _extract_gold_title(row)
        ranked_titles = _extract_ranked_titles_from_docs(row.get("retrieved_docs"), top_k=top_k)
        hit_positions = [idx + 1 for idx, title in enumerate(ranked_titles) if _title_soft_match(gold_title, title)]
        binary_rel = [1 if _title_soft_match(gold_title, title) else 0 for title in ranked_titles]
        dcg = _dcg(binary_rel)
        idcg = _dcg([1] * sum(binary_rel))
        ndcg = _safe_div(dcg, idcg)
        precision_at_hits: List[float] = []
        hits = 0
        for rank, rel in enumerate(binary_rel, start=1):
            if rel:
                hits += 1
                precision_at_hits.append(hits / rank)
        map_score = (sum(precision_at_hits) / hits) if hits else 0.0
        rows.append({
            "mode": row.get("mode", "all"),
            "recall_row": 1.0 if hit_positions else 0.0,
            "mrr_row": (1.0 / min(hit_positions)) if hit_positions else 0.0,
            "top1_row": 1.0 if hit_positions and min(hit_positions) == 1 else 0.0,
            "map_row": map_score,
            "ndcg_row": ndcg,
            "gold_title": gold_title,
        })
    return pd.DataFrame(rows)


def _aggregate_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    result_rows = []
    group_keys = ["ALL"] + sorted({str(x) for x in eval_df["mode"].dropna().tolist() if str(x).strip()})

    for key in group_keys:
        g = eval_df if key == "ALL" else eval_df[eval_df["mode"].astype(str) == key]

        tp = int(g["tp"].sum())
        fp = int(g["fp"].sum())
        fn = int(g["fn"].sum())

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) else 0.0
        top1_acc = float(g["top1"].mean()) if len(g) > 0 else 0.0

        result_rows.append({
            "mode": key,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "top1_accuracy": top1_acc,
            "mrr": float(g["mrr_row"].mean()) if len(g) > 0 else 0.0,
            "map": float(g["map_row"].mean()) if len(g) > 0 else 0.0,
            "ndcg": float(g["ndcg_row"].mean()) if len(g) > 0 else 0.0,
            "n_samples": int(len(g)),
        })

    return pd.DataFrame(result_rows)


def _aggregate_retrieval_metrics(eval_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    result_rows = []
    group_keys = ["ALL"] + sorted({str(x) for x in eval_df["mode"].dropna().tolist() if str(x).strip()})

    for key in group_keys:
        g = eval_df if key == "ALL" else eval_df[eval_df["mode"].astype(str) == key]
        row = {
            "mode": key,
            f"recall@{top_k}": float(g["recall_row"].mean()) if len(g) > 0 else 0.0,
            f"mrr@{top_k}": float(g["mrr_row"].mean()) if len(g) > 0 else 0.0,
            "n_samples": int(len(g)),
        }
        if "top1_row" in g.columns:
            row[f"top1@{top_k}"] = float(g["top1_row"].mean()) if len(g) > 0 else 0.0
        if "map_row" in g.columns:
            row[f"map@{top_k}"] = float(g["map_row"].mean()) if len(g) > 0 else 0.0
        if "ndcg_row" in g.columns:
            row[f"ndcg@{top_k}"] = float(g["ndcg_row"].mean()) if len(g) > 0 else 0.0
        result_rows.append(row)

    return pd.DataFrame(result_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate policy recommendation accuracy")
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--evaluation-target", choices=["policy", "retrieval", "retrieval_soft"], default="policy")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--gold-source-path", type=Path, default=None, help="Optional parquet/csv with doc_id to title mapping for retrieval_soft")
    args = parser.parse_args()

    if not args.input_path.exists():
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    df = pd.read_parquet(args.input_path)
    if args.evaluation_target == "retrieval":
        eval_df = _evaluate_retrieval_rows(df, top_k=args.top_k)
        metrics_df = _aggregate_retrieval_metrics(eval_df, top_k=args.top_k)
    elif args.evaluation_target == "retrieval_soft":
        gold_title_map = _load_gold_title_map(args.gold_source_path)
        eval_df = _evaluate_retrieval_soft_rows(df, top_k=args.top_k, gold_title_map=gold_title_map)
        metrics_df = _aggregate_retrieval_metrics(eval_df, top_k=args.top_k)
    else:
        eval_df = _evaluate_rows(df)
        metrics_df = _aggregate_metrics(eval_df)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(args.output_path, index=False, encoding="utf-8-sig")

    title = "Retrieval Summary" if args.evaluation_target in {"retrieval", "retrieval_soft"} else "Policy Accuracy Summary"
    print(title)
    print(metrics_df.to_string(index=False))
    print(f"\nSaved: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
