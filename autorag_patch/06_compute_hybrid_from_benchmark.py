from __future__ import annotations

import argparse
import ast
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd


def _latest_trial(project_dir: Path) -> Path:
    trials = [p for p in project_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not trials:
        raise FileNotFoundError(f"No numeric trial dirs under: {project_dir}")
    return max(trials, key=lambda p: int(p.name))


def _extract_vectordb_name(module_params: Any) -> str:
    text = str(module_params or "")
    match = re.search(r"vectordb['\"]?\s*:\s*['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1)
    match = re.search(r"vectordb_[-a-zA-Z0-9_]+", text)
    if match:
        return match.group(0)
    return "unknown_vectordb"


def _parse_list_cell(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value]
    if hasattr(value, 'tolist'):
        try:
            return [str(x) for x in value.tolist()]
        except Exception:
            pass
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
            return [str(parsed)]
        except Exception:
            return [s]
    return [str(value)]


def _normalize_doc_id(value: str) -> str:
    text = str(value or '').strip()
    if not text:
        return ''
    if '__chunk_' in text:
        return text.split('__chunk_', 1)[0]
    return text


def reciprocal_rank_fusion(rank_lists: list[list[str]], k: int = 60, top_k: int = 20) -> list[str]:
    scores: dict[str, float] = {}
    for docs in rank_lists:
        seen = set()
        for rank, doc_id in enumerate(docs, start=1):
            doc_id = _normalize_doc_id(doc_id)
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:top_k]]


def _precision(pred_ids: list[str], gt_ids: set[str]) -> float:
    return sum(1 for x in pred_ids if x in gt_ids) / len(pred_ids) if pred_ids else 0.0


def _recall(pred_ids: list[str], gt_ids: set[str]) -> float:
    return sum(1 for x in pred_ids if x in gt_ids) / len(gt_ids) if gt_ids else 0.0


def _f1(p: float, r: float) -> float:
    return (2 * p * r / (p + r)) if (p + r) else 0.0


def _mrr(pred_ids: list[str], gt_ids: set[str]) -> float:
    for i, x in enumerate(pred_ids, start=1):
        if x in gt_ids:
            return 1.0 / i
    return 0.0


def _ap(pred_ids: list[str], gt_ids: set[str]) -> float:
    if not gt_ids:
        return 0.0
    hit = 0
    total = 0.0
    for i, x in enumerate(pred_ids, start=1):
        if x in gt_ids:
            hit += 1
            total += hit / i
    return total / len(gt_ids)


def _ndcg(pred_ids: list[str], gt_ids: set[str]) -> float:
    if not gt_ids:
        return 0.0
    dcg = 0.0
    for i, x in enumerate(pred_ids, start=1):
        if x in gt_ids:
            dcg += 1.0 / math.log2(i + 1)
    ideal_len = min(len(gt_ids), len(pred_ids))
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_len + 1))
    return dcg / idcg if idcg else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description='Compute hybrid retrieval metrics from AutoRAG lexical + dense outputs')
    parser.add_argument('--project-dir', default='autorag_workspace/results_retrieval_strategy_fresh')
    parser.add_argument('--qa-path', default='autorag_workspace/qa/qa_fresh.parquet')
    parser.add_argument('--bm25-parquet', default=None)
    parser.add_argument('--dense-parquet', default=None, help='Dense parquet filename under semantic_retrieval')
    parser.add_argument('--dense-module-pattern', default=None, help='If set, only compute hybrid for matching dense module')
    parser.add_argument('--output-csv', default=None)
    parser.add_argument('--rrf-k', type=int, default=60)
    parser.add_argument('--top-k', type=int, default=20)
    args = parser.parse_args()

    trial = _latest_trial(Path(args.project_dir))
    lexical_dir = trial / 'retrieve_node_line' / 'lexical_retrieval'
    semantic_dir = trial / 'retrieve_node_line' / 'semantic_retrieval'
    hybrid_dir = trial / 'retrieve_node_line' / 'hybrid_retrieval'
    hybrid_dir.mkdir(parents=True, exist_ok=True)

    bm25_summary_df = pd.read_csv(lexical_dir / 'summary.csv')
    if bm25_summary_df.empty:
        raise ValueError(f'Empty lexical summary under: {lexical_dir}')
    bm25_filename = args.bm25_parquet or str(bm25_summary_df.iloc[0]['filename'])
    bm25_path = lexical_dir / bm25_filename
    bm25_df = pd.read_parquet(bm25_path)

    semantic_summary_df = pd.read_csv(semantic_dir / 'summary.csv')
    if semantic_summary_df.empty:
        raise ValueError(f'Empty semantic summary under: {semantic_dir}')

    if args.dense_parquet:
        semantic_summary_df = semantic_summary_df[semantic_summary_df['filename'].astype(str) == str(args.dense_parquet)]
    if args.dense_module_pattern and 'module_params' in semantic_summary_df.columns:
        semantic_summary_df = semantic_summary_df[
            semantic_summary_df['module_params'].astype(str).str.contains(args.dense_module_pattern, na=False)
        ]
    if semantic_summary_df.empty:
        raise ValueError('No semantic rows matched the requested dense filter')

    qa_df = pd.read_parquet(args.qa_path)

    summary_rows = []

    for _, semantic_row in semantic_summary_df.iterrows():
        dense_filename = str(semantic_row['filename'])
        dense_module_name = _extract_vectordb_name(semantic_row.get('module_params'))
        dense_path = semantic_dir / dense_filename
        dense_df = pd.read_parquet(dense_path)

        if len(bm25_df) != len(dense_df) or len(bm25_df) != len(qa_df):
            raise ValueError(
                f'Row count mismatch for {dense_module_name}: '
                f'bm25={len(bm25_df)}, dense={len(dense_df)}, qa={len(qa_df)}'
            )

        detail_rows = []
        precisions = []
        recalls = []
        f1s = []
        mrrs = []
        maps = []
        ndcgs = []

        for i in range(len(qa_df)):
            bm25_ids = [_normalize_doc_id(x) for x in _parse_list_cell(bm25_df.iloc[i]['retrieved_ids']) if _normalize_doc_id(x)]
            dense_ids = [_normalize_doc_id(x) for x in _parse_list_cell(dense_df.iloc[i]['retrieved_ids']) if _normalize_doc_id(x)]
            gt_ids = {_normalize_doc_id(x) for x in _parse_list_cell(qa_df.iloc[i].get('retrieval_gt')) if _normalize_doc_id(x)}
            fused = reciprocal_rank_fusion([bm25_ids, dense_ids], k=args.rrf_k, top_k=args.top_k)

            p = _precision(fused, gt_ids)
            r = _recall(fused, gt_ids)
            f1 = _f1(p, r)
            mrr = _mrr(fused, gt_ids)
            ap = _ap(fused, gt_ids)
            ndcg = _ndcg(fused, gt_ids)

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
            mrrs.append(mrr)
            maps.append(ap)
            ndcgs.append(ndcg)

            detail_rows.append({
                'retrieved_ids': fused,
                'retrieval_f1': f1,
                'retrieval_recall': r,
                'retrieval_precision': p,
                'retrieval_ndcg': ndcg,
                'retrieval_map': ap,
                'retrieval_mrr': mrr,
            })

        out_parquet = hybrid_dir / f'{dense_module_name}.parquet'
        pd.DataFrame(detail_rows).to_parquet(out_parquet, index=False)
        summary_rows.append({
            'filename': out_parquet.name,
            'module_name': 'HybridRRF',
            'module_params': str({'rrf_k': args.rrf_k, 'top_k': args.top_k, 'sources': ['bm25', dense_module_name]}),
            'execution_time': None,
            'retrieval_f1': sum(f1s) / len(f1s),
            'retrieval_recall': sum(recalls) / len(recalls),
            'retrieval_precision': sum(precisions) / len(precisions),
            'retrieval_ndcg': sum(ndcgs) / len(ndcgs),
            'retrieval_map': sum(maps) / len(maps),
            'retrieval_mrr': sum(mrrs) / len(mrrs),
            'is_best': False,
        })

    out_csv = Path(args.output_csv) if args.output_csv else hybrid_dir / 'summary.csv'
    summary_df = pd.DataFrame(summary_rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f'Saved summary: {out_csv}')
    print(summary_df.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
