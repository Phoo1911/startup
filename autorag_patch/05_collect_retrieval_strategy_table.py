from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import pandas as pd


def _latest_trial(project_dir: Path) -> Path:
    trials = [p for p in project_dir.iterdir() if p.is_dir() and p.name.isdigit()]
    if not trials:
        raise FileNotFoundError(f"No numeric trial dirs under: {project_dir}")
    return max(trials, key=lambda p: int(p.name))


def _extract_vectordb_name(module_params: object) -> str:
    text = str(module_params or "")
    match = re.search(r"vectordb['\"]?\s*:\s*['\"]([^'\"]+)['\"]", text)
    if match:
        return match.group(1)
    match = re.search(r"vectordb_[-a-zA-Z0-9_]+", text)
    if match:
        return match.group(0)
    return "unknown_vectordb"


def _backbone_label(vectordb_name: str) -> str:
    mapping = {
        "vectordb_kosroberta": "KoSRoBERTa",
        "vectordb_kosimcse": "KoSimCSE",
        "vectordb_mpnet": "MPNet",
        "vectordb_minilm": "MiniLM",
        "vectordb_e5": "E5",
        "vectordb_bgem3": "BGE-M3",
    }
    return mapping.get(vectordb_name, vectordb_name)


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect lexical/semantic/hybrid retrieval results by backbone")
    parser.add_argument('--autorag-project-dir', default='autorag_workspace/results_backbone_fresh')
    parser.add_argument('--output-csv', default='autorag_workspace/results_agent/retrieval_strategy_table_all_backbones.csv')
    args = parser.parse_args()

    trial = _latest_trial(Path(args.autorag_project_dir))
    lexical_csv = trial / 'retrieve_node_line' / 'lexical_retrieval' / 'summary.csv'
    semantic_csv = trial / 'retrieve_node_line' / 'semantic_retrieval' / 'summary.csv'
    hybrid_csv = trial / 'retrieve_node_line' / 'hybrid_retrieval' / 'summary.csv'

    lexical_df = pd.read_csv(lexical_csv)
    semantic_df = pd.read_csv(semantic_csv)
    hybrid_df = pd.read_csv(hybrid_csv)

    if lexical_df.empty:
        raise ValueError(f'Empty lexical summary: {lexical_csv}')

    bm25_row = lexical_df.iloc[0].to_dict()

    rows = []
    for _, sem_row in semantic_df.iterrows():
        vectordb_name = _extract_vectordb_name(sem_row.get('module_params'))
        backbone = _backbone_label(vectordb_name)
        hybrid_match = hybrid_df[hybrid_df['module_params'].astype(str).str.contains(vectordb_name, na=False)]
        hybrid_row = hybrid_match.iloc[0].to_dict() if not hybrid_match.empty else None

        rows.append({
            'backbone': backbone,
            'strategy': 'Lexical (BM25)',
            'recall': bm25_row.get('retrieval_recall'),
            'mrr': bm25_row.get('retrieval_mrr'),
            'precision': bm25_row.get('retrieval_precision'),
            'f1': bm25_row.get('retrieval_f1'),
            'ndcg': bm25_row.get('retrieval_ndcg'),
            'map': bm25_row.get('retrieval_map'),
        })
        rows.append({
            'backbone': backbone,
            'strategy': 'Semantic',
            'recall': sem_row.get('retrieval_recall'),
            'mrr': sem_row.get('retrieval_mrr'),
            'precision': sem_row.get('retrieval_precision'),
            'f1': sem_row.get('retrieval_f1'),
            'ndcg': sem_row.get('retrieval_ndcg'),
            'map': sem_row.get('retrieval_map'),
        })
        if hybrid_row is not None:
            rows.append({
                'backbone': backbone,
                'strategy': 'Hybrid (RRF)',
                'recall': hybrid_row.get('retrieval_recall'),
                'mrr': hybrid_row.get('retrieval_mrr'),
                'precision': hybrid_row.get('retrieval_precision'),
                'f1': hybrid_row.get('retrieval_f1'),
                'ndcg': hybrid_row.get('retrieval_ndcg'),
                'map': hybrid_row.get('retrieval_map'),
            })

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f'Saved: {out_path}')
    print(out_df.to_string(index=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
