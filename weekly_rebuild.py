from __future__ import annotations

import argparse
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from agents.data_collector import DataCollectionAgent
from agents.rag_builder import RAGBuilderAgent
from config.settings import Config
from core.rag_system import RAGSystem


def _count_items(raw_data: Dict[str, List[Dict]]) -> Dict[str, int]:
    return {key: len(value or []) for key, value in raw_data.items()}


def run_weekly_rebuild(
    *,
    service_key: str,
    cache_dir: Path,
    max_pages: int,
    days_range: int,
    embedding_model: str,
    provider: str,
) -> Dict[str, object]:
    started_at = datetime.now()
    cache_dir.mkdir(parents=True, exist_ok=True)

    raw_cache_path = cache_dir / "raw_data.pkl"
    index_prefix = cache_dir / "rag_index"
    summary_path = cache_dir / "weekly_rebuild.last.json"

    print("=" * 80)
    print("Weekly Rebuild Start")
    print("=" * 80)
    print(f"cache_dir={cache_dir}")
    print(f"embedding_model={embedding_model}")
    print(f"provider={provider}")
    print(f"max_pages={max_pages}, days_range={days_range}")

    collector = DataCollectionAgent(service_key=service_key, llm_client=None)
    rag = RAGSystem(embedding_model=embedding_model, provider=provider)
    builder = RAGBuilderAgent(rag, llm_client=None)

    raw_data = collector.collect_all(max_pages=max_pages, days_range=days_range)
    counts = _count_items(raw_data)
    total_items = sum(counts.values())

    with raw_cache_path.open("wb") as f:
        pickle.dump(raw_data, f)
    print(f"[saved] raw data -> {raw_cache_path}")

    built_docs = builder.build_index(raw_data)
    rag.save(str(index_prefix))
    print(f"[saved] rag index -> {index_prefix}.faiss / {index_prefix}.docs.pkl")

    finished_at = datetime.now()
    summary: Dict[str, object] = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "duration_seconds": round((finished_at - started_at).total_seconds(), 2),
        "cache_dir": str(cache_dir),
        "embedding_model": embedding_model,
        "provider": provider,
        "max_pages": max_pages,
        "days_range": days_range,
        "total_raw_items": total_items,
        "built_documents": built_docs,
        "counts_by_type": counts,
        "raw_cache_path": str(raw_cache_path),
        "index_prefix": str(index_prefix),
    }

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] summary -> {summary_path}")
    print("=" * 80)
    print("Weekly Rebuild Complete")
    print("=" * 80)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Weekly full rebuild for KISED corpus and RAG index")
    parser.add_argument("--cache-dir", default=str(Config.CACHE_DIR), help="Directory to write raw_data.pkl and rag_index.*")
    parser.add_argument("--max-pages", type=int, default=Config.MAX_PAGES_PER_ENDPOINT, help="Pages to collect per endpoint")
    parser.add_argument("--days-range", type=int, default=Config.DATA_DAYS_RANGE, help="Recency filter in days for supported endpoints")
    parser.add_argument("--embedding-model", default=Config.EMBEDDING_MODEL, help="Embedding model for index build")
    parser.add_argument("--provider", default=Config.EMBEDDING_PROVIDER, choices=["faiss", "simple", "chroma"], help="RAG backend provider")
    parser.add_argument("--service-key", default=Config.SERVICE_KEY, help="Override KISED service key")
    args = parser.parse_args()

    service_key = str(args.service_key or "").strip()
    if not service_key:
        raise SystemExit("KISED service key is required. Set KISED_SERVICE_KEY or pass --service-key.")

    run_weekly_rebuild(
        service_key=service_key,
        cache_dir=Path(args.cache_dir),
        max_pages=int(args.max_pages),
        days_range=int(args.days_range),
        embedding_model=str(args.embedding_model),
        provider=str(args.provider),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
