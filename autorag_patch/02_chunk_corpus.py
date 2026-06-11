from __future__ import annotations

import shutil
import time
import math
from pathlib import Path

import pandas as pd
from autorag.chunker import Chunker


EXCLUDED_PARTS = (
    '/parsed/',
    '/qa/',
    '/results/',
    '/results_',
    '/result/',
    '/result_',
    '/experiment_results/',
    '/benchmark_results/',
)
VALID_SUFFIXES = {'.parquet', '.csv'}
PREFERRED_CORPUS_NAMES = (
    'corpus.parquet',
    'corpus.csv',
    '0.parquet',
    '0.csv',
)


def _is_excluded_path(path: Path) -> bool:
    path_str = str(path).replace('\\', '/').lower()
    return any(part in path_str for part in EXCLUDED_PARTS)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == '.csv':
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _looks_like_corpus_table(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() not in VALID_SUFFIXES:
        return False
    if 'prediction' in name or name == 'qa.parquet' or name == 'qa.csv' or name.endswith('.bak'):
        return False

    try:
        df = _read_table(path)
    except Exception:
        return False

    cols = set(df.columns.astype(str).tolist())
    if {'question', 'answer', 'mode'} & cols:
        return False

    text_cols = {'contents', 'text', 'texts'}
    id_cols = {'doc_id', 'id', 'chunk_id', 'document_id', 'source_doc_id'}
    return bool(cols & text_cols) and (bool(cols & id_cols) or 'metadata' in cols)


def _normalize_to_standard_corpus_path(corpus_root: Path, candidate: Path) -> Path:
    corpus_root.mkdir(parents=True, exist_ok=True)
    normalized = corpus_root / 'corpus.parquet'
    if candidate.suffix.lower() == '.csv':
        df = pd.read_csv(candidate)
        df.to_parquet(normalized, index=False)
        print(f'?? normalized corpus csv -> parquet: {candidate} -> {normalized}')
        return normalized
    if candidate.resolve() != normalized.resolve():
        shutil.copyfile(candidate, normalized)
        print(f'?? normalized corpus parquet: {candidate} -> {normalized}')
    return normalized


def _collect_candidates(root: Path, started_at: float, strict: bool) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in VALID_SUFFIXES or _is_excluded_path(p):
            continue
        try:
            if p.stat().st_mtime + 1 < started_at:
                continue
        except OSError:
            continue
        if strict:
            if _looks_like_corpus_table(p):
                out.append(p)
        else:
            out.append(p)
    return out


def _collect_recent_even_if_excluded(root: Path, started_at: float, strict: bool) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob('*'):
        if not p.is_file() or p.suffix.lower() not in VALID_SUFFIXES:
            continue
        try:
            if p.stat().st_mtime + 1 < started_at:
                continue
        except OSError:
            continue
        if strict and not _looks_like_corpus_table(p):
            continue
        out.append(p)
    return out


def _collect_preferred_local_candidates(corpus_root: Path) -> list[Path]:
    out: list[Path] = []
    for name in PREFERRED_CORPUS_NAMES:
        p = corpus_root / name
        if p.is_file() and p.suffix.lower() in VALID_SUFFIXES:
            out.append(p)
    if out:
        return out

    for p in corpus_root.iterdir():
        if not p.is_file() or p.suffix.lower() not in VALID_SUFFIXES:
            continue
        if p.name.lower() in {'summary.csv', 'qa.csv', 'qa.parquet'} or p.name.endswith('.bak'):
            continue
        out.append(p)
    return out


def _describe_candidate(path: Path) -> str:
    try:
        df = _read_table(path)
        cols = ', '.join(df.columns.astype(str).tolist()[:10])
        return f'{path} [cols: {cols}]'
    except Exception:
        return str(path)


def _coerce_metadata(value) -> dict:
    if isinstance(value, dict):
        return dict(value)
    if pd.isna(value):
        return {}
    return {}


def _split_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    step = max(1, chunk_size - overlap)
    out: list[str] = []
    for start in range(0, len(text), step):
        chunk = text[start:start + chunk_size].strip()
        if chunk:
            out.append(chunk)
        if start + chunk_size >= len(text):
            break
    return out


def _build_fallback_corpus_from_parsed(parsed_data_path: str, corpus_root: Path) -> Path:
    parsed_path = Path(parsed_data_path)
    df = pd.read_parquet(parsed_path)
    rows: list[dict] = []

    for _, row in df.iterrows():
        source_doc_id = str(row.get('doc_id', '')).strip()
        contents = str(row.get('contents', '') or '')
        data_type = row.get('data_type')
        metadata = _coerce_metadata(row.get('metadata'))
        if source_doc_id:
            metadata.setdefault('doc_id', source_doc_id)
            metadata.setdefault('source_doc_id', source_doc_id)
        if pd.notna(data_type):
            metadata.setdefault('data_type', str(data_type))

        chunks = _split_text(contents)
        if not chunks and contents.strip():
            chunks = [contents.strip()]

        for idx, chunk_text in enumerate(chunks):
            chunk_doc_id = f'{source_doc_id}__chunk_{idx}' if source_doc_id else f'chunk_{len(rows)}'
            rows.append(
                {
                    'doc_id': chunk_doc_id,
                    'source_doc_id': source_doc_id or chunk_doc_id,
                    'contents': chunk_text,
                    'metadata': metadata,
                    'data_type': data_type,
                }
            )

    fallback_df = pd.DataFrame(rows)
    corpus_root.mkdir(parents=True, exist_ok=True)
    normalized = corpus_root / 'corpus.parquet'
    fallback_df.to_parquet(normalized, index=False)
    print(f'?? fallback corpus built from parsed: {parsed_path} -> {normalized} ({len(fallback_df)} chunks)')
    return normalized


def chunk_documents(
    parsed_data_path: str = 'autorag_workspace/parsed/parsed.parquet',
    chunk_config_path: str = 'autorag_workspace/configs/chunk.yaml',
    corpus_dir: str = 'autorag_workspace/corpus',
) -> Path:
    print('?? chunking documents...')
    started_at = time.time()

    chunker = Chunker.from_parquet(parsed_data_path=parsed_data_path)
    chunker.start_chunking(chunk_config_path)

    corpus_root = Path(corpus_dir)
    candidates = _collect_preferred_local_candidates(corpus_root)
    if not candidates:
        candidates = _collect_candidates(corpus_root, started_at, strict=True)
    if not candidates:
        candidates = _collect_candidates(Path('autorag_workspace'), started_at, strict=True)
    if not candidates:
        candidates = _collect_preferred_local_candidates(corpus_root)
    if not candidates:
        candidates = _collect_candidates(corpus_root, started_at, strict=False)
    if not candidates:
        candidates = _collect_candidates(Path('autorag_workspace'), started_at, strict=False)
    if not candidates:
        # Some AutoRAG versions write fresh chunk outputs under benchmark/results-style folders.
        # We allow those only if they were created in this run.
        candidates = _collect_recent_even_if_excluded(Path('autorag_workspace'), started_at, strict=True)
    if not candidates:
        candidates = _collect_recent_even_if_excluded(Path('autorag_workspace'), started_at, strict=False)
    if not candidates:
        candidates = [
            p
            for p in corpus_root.rglob('*')
            if p.is_file()
            and p.suffix.lower() in VALID_SUFFIXES
            and not _is_excluded_path(p)
            and p.name.lower() not in {'summary.csv', 'qa.csv', 'qa.parquet'}
            and not p.name.endswith('.bak')
        ]

    if not candidates:
        print('?? no corpus table found after chunking.')
        return _build_fallback_corpus_from_parsed(parsed_data_path, corpus_root)

    print('? chunking done')
    print('?? candidate corpus files:')
    for p in candidates[:10]:
        print('  - ' + _describe_candidate(p))

    preferred = [p for p in candidates if 'corpus' in p.name.lower()]
    latest = max(preferred or candidates, key=lambda p: p.stat().st_mtime)
    normalized = _normalize_to_standard_corpus_path(corpus_root, latest)
    print(f'?? selected corpus file: {normalized}')
    return normalized


if __name__ == '__main__':
    chunk_documents()
