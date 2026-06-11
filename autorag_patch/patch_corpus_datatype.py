from __future__ import annotations

import argparse
import ast
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

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
    "slp_center": "centers",
    "center": "centers",
    "centers": "centers",
    "cert_product": "products",
    "product": "products",
    "products": "products",
    "cert_corporate": "corporates",
    "corporate": "corporates",
    "corporates": "corporates",
    "institution": "institutions",
    "institutions": "institutions",
}
POLICY_TYPES = {"announcements", "business"}


def _glob_latest_parquet(corpus_dir: Path) -> Path:
    candidates = [p for p in corpus_dir.rglob("*.parquet") if p.is_file() and not p.name.endswith('.bak')]
    if not candidates:
        raise FileNotFoundError(f"No parquet found under: {corpus_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _canonicalize_dtype(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip().lower()
    if not text or text == 'nan':
        return None
    return CANONICAL_TYPES.get(text, text)


def _as_dict(x: Any) -> Optional[Dict[str, Any]]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    if isinstance(x, dict):
        return dict(x)
    if isinstance(x, str):
        s = x.strip()
        if len(s) < 2:
            return None
        if (s.startswith('{') and s.endswith('}')) or (s.startswith('[') and s.endswith(']')):
            try:
                obj = ast.literal_eval(s)
                return obj if isinstance(obj, dict) else None
            except Exception:
                return None
    return None


def _get_meta_value(meta: Any, key: str) -> Optional[Any]:
    d = _as_dict(meta)
    if not d:
        return None
    return d.get(key)


def _normalize_title_key(title: Any) -> str:
    return ' '.join(str(title or '').split()).strip().lower()


def _docid_candidates(doc_id: Any) -> List[str]:
    if doc_id is None or (isinstance(doc_id, float) and pd.isna(doc_id)):
        return []
    s = str(doc_id).strip()
    if not s:
        return []
    candidates: List[str] = [s]

    for sep in ('::', '#', '|', '/chunk/', '/doc/'):
        if sep in s:
            candidates.append(s.split(sep)[0].strip())

    tmp = s
    for _ in range(6):
        if '_' not in tmp:
            break
        head, tail = tmp.rsplit('_', 1)
        if tail.isdigit():
            tmp = head.strip()
            candidates.append(tmp)
        else:
            break

    uniq: List[str] = []
    seen = set()
    for cand in candidates:
        if cand and cand not in seen:
            seen.add(cand)
            uniq.append(cand)
    return uniq


def _make_parsed_indexes(parsed_df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    if 'doc_id' in parsed_df.columns:
        doc_ids = parsed_df['doc_id'].astype(str)
    elif 'id' in parsed_df.columns:
        doc_ids = parsed_df['id'].astype(str)
    else:
        raise RuntimeError('parsed.parquet must contain doc_id or id')

    if 'data_type' in parsed_df.columns:
        dtypes = parsed_df['data_type'].map(_canonicalize_dtype)
    elif 'metadata' in parsed_df.columns:
        dtypes = parsed_df['metadata'].apply(lambda m: _canonicalize_dtype(_get_meta_value(m, 'data_type')))
    else:
        raise RuntimeError('parsed.parquet must contain data_type or metadata')

    if 'metadata' in parsed_df.columns:
        titles = parsed_df['metadata'].apply(lambda m: _get_meta_value(m, 'title'))
    else:
        titles = pd.Series([None] * len(parsed_df))

    docid_to_dtype: Dict[str, str] = {}
    title_to_dtype: Dict[str, str] = {}
    title_to_docid: Dict[str, str] = {}

    for doc_id, dtype, title in zip(doc_ids.tolist(), dtypes.tolist(), titles.tolist()):
        if doc_id and dtype:
            docid_to_dtype[str(doc_id)] = str(dtype)
        title_key = _normalize_title_key(title)
        if title_key and dtype and doc_id:
            title_to_dtype.setdefault(title_key, str(dtype))
            title_to_docid.setdefault(title_key, str(doc_id))

    if not docid_to_dtype:
        raise RuntimeError('failed to build parsed doc_id -> data_type mapping')
    return docid_to_dtype, title_to_dtype, title_to_docid


def _pick_existing_source_id(row: pd.Series) -> Optional[str]:
    for key in ('source_doc_id', 'doc_id', 'id', 'chunk_id', 'document_id'):
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            text = str(value).strip()
            if text:
                return text
    meta = _as_dict(row.get('metadata'))
    if meta:
        for key in ('source_doc_id', 'doc_id', 'document_id', 'id', 'source_id'):
            value = meta.get(key)
            if value is not None:
                text = str(value).strip()
                if text:
                    return text
    return None


def _pick_title(row: pd.Series) -> str:
    for key in ('title', 'name'):
        value = row.get(key)
        if value is not None and not (isinstance(value, float) and pd.isna(value)):
            text = str(value).strip()
            if text:
                return text
    meta = _as_dict(row.get('metadata'))
    if meta:
        title = str(meta.get('title') or '').strip()
        if title:
            return title
    return ''


def _enrich_metadata(meta: Any, source_doc_id: Optional[str], data_type: Optional[str]) -> Dict[str, Any]:
    out = _as_dict(meta) or {}
    if source_doc_id:
        out['source_doc_id'] = source_doc_id
        out.setdefault('doc_id', source_doc_id)
    if data_type:
        out['data_type'] = data_type
    return out


def _infer_data_type_for_corpus(corpus_df: pd.DataFrame, parsed_indexes: Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]) -> pd.DataFrame:
    docid_to_dtype, title_to_dtype, title_to_docid = parsed_indexes
    out = corpus_df.copy()

    existing_dtype = out['data_type'].map(_canonicalize_dtype) if 'data_type' in out.columns else pd.Series([None] * len(out), index=out.index)
    source_doc_id = out.apply(_pick_existing_source_id, axis=1)
    titles = out.apply(_pick_title, axis=1).map(_normalize_title_key)

    if 'metadata' in out.columns:
        meta_dtype = out['metadata'].apply(lambda m: _canonicalize_dtype(_get_meta_value(m, 'data_type')))
        existing_dtype = existing_dtype.where(existing_dtype.notna(), meta_dtype)

    mapped_dtype = source_doc_id.map(lambda x: docid_to_dtype.get(str(x).strip()) if x else None)
    existing_dtype = existing_dtype.where(existing_dtype.notna(), mapped_dtype)

    inferred_source = source_doc_id.copy()
    missing_source = inferred_source.isna() | (inferred_source.astype(str).str.strip() == '')
    inferred_source = inferred_source.where(~missing_source, titles.map(title_to_docid))

    def infer_from_candidates(value: Any) -> Optional[str]:
        for cand in _docid_candidates(value):
            if cand in docid_to_dtype:
                return cand
        return None

    candidate_cols = [c for c in ('source_doc_id', 'doc_id', 'id', 'chunk_id', 'document_id') if c in out.columns]
    if candidate_cols:
        candidate_series = out[candidate_cols[0]].copy()
        for col in candidate_cols[1:]:
            candidate_series = candidate_series.where(candidate_series.notna(), out[col])
        inferred_from_ids = candidate_series.apply(infer_from_candidates)
        inferred_source = inferred_source.where(inferred_source.notna(), inferred_from_ids)
        existing_dtype = existing_dtype.where(existing_dtype.notna(), inferred_from_ids.map(docid_to_dtype))

    existing_dtype = existing_dtype.where(existing_dtype.notna(), titles.map(title_to_dtype))
    inferred_source = inferred_source.where(inferred_source.notna(), titles.map(title_to_docid))

    out['source_doc_id'] = inferred_source
    out['data_type'] = existing_dtype.map(_canonicalize_dtype)
    out['metadata'] = [
        _enrich_metadata(meta, src if pd.notna(src) else None, dt if pd.notna(dt) else None)
        for meta, src, dt in zip(out.get('metadata', pd.Series([None] * len(out))), out['source_doc_id'], out['data_type'])
    ]
    return out


def patch_corpus_datatype(
    parsed_path: Path,
    corpus_path: Path,
    out_path: Optional[Path] = None,
    inplace: bool = True,
) -> Path:
    parsed_df = pd.read_parquet(parsed_path)
    parsed_indexes = _make_parsed_indexes(parsed_df)

    corpus_df = pd.read_parquet(corpus_path)
    patched = _infer_data_type_for_corpus(corpus_df, parsed_indexes)

    if 'data_type' not in patched.columns or patched['data_type'].isna().all():
        raise RuntimeError(
            'failed to recover data_type for corpus rows. '            'chunk output likely lost source ids and titles needed for mapping.'
        )

    match_rate = float(patched['data_type'].notna().mean())
    print(f'? data_type recovery match rate: {match_rate:.1%}')
    print('?? corpus data_type distribution:')
    print(patched['data_type'].value_counts(dropna=False).head(20).to_string())

    missing_policy = [dtype for dtype in POLICY_TYPES if dtype not in set(patched['data_type'].dropna().astype(str))]
    if missing_policy:
        print(f'?? policy-like types still missing after patch: {missing_policy}')

    if out_path is None:
        if inplace:
            backup = corpus_path.with_suffix(corpus_path.suffix + '.bak')
            if not backup.exists():
                shutil.copyfile(corpus_path, backup)
                print(f'?? backup created: {backup}')
            out_path = corpus_path
        else:
            out_path = corpus_path.parent / (corpus_path.stem + '_with_datatype.parquet')

    patched.to_parquet(out_path, index=False)
    print(f'?? saved: {out_path}')
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description='Patch AutoRAG corpus parquet to include source_doc_id and data_type')
    parser.add_argument('--parsed-path', default='autorag_workspace/parsed/parsed.parquet')
    parser.add_argument('--corpus-path', default=None)
    parser.add_argument('--out-path', default=None)
    parser.add_argument('--no-inplace', action='store_true')
    args = parser.parse_args()

    parsed_path = Path(args.parsed_path)
    corpus_path = Path(args.corpus_path) if args.corpus_path else _glob_latest_parquet(Path('autorag_workspace/corpus'))
    if args.corpus_path is None:
        print(f'?? auto-selected corpus parquet: {corpus_path}')

    patch_corpus_datatype(
        parsed_path=parsed_path,
        corpus_path=corpus_path,
        out_path=Path(args.out_path) if args.out_path else None,
        inplace=not args.no_inplace,
    )


if __name__ == '__main__':
    main()
