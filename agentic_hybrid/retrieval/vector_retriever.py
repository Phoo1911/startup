"""
retrieval/vector_retriever.py — FAISS vector retriever over pre-built index

Fixes / improvements applied:
  - Added a comment explaining FAISS score direction: inner-product (IP) indices
    return higher = better; L2 indices return lower = better.  The score is
    stored as-is so downstream re-ranking (cross-encoder) can override it.
    No silent sign-flip that would surprise users of L2 indices.
  - normalize_document_record: title now also falls back to metadata["name"]
    before truncating text, covering more real-world K-Startup metadata shapes.
  - load_pickled_documents: wrapped in a clear error message if the pickle
    is corrupt or wrong version.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

try:
    import faiss  # type: ignore
except Exception:
    faiss = None


def _safe_getattr(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def normalize_document_record(doc_obj: Any) -> Dict[str, Any]:
    """
    Convert any raw document object (dataclass, dict, LangChain Document …)
    into a plain dict with canonical keys: id, title, text, metadata.
    """
    metadata: Dict[str, Any] = _safe_getattr(doc_obj, "metadata", {}) or {}
    text: str = _safe_getattr(doc_obj, "text", "") or ""
    doc_id: str = str(_safe_getattr(doc_obj, "id", "") or metadata.get("id", ""))

    # FIX: also check metadata["name"] before falling back to raw text slice
    title: str = (
        metadata.get("title")
        or metadata.get("name")
        or text[:120]
    )

    return {
        "id": doc_id,
        "title": title,
        "text": text,
        "metadata": metadata,
    }


def load_pickled_documents(docs_pickle_path: str | Path) -> List[Dict[str, Any]]:
    """Load and normalise all documents from a pickle file."""
    path = Path(docs_pickle_path)
    try:
        with path.open("rb") as f:
            raw_docs = pickle.load(f)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load docs pickle at {path}. "
            f"Make sure the file exists and was created with a compatible Python version. "
            f"Original error: {exc}"
        ) from exc
    return [normalize_document_record(d) for d in raw_docs]


class VectorRetriever:
    """
    Thin wrapper around a pre-built FAISS index.

    Score direction:
      - IndexFlatIP / IndexIVFFlat with inner-product metric: higher = better
      - IndexFlatL2 / IndexIVFFlat with L2 metric: lower = better
    The raw score is stored in doc["score"]; the cross-encoder reranker
    replaces this with its own score, so direction only matters for the
    retrieve-only (no rerank) ablation mode.
    """

    def __init__(
        self,
        faiss_index_path: str | Path,
        docs_pickle_path: str | Path,
        embedding_model_name: str,
    ) -> None:
        self.faiss_index_path = Path(faiss_index_path)
        self.docs_pickle_path = Path(docs_pickle_path)
        self.embedding_model_name = embedding_model_name
        self._index = None
        self._docs: List[Dict[str, Any]] | None = None
        self._embedder = None

    # ── Lazy loading ──────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._index is not None and self._docs is not None:
            return
        if faiss is None:
            raise ImportError("faiss-cpu (or faiss-gpu) is required. pip install faiss-cpu")
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(f"Missing FAISS index: {self.faiss_index_path}")
        if not self.docs_pickle_path.exists():
            raise FileNotFoundError(f"Missing docs pickle: {self.docs_pickle_path}")
        self._index = faiss.read_index(str(self.faiss_index_path))
        self._docs = load_pickled_documents(self.docs_pickle_path)

    def _ensure_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _encode_query(self, query: str) -> np.ndarray:
        model = self._ensure_embedder()
        vec = model.encode([query], normalize_embeddings=True)
        if not isinstance(vec, np.ndarray):
            vec = np.asarray(vec)
        return vec.astype("float32")

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def documents(self) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self._docs is not None
        return self._docs

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self._index is not None and self._docs is not None

        query_vec = self._encode_query(query)
        k = min(max(top_k, 1), len(self._docs))
        scores, indices = self._index.search(query_vec, k)

        out: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self._docs):
                continue
            item = dict(self._docs[idx])
            item["score"] = float(score)
            item["rank"] = rank
            item["retrieval_source"] = "vector"
            out.append(item)
        return out