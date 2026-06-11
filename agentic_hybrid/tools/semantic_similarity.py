from __future__ import annotations

from typing import Dict, Optional

import numpy as np

_EMBEDDER = None
_EMBEDDER_NAME: Optional[str] = None
_EMBED_CACHE: Dict[str, np.ndarray] = {}


def _get_embedder(model_name: str):
    global _EMBEDDER, _EMBEDDER_NAME
    if _EMBEDDER is None or _EMBEDDER_NAME != model_name:
        from sentence_transformers import SentenceTransformer

        _EMBEDDER = SentenceTransformer(model_name)
        _EMBEDDER_NAME = model_name
        _EMBED_CACHE.clear()
    return _EMBEDDER


def encode_text(text: str, model_name: str) -> np.ndarray:
    normalized = str(text or "").strip()
    if normalized in _EMBED_CACHE:
        return _EMBED_CACHE[normalized]

    model = _get_embedder(model_name)
    vec = model.encode([normalized], normalize_embeddings=True)
    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec)
    out = vec[0].astype("float32")
    _EMBED_CACHE[normalized] = out
    return out


def cosine_similarity(text_a: str, text_b: str, model_name: str) -> float:
    vec_a = encode_text(text_a, model_name)
    vec_b = encode_text(text_b, model_name)
    return float(np.dot(vec_a, vec_b))
