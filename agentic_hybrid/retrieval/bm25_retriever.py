from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List

from agentic_hybrid.retrieval.vector_retriever import load_pickled_documents


TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text or "")]


class BM25Retriever:
    def __init__(self, docs_pickle_path: str | Path, k1: float = 1.5, b: float = 0.75) -> None:
        self.docs_pickle_path = Path(docs_pickle_path)
        self.k1 = k1
        self.b = b
        self._docs: List[Dict[str, Any]] | None = None
        self._doc_tokens: List[List[str]] | None = None
        self._doc_lens: List[int] | None = None
        self._idf: Dict[str, float] | None = None
        self._avgdl = 0.0

    def _ensure_loaded(self) -> None:
        if self._docs is not None:
            return
        self._docs = load_pickled_documents(self.docs_pickle_path)
        self._doc_tokens = []
        self._doc_lens = []
        df: Dict[str, int] = {}
        for doc in self._docs:
            tokens = _tokenize(f"{doc.get('title','')} {doc.get('text','')}")
            self._doc_tokens.append(tokens)
            self._doc_lens.append(len(tokens))
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        n_docs = max(len(self._docs), 1)
        self._avgdl = sum(self._doc_lens) / n_docs
        self._idf = {t: math.log(1 + (n_docs - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        self._ensure_loaded()
        assert self._docs is not None and self._doc_tokens is not None and self._doc_lens is not None and self._idf is not None
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []

        scored: List[tuple[int, float]] = []
        for idx, tokens in enumerate(self._doc_tokens):
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            dl = self._doc_lens[idx]
            score = 0.0
            for term in q_tokens:
                f = tf.get(term, 0)
                if not f:
                    continue
                idf = self._idf.get(term, 0.0)
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self._avgdl, 1e-9))
                score += idf * ((f * (self.k1 + 1)) / max(denom, 1e-9))
            if score > 0:
                scored.append((idx, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(scored[:top_k], start=1):
            item = dict(self._docs[idx])
            item["score"] = float(score)
            item["rank"] = rank
            item["retrieval_source"] = "bm25"
            out.append(item)
        return out

