from __future__ import annotations

from typing import Any, Dict, List


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, docs: List[Dict[str, Any]], top_n: int) -> List[Dict[str, Any]]:
        if not docs:
            return []
        model = self._ensure_model()
        pairs = [(query, f"{d.get('title','')}\n{d.get('text','')}") for d in docs]
        scores = model.predict(pairs)
        rescored: List[Dict[str, Any]] = []
        for doc, score in zip(docs, scores):
            item = dict(doc)
            item["rerank_score"] = float(score)
            rescored.append(item)
        rescored.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)
        return rescored[:top_n]

