# core/rag_system.py
"""
RAG System with Multiple Backends (retrieve 메서드 추가)
"""
from __future__ import annotations
from typing import List, Dict, Any, Optional, Protocol, Callable, Tuple
from pathlib import Path
import numpy as np
import pickle

from models.data import Document
from legacy_core.embedder import get_embedding_function, chunk_text


# ─────────────────────────────────────────────
# Backend Interface
# ─────────────────────────────────────────────
class RAGBackend(Protocol):
    def add_documents(self, docs: List[Document]) -> None: ...
    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]: ...
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


EmbedFn = Callable[[List[str]], np.ndarray]


# ─────────────────────────────────────────────
# Simple Backend
# ─────────────────────────────────────────────
class SimpleBackend:
    def __init__(self, embed_fn: EmbedFn):
        self.embed_fn = embed_fn
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[Document]) -> None:
        texts = [d.text for d in docs]
        vecs = self.embed_fn(texts)

        if self.embeddings is None:
            self.embeddings = vecs
            self.documents = list(docs)
        else:
            self.embeddings = np.vstack([self.embeddings, vecs])
            self.documents.extend(docs)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        q_vec = self.embed_fn([query])[0:1]
        doc_vecs = self.embeddings

        q_norm = q_vec / (np.linalg.norm(q_vec, axis=1, keepdims=True) + 1e-8)
        d_norm = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-8)
        scores_all = (q_norm @ d_norm.T)[0]

        # 필터 적용 ($in 연산자 지원)
        idxs = np.arange(len(self.documents))
        if filters:
            keep = []
            for i, doc in enumerate(self.documents):
                meta = getattr(doc, "metadata", {}) or {}
                ok = True
                for k, v in filters.items():
                    if isinstance(v, dict):
                        if "$in" in v:
                            if meta.get(k) not in v["$in"]:
                                ok = False
                                break
                    else:
                        if meta.get(k) != v:
                            ok = False
                            break
                if ok:
                    keep.append(i)
            if not keep:
                return []
            idxs = np.array(keep, dtype=int)
            scores = scores_all[idxs]
        else:
            scores = scores_all

        sorted_idx = np.argsort(scores)[::-1][:top_k]
        sel_doc_idxs = idxs[sorted_idx]
        sel_scores = scores[sorted_idx]

        results = []
        for doc_idx, score in zip(sel_doc_idxs, sel_scores):
            doc = self.documents[doc_idx]
            meta = getattr(doc, "metadata", {}) or {}
            results.append({
                "document": doc,
                "score": float(score),
                "metadata": meta,
            })
        return results

    def save(self, path: str) -> None:
        data = {
            "documents": self.documents,
            "embeddings": self.embeddings,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path: str) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.documents = data["documents"]
        self.embeddings = data["embeddings"]
        print(f"✅ 로드 완료: {len(self.documents)}개 문서")


# ─────────────────────────────────────────────
# FAISS Backend
# ─────────────────────────────────────────────
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class FaissBackend:
    def __init__(self, embed_fn: EmbedFn, dim: int):
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS 사용을 위해 패키지를 설치하세요: pip install faiss-cpu")
        self.embed_fn = embed_fn
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.documents: List[Document] = []

    def add_documents(self, docs: List[Document]) -> None:
        texts = [d.text for d in docs]
        vecs = self.embed_fn(texts).astype("float32")
        faiss.normalize_L2(vecs)
        self.index.add(vecs)
        self.documents.extend(docs)

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []

        q_vec = self.embed_fn([query]).astype("float32")
        faiss.normalize_L2(q_vec)
        scores, idxs = self.index.search(q_vec, min(top_k * 3, self.index.ntotal))
        idxs = idxs[0]
        scores = scores[0]

        results = []
        for idx, score in zip(idxs, scores):
            if idx < 0 or idx >= len(self.documents):
                continue

            doc = self.documents[idx]
            meta = getattr(doc, "metadata", {}) or {}

            if filters:
                ok = True
                for k, v in filters.items():
                    if isinstance(v, dict):
                        if "$in" in v:
                            if meta.get(k) not in v["$in"]:
                                ok = False
                                break
                    else:
                        if meta.get(k) != v:
                            ok = False
                            break
                if not ok:
                    continue

            results.append({
                "document": doc,
                "score": float(score),
                "metadata": meta,
            })
            
            if len(results) >= top_k:
                break

        return results

    def save(self, path: str) -> None:
        path = Path(path)
        faiss.write_index(self.index, str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".docs.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

    def load(self, path: str) -> None:
        path = Path(path)
        self.index = faiss.read_index(str(path.with_suffix(".faiss")))
        with open(path.with_suffix(".docs.pkl"), "rb") as f:
            self.documents = pickle.load(f)
        print(f"✅ 로드 완료: {len(self.documents)}개 문서")


# ─────────────────────────────────────────────
# Chroma Backend (기본 버전 - 안정성 우선)
# ─────────────────────────────────────────────
chromadb = None
CHROMA_AVAILABLE = False


class ChromaBackend:
    def __init__(self, embed_fn: EmbedFn, collection_name: str = "kised_rag"):
        global chromadb, CHROMA_AVAILABLE
        if not CHROMA_AVAILABLE:
            try:
                import chromadb as _chromadb  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Chroma backend is unavailable on this environment. "
                    f"Original error: {exc}"
                ) from exc
            chromadb = _chromadb
            CHROMA_AVAILABLE = True
        if not CHROMA_AVAILABLE:
            raise ImportError("Chroma 사용을 위해 패키지를 설치하세요: pip install chromadb")

        self.embed_fn = embed_fn
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection(name=collection_name)   

class RAGSystem:
    def __init__(self, embedding_model: str, provider: str = "faiss"):
        """
        embedding_model : 사용할 임베딩 모델 이름 (text-embedding-3-large 등)
        provider        : 'faiss', 'simple', 'chroma' 중 선택
        """

        # 1) 임베딩 함수 준비
        self.embed_fn = get_embedding_function(embedding_model)

        # 2) 백엔드 선택
        if provider == "faiss":
            # 차원 자동 계산
            sample_vec = self.embed_fn(["hello"])[0]
            dim = len(sample_vec)
            self.backend = FaissBackend(self.embed_fn, dim)

        elif provider == "simple":
            self.backend = SimpleBackend(self.embed_fn)

        elif provider == "chroma":
            self.backend = ChromaBackend(self.embed_fn)

        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def embed(self, text):
        """
        SemanticMatchingAgent에서 사용하는 단일 텍스트 → 벡터 변환 함수.
        Simple/FAISS/Chroma 모두 동일하게 동작함.
        """
        if text is None:
            return None

        # 리스트이면 그대로 처리
        if isinstance(text, list):
            return self.embed_fn(text)

        # 문자열이면 리스트로 감싸서 처리
        return self.embed_fn([text])[0]

        

    # Wrapper
    def add(self, docs):
        self.backend.add_documents(docs)
        
    def add_documents(self, docs):
        self.backend.add_documents(docs)

    def retrieve(self, query, top_k=5, filters=None):
        return self.backend.search(query, top_k=top_k, filters=filters)
    

    def search(self, query, top_k=10, filters=None):
        return self.backend.search(query, top_k=top_k, filters=filters)

    def save(self, path):
        self.backend.save(path)

    def load(self, path):
        self.backend.load(path)

            
