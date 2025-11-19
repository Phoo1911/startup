"""
RAG 시스템
"""
import numpy as np
import pickle
from pathlib import Path
from typing import List, Tuple
from models.data import Document

# Sentence-BERT 임포트 시도
try:
    from sentence_transformers import SentenceTransformer
    SBERT_AVAILABLE = True
except ImportError:
    SBERT_AVAILABLE = False

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """코사인 유사도"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

class RAGSystem:
    """RAG (검색증강생성) 시스템"""
    
    def __init__(self, embedding_model: str = "simple"):
        self.documents: List[Document] = []
        self.embedding_model = embedding_model
        
        if embedding_model == "sentence-transformers" and SBERT_AVAILABLE:
            print("🔄 Sentence-BERT 모델 로딩 중...")
            self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            print("✅ 임베딩 모델 로드 완료")
        else:
            self.embedder = None
            print("✅ 간단한 벡터 모델 사용")
    
    def add_documents(self, documents: List[Document]):
        """문서 추가 및 임베딩"""
        print(f"📚 {len(documents)}개 문서 임베딩 중...")
        
        for i, doc in enumerate(documents):
            if doc.embedding is None:
                doc.embedding = self._embed_text(doc.text)
            
            if (i + 1) % 100 == 0:
                print(f"   진행: {i+1}/{len(documents)}")
        
        self.documents.extend(documents)
        print(f"✅ 총 {len(self.documents)}개 문서 인덱싱 완료")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[Document, float]]:
        """유사 문서 검색"""
        if not self.documents:
            return []
        
        query_embedding = self._embed_text(query)
        similarities = []
        
        for doc in self.documents:
            if doc.embedding is not None:
                sim = cosine_similarity(query_embedding, doc.embedding)
                similarities.append((doc, sim))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def _embed_text(self, text: str) -> np.ndarray:
        """텍스트 임베딩"""
        if self.embedder:
            return self.embedder.encode(text, convert_to_numpy=True)
        else:
            # 간단한 TF-IDF 스타일 벡터
            words = text.lower().split()
            vocab = list(set(words))[:300]
            vector = np.zeros(300)
            for i, word in enumerate(vocab):
                if i < 300:
                    vector[i] = words.count(word)
            return vector / (np.linalg.norm(vector) + 1e-8)
    
    def save(self, path: str):
        """RAG 인덱스 저장"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.documents, f)
        print(f"💾 RAG 인덱스 저장: {path}")
    
    def load(self, path: str):
        """RAG 인덱스 로드"""
        with open(path, 'rb') as f:
            self.documents = pickle.load(f)
        print(f"📂 RAG 인덱스 로드: {len(self.documents)}개 문서")
