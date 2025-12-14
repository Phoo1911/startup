# core/embedder.py
"""
임베딩 함수 및 텍스트 청킹 (KoSimCSE 최적화)
- OpenAI
- Sentence Transformers (SBERT) - 메모리 최적화
- Simple (TF-IDF fallback)
"""

from typing import List, Callable
import numpy as np
from config.settings import Config

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ─────────────────────────────────────────────
# 1) OpenAI Embeddings
# ─────────────────────────────────────────────
def get_openai_embedder(model_name: str) -> Callable[[List[str]], np.ndarray]:
    """OpenAI 임베딩 함수 생성"""
    try:
        from openai import OpenAI
        
        client = OpenAI(api_key=Config.LLM_API_KEY)
        
        def embed_fn(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.array([])
            
            response = client.embeddings.create(
                model=model_name,
                input=texts
            )
            
            embeddings = [item.embedding for item in response.data]
            return np.array(embeddings, dtype=np.float32)
        
        return embed_fn
        
    except ImportError:
        raise ImportError("OpenAI 사용을 위해 'openai' 패키지를 설치하세요: pip install openai")
    except Exception as e:
        raise RuntimeError(f"OpenAI 임베딩 초기화 실패: {e}")


# ─────────────────────────────────────────────
# 2) Sentence Transformers (SBERT) - 메모리 최적화
# ─────────────────────────────────────────────
def get_sbert_embedder(model_name: str) -> Callable[[List[str]], np.ndarray]:
    """Sentence-BERT 임베딩 함수 생성 (메모리 최적화)"""
    try:
        from sentence_transformers import SentenceTransformer
        
        # 🔥 메모리 최적화: CPU 사용 또는 GPU
        device = "cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu"
        
        print(f"📦 모델 로딩 중: {model_name}")
        print(f"   - Device: {device}")
        
        model = SentenceTransformer(model_name, device=device)
        
        # 🔥 GPU인 경우 half precision (메모리 절약)
        if device == "cuda":
            try:
                model = model.half()
                print("   - Half precision 적용 (메모리 50% 절약)")
            except Exception as e:
                print(f"   - Half precision 실패: {e}")
        
        # 배치 크기 설정 (Config에서 가져오기)
        batch_size = int(getattr(Config, "EMBEDDING_BATCH_SIZE", 8))
        print(f"   - Batch size: {batch_size}")
        
        def embed_fn(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.array([])
            
            # 🔥 배치 단위로 처리 (메모리 절약)
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # 🔥 그래디언트 계산 안 함 (메모리 절약)
                if TORCH_AVAILABLE:
                    with torch.no_grad():
                        embeddings = model.encode(
                            batch,
                            show_progress_bar=False,
                            convert_to_numpy=True,
                            normalize_embeddings=True,
                            batch_size=len(batch),
                        )
                else:
                    embeddings = model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    )
                
                all_embeddings.append(embeddings)
                
                # 🔥 GPU 메모리 정리
                if device == "cuda" and TORCH_AVAILABLE:
                    torch.cuda.empty_cache()
                
                # 진행률 표시 (큰 데이터셋인 경우)
                if len(texts) > 100 and i % (batch_size * 10) == 0 and i > 0:
                    print(f"   - Progress: {i}/{len(texts)} ({i*100//len(texts)}%)")
            
            result = np.vstack(all_embeddings) if len(all_embeddings) > 1 else all_embeddings[0]
            return result.astype(np.float32)
        
        return embed_fn
        
    except ImportError:
        raise ImportError("Sentence Transformers 사용을 위해 패키지를 설치하세요: pip install sentence-transformers torch")
    except Exception as e:
        raise RuntimeError(f"SBERT 임베딩 초기화 실패: {e}")


# ─────────────────────────────────────────────
# 3) Simple TF-IDF Embedder (Fallback)
# ─────────────────────────────────────────────
def get_simple_embedder() -> Callable[[List[str]], np.ndarray]:
    """TF-IDF 기반 간단한 임베딩 (fallback)"""
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        
        vectorizer = TfidfVectorizer(
            max_features=384,
            ngram_range=(1, 2),
            min_df=1
        )
        
        # 초기화용 더미 데이터
        vectorizer.fit(["초기화 텍스트"])
        
        def embed_fn(texts: List[str]) -> np.ndarray:
            if not texts:
                return np.array([])
            
            # 새로운 텍스트로 재학습
            try:
                embeddings = vectorizer.transform(texts).toarray()
            except:
                # 학습되지 않은 단어가 있으면 재학습
                vectorizer.fit(texts)
                embeddings = vectorizer.transform(texts).toarray()
            
            return embeddings.astype(np.float32)
        
        return embed_fn
        
    except ImportError:
        raise ImportError("Simple embedder 사용을 위해 'scikit-learn'을 설치하세요: pip install scikit-learn")


# ─────────────────────────────────────────────
# 4) 통합 임베딩 함수 팩토리
# ─────────────────────────────────────────────
def get_embedding_function(model_name: str) -> Callable[[List[str]], np.ndarray]:
    """
    모델 이름에 따라 적절한 임베딩 함수 반환
    
    Args:
        model_name: 임베딩 모델 이름
            - "text-embedding-3-small" → OpenAI
            - "BM-K/KoSimCSE-roberta" → SBERT (KoSimCSE)
            - "sentence-transformers/..." → SBERT
            - "simple" → TF-IDF
    
    Returns:
        embed_fn: List[str] → np.ndarray 함수
    """
    print(f"\n📦 임베딩 모델 로드 중: {model_name}")
    
    # OpenAI 모델
    if model_name.startswith("text-embedding"):
        print("   - Provider: OpenAI")
        return get_openai_embedder(model_name)
    
    # Sentence Transformers 모델
    elif model_name.startswith("sentence-transformers/") or "/" in model_name:
        print("   - Provider: Sentence Transformers")
        return get_sbert_embedder(model_name)
    
    # Simple fallback
    elif model_name == "simple":
        print("   - Provider: Simple TF-IDF (fallback)")
        print("⚠️  프로덕션용이 아닙니다")
        return get_simple_embedder()
    
    # 기본값: SBERT paraphrase-multilingual 모델
    else:
        print(f"⚠️  알 수 없는 모델: {model_name}")
        print("   - 기본 모델 사용: paraphrase-multilingual-MiniLM-L12-v2")
        return get_sbert_embedder("paraphrase-multilingual-MiniLM-L12-v2")


# ─────────────────────────────────────────────
# 5) 텍스트 청킹 (Config 지원)
# ─────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    긴 텍스트를 청크로 분할
    
    Args:
        text: 원본 텍스트
        chunk_size: 청크 크기 (문자 단위) - None이면 Config에서 가져옴
        overlap: 청크 간 겹침 (문자 단위) - None이면 Config에서 가져옴
    
    Returns:
        청크 리스트
    """
    # Config에서 설정 가져오기
    if chunk_size is None:
        chunk_size = int(getattr(Config, "CHUNK_SIZE", 500))
    if overlap is None:
        overlap = int(getattr(Config, "CHUNK_OVERLAP", 50))
    
    if not text or len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # 문장 경계에서 자르기 시도
        if end < len(text):
            # 한국어/영어 문장 부호 모두 지원
            for sep in ['. ', '! ', '? ', '\n', '。', '！', '？']:
                last_sep = text[start:end].rfind(sep)
                if last_sep != -1:
                    end = start + last_sep + len(sep)
                    break
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        start = end - overlap
        
        # 무한 루프 방지
        if start >= len(text):
            break
    
    return chunks


# ─────────────────────────────────────────────
# 6) 배치 처리 유틸리티 (Config 지원)
# ─────────────────────────────────────────────
def embed_in_batches(
    texts: List[str],
    embed_fn: Callable[[List[str]], np.ndarray],
    batch_size: int = None
) -> np.ndarray:
    """
    대량의 텍스트를 배치 단위로 임베딩
    
    Args:
        texts: 텍스트 리스트
        embed_fn: 임베딩 함수
        batch_size: 배치 크기 - None이면 Config에서 가져옴
    
    Returns:
        전체 임베딩 배열 (N, D)
    """
    if not texts:
        return np.array([])
    
    if batch_size is None:
        batch_size = int(getattr(Config, "EMBEDDING_BATCH_SIZE", 16))
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        embeddings = embed_fn(batch)
        all_embeddings.append(embeddings)
        
        # 진행률 표시
        if len(texts) > 100 and i % (batch_size * 10) == 0 and i > 0:
            print(f"   배치 처리: {i}/{len(texts)} ({i*100//len(texts)}%)")
    
    return np.vstack(all_embeddings)