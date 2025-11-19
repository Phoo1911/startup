"""
시스템 설정 관리
"""
import os
from pathlib import Path

class Config:
    """시스템 설정"""
    # API 설정
    BASE_URL = "https://apis.data.go.kr/B552735"
    TIMEOUT = 30
    ENCODING = 'utf-8-sig'
    
    # 환경변수에서 API 키 로드
    SERVICE_KEY = os.getenv("KISED_SERVICE_KEY", "")
    
    # LLM 설정
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai", "anthropic", None
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", None)
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)
    
    # RAG 설정
    RAG_ENABLED = True
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers")
    TOP_K_RETRIEVAL = 20
    SIMILARITY_THRESHOLD = 0.65
    
    # Agent 설정
    AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
    REASONING_ENABLED = True
    
    # 캐시 설정
    CACHE_DIR = Path("cache")
    CACHE_REFRESH_DAYS = 1
    REPORTS_DIR = Path("reports")
    
    # 웹 설정
    WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT = int(os.getenv("WEB_PORT", "8000"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"
    
    @classmethod
    def init_dirs(cls):
        """필요한 디렉토리 생성"""
        cls.CACHE_DIR.mkdir(exist_ok=True)
        cls.REPORTS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def validate(cls):
        """필수 설정 검증"""
        if not cls.SERVICE_KEY:
            raise ValueError("KISED_SERVICE_KEY 환경변수가 설정되지 않았습니다")
        
        if cls.LLM_PROVIDER and not cls.LLM_API_KEY:
            raise ValueError(f"{cls.LLM_PROVIDER} 사용 시 API 키가 필요합니다")

# 초기화
Config.init_dirs()
