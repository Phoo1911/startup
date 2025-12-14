"""
시스템 설정 관리
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


class Config:
    """시스템 설정"""
    BASE_URL = "https://apis.data.go.kr/B552735"
    TIMEOUT = 30
    ENCODING = "utf-8-sig"

    SERVICE_KEY = os.getenv("KISED_SERVICE_KEY", "")

    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
    LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
    LOCAL_LLM_MODEL = os.getenv("LOCAL_LLM_MODEL", "Qwen/Qwen2-7B-Instruct")
    LLM_API_KEY = os.getenv("OPENAI_API_KEY", None)
    LLM_BASE_URL = os.getenv("LLM_BASE_URL", None)

    RAG_ENABLED = True
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BM-K/KoSimCSE-roberta")
    TOP_K_RETRIEVAL = 20
    SIMILARITY_THRESHOLD = 0.65
    EMBEDDING_PROVIDER = "faiss"

    AGENT_VERBOSE = os.getenv("AGENT_VERBOSE", "false").lower() == "true"
    REASONING_ENABLED = True

    DATA_DAYS_RANGE = int(os.getenv("DATA_DAYS_RANGE", "180"))
    MAX_PAGES_PER_ENDPOINT = int(os.getenv("MAX_PAGES_PER_ENDPOINT", "5"))

    CACHE_DIR = Path("cache")
    CACHE_REFRESH_DAYS = int(os.getenv("CACHE_REFRESH_DAYS", "1"))
    REPORTS_DIR = Path("reports")

    EVAL_LOG_ENABLED = os.getenv("EVAL_LOG_ENABLED", "false").lower() == "true"
    EVAL_LOG_DIR = Path("eval_logs")

    WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
    WEB_PORT = int(os.getenv("WEB_PORT", "8501"))
    DEBUG = os.getenv("DEBUG", "false").lower() == "true"

    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

    @classmethod
    def init_dirs(cls):
        cls.CACHE_DIR.mkdir(exist_ok=True)
        cls.REPORTS_DIR.mkdir(exist_ok=True)
        cls.EVAL_LOG_DIR.mkdir(exist_ok=True)
        Path("logs").mkdir(exist_ok=True)

    @classmethod
    def validate(cls):
        errors = []
        if not cls.SERVICE_KEY:
            errors.append("KISED_SERVICE_KEY 환경변수가 설정되지 않았습니다")
        if cls.LLM_PROVIDER in ("openai", "anthropic") and not cls.LLM_API_KEY:
            errors.append(f"{cls.LLM_PROVIDER} 사용 시 API 키가 필요합니다")
        if errors:
            raise ValueError("\n".join(errors))
        return True


Config.init_dirs()
