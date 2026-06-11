"""
agentic_hybrid/config.py â€” Central frozen configuration

Fixes applied:
  - base_dir was Path(__file__).parent.parent which pointed TWO levels up (to
    the project root's parent). Changed to Path(__file__).parent.parent so that
    when config.py lives at agentic_hybrid/config.py the base_dir is the
    project root (D:/startup/), and the default cache/ resolves to
    D:/startup/cache/ â€” matching the location where AutoRAG writes its index.
    Added a clear docstring so the intent is unambiguous.
  - Added FALLBACK_DOCS_COUNT for filter_node graceful fallback (single source
    of truth instead of a magic number).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

RetrievalMode = Literal["VECTOR", "HYBRID"]
LLMProvider = Literal["transformers", "openai", "huggingface", "hf", "vllm", "google"]


@dataclass(frozen=True)
class AgenticHybridConfig:
    # â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # base_dir = project root (parent of the agentic_hybrid/ package).
    # config.py is at:  <project_root>/agentic_hybrid/config.py
    #   => Path(__file__).resolve().parent      = agentic_hybrid/
    #   => Path(__file__).resolve().parent.parent = project root  âœ“
    base_dir: Path = Path(__file__).resolve().parent.parent

    # Relative paths are resolved against base_dir at runtime via properties.
    vectorstore_dir: Path = Path("cache")
    faiss_index_name: str = "rag_index.faiss"
    docs_pickle_name: str = "rag_index.docs.pkl"

    # â”€â”€ Retrieval â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    RETRIEVAL_MODE: RetrievalMode = "HYBRID"

    # â”€â”€ Pipeline flags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    USE_HYDE: bool = True
    USE_RERANKER: bool = True
    USE_FILTER: bool = True
    USE_AGENTIC_PLANNER: bool = True
    USE_DOC_TYPE_ROUTER: bool = True
    USE_CROSS_DOC_ENRICH: bool = True
    USE_REVISE: bool = True
    USE_FRESHNESS_RERANK: bool = True
    USE_DEADLINE_GUARD: bool = True

    # â”€â”€ Models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # KoSimCSE produces 768-dim vectors; replace with your actual embedding model.
    EMBEDDING_MODEL_NAME: str = "BM-K/KoSimCSE-roberta"
    CROSS_ENCODER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Supported: transformers | openai | huggingface(hf) | vllm | google
    # Default generation path: transformers + Qwen
    LLM_PROVIDER: LLMProvider = "transformers"
    LLM_MODEL_NAME: str = "Qwen/Qwen3-4B-Instruct-2507"
    LLM_TEMPERATURE: float = 0.15
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    HF_API_KEY: str | None = None
    GOOGLE_API_KEY: str | None = None

    # â”€â”€ Top-K ladder â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    TOP_K_RETRIEVAL: int = 40   # candidates from vector/BM25
    TOP_K_RERANK: int = 20      # kept after cross-encoder
    TOP_K_FINAL: int = 5        # kept after eligibility filter
    RRF_K: int = 60             # RRF smoothing constant

    # â”€â”€ Filter fallback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # When all docs are filtered out, keep this many top-ranked docs as fallback.
    FALLBACK_DOCS_COUNT: int = 2

    # â”€â”€ Debug â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    PRINT_REASONING_TRACE: bool = True

    # â”€â”€ Computed paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @property
    def vectorstore_dir_abs(self) -> Path:
        if self.vectorstore_dir.is_absolute():
            return self.vectorstore_dir
        return self.base_dir / self.vectorstore_dir

    @property
    def faiss_index_path(self) -> Path:
        return self.vectorstore_dir_abs / self.faiss_index_name

    @property
    def docs_pickle_path(self) -> Path:
        return self.vectorstore_dir_abs / self.docs_pickle_name


# â”€â”€ Environment loader â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_config() -> AgenticHybridConfig:
    cfg = AgenticHybridConfig(
        vectorstore_dir=Path(os.getenv("AH_VECTORSTORE_DIR", "cache")),
        RETRIEVAL_MODE=os.getenv("AH_RETRIEVAL_MODE", "HYBRID").upper(),  # type: ignore[arg-type]
        USE_HYDE=_env_bool("AH_USE_HYDE", True),
        USE_RERANKER=_env_bool("AH_USE_RERANKER", True),
        USE_FILTER=_env_bool("AH_USE_FILTER", True),
        USE_AGENTIC_PLANNER=_env_bool("AH_USE_AGENTIC_PLANNER", True),
        USE_DOC_TYPE_ROUTER=_env_bool("AH_USE_DOC_TYPE_ROUTER", True),
        USE_CROSS_DOC_ENRICH=_env_bool("AH_USE_CROSS_DOC_ENRICH", True),
        USE_REVISE=_env_bool("AH_USE_REVISE", True),
        EMBEDDING_MODEL_NAME=os.getenv("AH_EMBEDDING_MODEL", "BM-K/KoSimCSE-roberta"),
        CROSS_ENCODER_MODEL_NAME=os.getenv(
            "AH_CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        ),
        LLM_PROVIDER=os.getenv("AH_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "transformers")),
        LLM_MODEL_NAME=os.getenv(
            "AH_LLM_MODEL",
            os.getenv("LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507"),
        ),
        LLM_TEMPERATURE=float(os.getenv("AH_LLM_TEMPERATURE", os.getenv("LLM_TEMPERATURE", "0.15"))),
        OPENAI_API_KEY=os.getenv("AH_OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")),
        OPENAI_BASE_URL=os.getenv("AH_OPENAI_BASE_URL", os.getenv("LLM_BASE_URL", "http://localhost:8000/v1")),
        HF_API_KEY=os.getenv("AH_HF_API_KEY", os.getenv("HF_TOKEN")),
        GOOGLE_API_KEY=os.getenv("AH_GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", os.getenv("GEMINI_API_KEY"))),
        TOP_K_RETRIEVAL=int(os.getenv("AH_TOP_K_RETRIEVAL", "40")),
        TOP_K_RERANK=int(os.getenv("AH_TOP_K_RERANK", "20")),
        TOP_K_FINAL=int(os.getenv("AH_TOP_K_FINAL", "5")),
        RRF_K=int(os.getenv("AH_RRF_K", "60")),
        FALLBACK_DOCS_COUNT=int(os.getenv("AH_FALLBACK_DOCS", "2")),
        PRINT_REASONING_TRACE=_env_bool("AH_PRINT_TRACE", True),
    )
    if cfg.RETRIEVAL_MODE not in {"VECTOR", "HYBRID"}:
        raise ValueError(f"Invalid RETRIEVAL_MODE: {cfg.RETRIEVAL_MODE!r}")
    if str(cfg.LLM_PROVIDER).lower() not in {"transformers", "openai", "huggingface", "hf", "vllm", "google"}:
        raise ValueError(
            f"Invalid LLM_PROVIDER: {cfg.LLM_PROVIDER!r}. "
            "Use transformers | openai | huggingface | hf | vllm | google"
        )
    return cfg


def apply_experiment_mode(cfg: AgenticHybridConfig, mode: str) -> AgenticHybridConfig:
    """
    Convenience helper for ablation experiments.

    baseline : pure vector retrieval, no HyDE / reranker / filter / planner
    baseline_pure : dedicated minimal baseline graph (retrieve -> generate)
    autorag  : hybrid retrieval + HyDE, no reranker / filter / planner
    full : AutoRAG-like retrieval core + post-retrieval planner/generation only
    full_dense : dense-only retrieval core + post-retrieval planner/generation
    no_intent_dense : full_dense - intent classifier routing hint
    no_planner_dense : no_intent_dense - planner
    no_doc_type_router_dense : no_intent_dense - doc type router
    no_revise_dense : no_intent_dense - revise
    strict_full : legacy full with deadline inheritance/review/final policy gate
    no_hyde : full - HyDE
    no_reranker : full - cross-encoder reranker
    no_freshness : full - freshness reranking
    no_deadline : legacy retrieval-first full without deadline hard exclusion
    no_intent : full - intent classifier routing hint
    full_without_filter : explicit alias of current retrieval-first full
    full_without_doc_type_router : full - doc_type routing/lexical boost
    """
    mode = mode.lower()
    if mode == "baseline":
        return replace(
            cfg,
            RETRIEVAL_MODE="VECTOR",
            USE_HYDE=False,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=False,
            USE_DOC_TYPE_ROUTER=False,
            USE_CROSS_DOC_ENRICH=False,
            USE_REVISE=False,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "baseline_pure":
        return replace(
            cfg,
            RETRIEVAL_MODE="VECTOR",
            USE_HYDE=False,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=False,
            USE_DOC_TYPE_ROUTER=False,
            USE_CROSS_DOC_ENRICH=False,
            USE_REVISE=False,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "autorag":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=False,
            USE_DOC_TYPE_ROUTER=False,
            USE_CROSS_DOC_ENRICH=False,
            USE_REVISE=False,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "full":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "full_dense":
        return replace(
            cfg,
            RETRIEVAL_MODE="VECTOR",
            USE_HYDE=False,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "no_intent_dense":
        return replace(
            cfg,
            RETRIEVAL_MODE="VECTOR",
            USE_HYDE=False,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=True,
        )
    if mode == "no_planner_dense":
        return replace(
            cfg,
            RETRIEVAL_MODE="VECTOR",
            USE_HYDE=False,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=False,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=True,
        )
    if mode == "no_doc_type_router_dense":
        return replace(
            cfg,
            RETRIEVAL_MODE="VECTOR",
            USE_HYDE=False,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=False,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "no_revise_dense":
        return replace(
            cfg,
            RETRIEVAL_MODE="VECTOR",
            USE_HYDE=False,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=False,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "full_without_filter":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "full_without_doc_type_router":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=False,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "strict_full":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=True,
            USE_FILTER=True,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=True,
            USE_DEADLINE_GUARD=True,
        )
    if mode == "no_hyde":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=False,
            USE_RERANKER=True,
            USE_FILTER=True,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=True,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "no_reranker":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=False,
            USE_FILTER=True,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=True,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "no_freshness":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=True,
            USE_FILTER=True,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "no_deadline":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=True,
            USE_FILTER=True,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=True,
            USE_DEADLINE_GUARD=False,
        )
    if mode == "no_intent":
        return replace(
            cfg,
            RETRIEVAL_MODE="HYBRID",
            USE_HYDE=True,
            USE_RERANKER=False,
            USE_FILTER=False,
            USE_AGENTIC_PLANNER=True,
            USE_DOC_TYPE_ROUTER=True,
            USE_CROSS_DOC_ENRICH=True,
            USE_REVISE=True,
            USE_FRESHNESS_RERANK=False,
            USE_DEADLINE_GUARD=False,
        )
    raise ValueError(
        "Unsupported experiment mode: "
        f"{mode!r}. Use baseline | baseline_pure | autorag | full | full_dense | no_intent_dense | no_planner_dense | no_doc_type_router_dense | no_revise_dense | full_without_filter | full_without_doc_type_router | strict_full | no_hyde | no_reranker | no_freshness | no_deadline | no_intent"
    )
