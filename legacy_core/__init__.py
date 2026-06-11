"""Lightweight exports for the core package.

Avoid importing optional heavy dependencies (for example Chroma) at package
import time. Some CLI/index-rebuild paths only need the FAISS pipeline and
should not fail just because Chroma is unavailable on the host.
"""

from .llm_client import LLMClient
from .orchestrator import AgenticOrchestrator

__all__ = [
    "LLMClient",
    "AgenticOrchestrator",
]

try:
    from .rag_system import RAGSystem
except Exception:
    RAGSystem = None  # type: ignore[assignment]
else:
    __all__.append("RAGSystem")

try:
    from .orchestrator_enhanced import EnhancedAgenticOrchestrator
except Exception:
    EnhancedAgenticOrchestrator = None  # type: ignore[assignment]
else:
    __all__.append("EnhancedAgenticOrchestrator")
