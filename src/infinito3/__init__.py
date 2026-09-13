"""INFINITO 3.0 cognitive layer.

This package is intentionally independent from Streamlit and any concrete LLM
provider. The public entry point is :class:`CognitiveEngine`.
"""

from .engine import CognitiveEngine
from .memory import InMemoryMemoryStore, RuleBasedMemoryGate
from .persistent_memory import (
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SimpleFactExtractor,
    SQLiteCognitiveMemoryStore,
)
from .safety import SensitiveInformationFilter
from .types import (
    CognitiveDecision,
    MaintenanceReport,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    SafetyLevel,
)

__all__ = [
    "CognitiveEngine",
    "CognitiveDecision",
    "HashEmbeddingProvider",
    "InMemoryMemoryStore",
    "MaintenanceReport",
    "MemoryKind",
    "MemoryRecord",
    "MemoryStatus",
    "OpenAIEmbeddingProvider",
    "RuleBasedMemoryGate",
    "SafetyLevel",
    "SensitiveInformationFilter",
    "SimpleFactExtractor",
    "SQLiteCognitiveMemoryStore",
]
