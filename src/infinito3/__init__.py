"""INFINITO 3.0 cognitive layer.

The package is independent from Streamlit. The cognitive core, context builder,
LLM adapters and orchestration loop are all replaceable components.
"""

from .cognitive_loop import CognitiveLoop
from .context_builder import ApproximateTokenEstimator, BalancedContextBuilder
from .engine import CognitiveEngine
from .llm_adapter import OpenAIResponsesAdapter, RecordingLLMAdapter
from .memory import InMemoryMemoryStore, RuleBasedMemoryGate
from .persistent_memory import (
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SimpleFactExtractor,
    SQLiteCognitiveMemoryStore,
)
from .safety import SensitiveInformationFilter
from .types import (
    ABComparison,
    CognitiveDecision,
    CognitiveRunResult,
    ContextItem,
    ContextPacket,
    ContextSource,
    ConversationTurn,
    LLMMessage,
    LLMRequest,
    LLMResponse,
    MaintenanceReport,
    MemoryKind,
    MemoryRecord,
    MemoryStatus,
    SafetyLevel,
)

__all__ = [
    "ABComparison",
    "ApproximateTokenEstimator",
    "BalancedContextBuilder",
    "CognitiveDecision",
    "CognitiveEngine",
    "CognitiveLoop",
    "CognitiveRunResult",
    "ContextItem",
    "ContextPacket",
    "ContextSource",
    "ConversationTurn",
    "HashEmbeddingProvider",
    "InMemoryMemoryStore",
    "LLMMessage",
    "LLMRequest",
    "LLMResponse",
    "MaintenanceReport",
    "MemoryKind",
    "MemoryRecord",
    "MemoryStatus",
    "OpenAIEmbeddingProvider",
    "OpenAIResponsesAdapter",
    "RecordingLLMAdapter",
    "RuleBasedMemoryGate",
    "SafetyLevel",
    "SensitiveInformationFilter",
    "SimpleFactExtractor",
    "SQLiteCognitiveMemoryStore",
]
