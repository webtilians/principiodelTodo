"""INFINITO 3.0 cognitive layer.

The package is independent from Streamlit. The cognitive core, context builder,
LLM adapters, orchestration loop and evaluation harness are replaceable
components.
"""

from .cognitive_loop import CognitiveLoop
from .context_builder import ApproximateTokenEstimator, BalancedContextBuilder
from .engine import CognitiveEngine
from .evaluation import (
    CallablePairwiseJudge,
    DeterministicEvaluator,
    EvaluationExpectation,
    EvaluationHarness,
    EvaluationReport,
    EvaluationScenario,
    EvaluationSummary,
    PairwiseJudgeResult,
    RunMetrics,
    ScenarioResult,
    standard_evaluation_suite,
)
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
    "CallablePairwiseJudge",
    "CognitiveDecision",
    "CognitiveEngine",
    "CognitiveLoop",
    "CognitiveRunResult",
    "ContextItem",
    "ContextPacket",
    "ContextSource",
    "ConversationTurn",
    "DeterministicEvaluator",
    "EvaluationExpectation",
    "EvaluationHarness",
    "EvaluationReport",
    "EvaluationScenario",
    "EvaluationSummary",
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
    "PairwiseJudgeResult",
    "RecordingLLMAdapter",
    "RuleBasedMemoryGate",
    "RunMetrics",
    "SafetyLevel",
    "ScenarioResult",
    "SensitiveInformationFilter",
    "SimpleFactExtractor",
    "SQLiteCognitiveMemoryStore",
    "standard_evaluation_suite",
]
