from typing import List, Optional, Protocol, Sequence

from .types import (
    ContextPacket,
    ConversationTurn,
    GateDecision,
    Goal,
    MemoryRecord,
    SafetyDecision,
)


class SafetyFilter(Protocol):
    def inspect(self, text: str) -> SafetyDecision:
        ...


class MemoryGate(Protocol):
    def evaluate(self, text: str) -> GateDecision:
        ...


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]:
        ...


class TokenEstimator(Protocol):
    def estimate(self, text: str) -> int:
        ...


class MemoryStore(Protocol):
    def add(self, record: MemoryRecord) -> MemoryRecord:
        ...

    def search(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        ...

    def all(self) -> List[MemoryRecord]:
        ...


class GoalEngine(Protocol):
    def ingest(self, text: str) -> List[Goal]:
        ...

    def due(self) -> List[Goal]:
        ...

    def all(self) -> List[Goal]:
        ...


class ContextBuilder(Protocol):
    def build(
        self,
        query: str,
        *,
        memory_candidates: Optional[Sequence[MemoryRecord]] = None,
        recent_turns: Optional[Sequence[ConversationTurn]] = None,
        max_tokens: int = 1200,
        candidate_k: int = 20,
    ) -> ContextPacket:
        ...
