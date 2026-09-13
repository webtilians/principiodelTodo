from typing import List, Protocol

from .types import GateDecision, Goal, MemoryRecord, SafetyDecision


class SafetyFilter(Protocol):
    def inspect(self, text: str) -> SafetyDecision:
        ...


class MemoryGate(Protocol):
    def evaluate(self, text: str) -> GateDecision:
        ...


class EmbeddingProvider(Protocol):
    def embed(self, text: str) -> List[float]:
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
