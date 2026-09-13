from typing import Optional

from .goals import SimpleGoalEngine
from .interfaces import GoalEngine, MemoryGate, MemoryStore, SafetyFilter
from .memory import InMemoryMemoryStore, RuleBasedMemoryGate
from .safety import SensitiveInformationFilter
from .types import CognitiveDecision, MemoryRecord, SafetyLevel


class CognitiveEngine:
    """Orchestrates the INFINITO 3.0 cognitive layer.

    Processing order is deliberate:
    1. Safety inspection
    2. Retrieval from existing memory
    3. Memory-gate evaluation
    4. Goal extraction
    5. Persistence, only when policy allows it

    No LLM provider or UI dependency belongs here.
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        memory_gate: Optional[MemoryGate] = None,
        safety_filter: Optional[SafetyFilter] = None,
        goal_engine: Optional[GoalEngine] = None,
    ):
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.memory_gate = memory_gate or RuleBasedMemoryGate()
        self.safety_filter = safety_filter or SensitiveInformationFilter()
        self.goal_engine = goal_engine or SimpleGoalEngine()

    def process(self, text: str, top_k: int = 5) -> CognitiveDecision:
        safety = self.safety_filter.inspect(text)

        # Secrets are blocked before any cognitive subsystem receives them.
        if safety.level == SafetyLevel.FORBIDDEN:
            return CognitiveDecision(
                input_text=text,
                safety=safety,
                gate=None,
                context=[],
            )

        # Retrieve before writing so the current message cannot retrieve itself.
        query = safety.redacted_text if safety.redacted_text else text
        context = self.memory_store.search(query, top_k=top_k)

        gate = self.memory_gate.evaluate(query)

        # Conservative v3.0 policy: sensitive PII can be used in the current
        # interaction but is never written to long-term memory by default.
        allow_persistence = safety.level == SafetyLevel.SAFE

        created_goals = self.goal_engine.ingest(query) if allow_persistence else []
        stored_memory_id = None

        if allow_persistence and gate.should_store:
            record = MemoryRecord(
                content=query,
                kind=gate.kind,
                importance=gate.importance,
                metadata={"gate_reasons": gate.reasons},
            )
            stored = self.memory_store.add(record)
            stored_memory_id = stored.id

        return CognitiveDecision(
            input_text=text,
            safety=safety,
            gate=gate,
            stored_memory_id=stored_memory_id,
            created_goal_ids=[goal.id for goal in created_goals],
            context=context,
        )
