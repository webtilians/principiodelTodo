from dataclasses import asdict
from typing import Optional, Sequence

from .cognitive_events import CognitiveEventType, RuleBasedCognitiveEventExtractor
from .generalized_context_builder import GeneralizedContextBuilder
from .interfaces import (
    ContextBuilder,
    EmbeddingProvider,
    GoalEngine,
    MemoryGate,
    MemoryStore,
    SafetyFilter,
)
from .memory import InMemoryMemoryStore, RuleBasedMemoryGate
from .safety import SensitiveInformationFilter
from .temporal_goals import TemporalGoalEngine
from .temporal_state import TemporalCognitiveState
from .types import CognitiveDecision, ConversationTurn, MemoryRecord, SafetyLevel


class CognitiveEngine:
    """Orchestrates the INFINITO 3.0 cognitive layer.

    Current processing order:
    1. Safety inspection
    2. Retrieval from pre-write memory state
    3. Cognitive-event extraction
    4. Temporal state reduction and projection
    5. Legacy gate/parser fallback for unstructured turns
    6. Context construction
    7. Legacy persistence only when no structured event already projected state

    Secrets are blocked before any cognitive subsystem. Retrieval stays pre-write
    so a user's current statement cannot retrieve itself as if it were old
    evidence, while newly-created goals may still be visible immediately.
    """

    def __init__(
        self,
        memory_store: Optional[MemoryStore] = None,
        memory_gate: Optional[MemoryGate] = None,
        safety_filter: Optional[SafetyFilter] = None,
        goal_engine: Optional[GoalEngine] = None,
        context_builder: Optional[ContextBuilder] = None,
        event_extractor=None,
        temporal_state: Optional[TemporalCognitiveState] = None,
    ):
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.memory_gate = memory_gate or RuleBasedMemoryGate()
        self.safety_filter = safety_filter or SensitiveInformationFilter()
        self.goal_engine = goal_engine or TemporalGoalEngine()
        self.event_extractor = event_extractor or RuleBasedCognitiveEventExtractor()
        self.temporal_state = temporal_state or TemporalCognitiveState()
        self.context_builder = context_builder or GeneralizedContextBuilder(
            memory_store=self.memory_store,
            goal_engine=self.goal_engine,
        )

    @classmethod
    def persistent(
        cls,
        db_path: str = "data/infinito3_memory.db",
        embedding_provider: Optional[EmbeddingProvider] = None,
        **kwargs,
    ) -> "CognitiveEngine":
        """Create an engine backed by temporal-aware persistent v3 memory."""
        from .temporal_memory import TemporalAwareSQLiteMemoryStore

        store = TemporalAwareSQLiteMemoryStore(
            path=db_path,
            embedding_provider=embedding_provider,
        )
        return cls(memory_store=store, **kwargs)

    def process(
        self,
        text: str,
        top_k: int = 5,
        *,
        context_budget_tokens: int = 1200,
        recent_turns: Optional[Sequence[ConversationTurn]] = None,
    ) -> CognitiveDecision:
        safety = self.safety_filter.inspect(text)

        if safety.level == SafetyLevel.FORBIDDEN:
            return CognitiveDecision(
                input_text=text,
                safety=safety,
                gate=None,
                context=[],
                context_packet=None,
            )

        query = safety.redacted_text if safety.redacted_text else text
        include_history = self.temporal_state.wants_history(query)
        context = self._search_memory(query, top_k=top_k, include_history=include_history)
        gate = self.memory_gate.evaluate(query)
        allow_persistence = safety.level == SafetyLevel.SAFE

        events = self.event_extractor.extract(query) if allow_persistence else []
        transitions = self.temporal_state.apply(
            events,
            memory_store=self.memory_store if allow_persistence else None,
            goal_engine=self.goal_engine if allow_persistence else None,
        ) if events else []

        event_goal_types = {
            CognitiveEventType.CREATE_GOAL,
            CognitiveEventType.COMPLETE_GOAL,
            CognitiveEventType.CANCEL_GOAL,
            CognitiveEventType.RESCHEDULE_GOAL,
        }
        structured_goal_event = any(event.type in event_goal_types for event in events)

        # Preserve the original parser as a fallback for goal forms that the new
        # event extractor does not yet understand. Avoid double-creating a goal
        # when an explicit structured event already handled the turn.
        created_goals = []
        if allow_persistence and not structured_goal_event:
            created_goals = self.goal_engine.ingest(query)

        context_packet = self.context_builder.build(
            query,
            memory_candidates=context,
            recent_turns=recent_turns,
            max_tokens=context_budget_tokens,
        )

        structured_persistence = any(
            event.type in {
                CognitiveEventType.ASSERT_FACT,
                CognitiveEventType.REPLACE_FACT,
                CognitiveEventType.RETRACT_FACT,
                CognitiveEventType.ASSERT_PREFERENCE,
                CognitiveEventType.RETRACT_PREFERENCE,
                CognitiveEventType.CREATE_GOAL,
                CognitiveEventType.COMPLETE_GOAL,
                CognitiveEventType.CANCEL_GOAL,
                CognitiveEventType.RESCHEDULE_GOAL,
                CognitiveEventType.STORE_NOTE,
            }
            for event in events
        )

        stored_memory_id = None
        if allow_persistence and gate.should_store and not structured_persistence:
            record = MemoryRecord(
                content=query,
                kind=gate.kind,
                importance=gate.importance,
                metadata={"gate_reasons": gate.reasons},
            )
            stored = self.memory_store.add(record)
            stored_memory_id = stored.id

        created_goal_ids = [goal.id for goal in created_goals]
        create_event_ids = {
            event.id for event in events if event.type == CognitiveEventType.CREATE_GOAL
        }
        for transition in transitions:
            if transition.event_id in create_event_ids:
                created_goal_ids.extend(transition.goal_ids)

        return CognitiveDecision(
            input_text=text,
            safety=safety,
            gate=gate,
            stored_memory_id=stored_memory_id,
            created_goal_ids=created_goal_ids,
            cognitive_event_ids=[event.id for event in events],
            temporal_transitions=[asdict(transition) for transition in transitions],
            context=context,
            context_packet=context_packet,
        )

    def _search_memory(self, query: str, *, top_k: int, include_history: bool):
        if include_history:
            try:
                return self.memory_store.search(query, top_k=top_k, include_inactive=True)
            except TypeError:
                pass
        return self.memory_store.search(query, top_k=top_k)
