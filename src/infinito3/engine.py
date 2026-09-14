from dataclasses import asdict
from typing import Optional, Sequence

from .advanced_temporal_state import ResolvedTemporalCognitiveState
from .cognitive_events import CognitiveEventType
from .generalized_context_builder import GeneralizedContextBuilder
from .interfaces import ContextBuilder, EmbeddingProvider, GoalEngine, MemoryGate, MemoryStore, SafetyFilter
from .memory import InMemoryMemoryStore, RuleBasedMemoryGate
from .safety import SensitiveInformationFilter
from .semantic_event_extractor import SemanticCognitiveEventExtractor
from .structured_context import StructuredTemporalContextBuilder
from .structured_retrieval import StructuredStateRetriever
from .temporal_goals import TemporalGoalEngine
from .types import CognitiveDecision, ConversationTurn, MemoryRecord, SafetyLevel


class CognitiveEngine:
    """Orchestrates the INFINITO 3.0 cognitive layer."""

    def __init__(self, memory_store: Optional[MemoryStore] = None, memory_gate: Optional[MemoryGate] = None,
                 safety_filter: Optional[SafetyFilter] = None, goal_engine: Optional[GoalEngine] = None,
                 context_builder: Optional[ContextBuilder] = None, event_extractor=None,
                 temporal_state=None, structured_retriever=None):
        self.memory_store = memory_store or InMemoryMemoryStore()
        self.memory_gate = memory_gate or RuleBasedMemoryGate()
        self.safety_filter = safety_filter or SensitiveInformationFilter()
        self.goal_engine = goal_engine or TemporalGoalEngine()
        self.event_extractor = event_extractor or SemanticCognitiveEventExtractor()
        self.temporal_state = temporal_state or ResolvedTemporalCognitiveState()
        self.structured_retriever = structured_retriever or StructuredStateRetriever(
            self.temporal_state,
            self.memory_store,
        )
        self.context_builder = context_builder or StructuredTemporalContextBuilder(
            memory_store=self.memory_store,
            goal_engine=self.goal_engine,
        )

    @classmethod
    def persistent(cls, db_path: str = "data/infinito3_memory.db", embedding_provider: Optional[EmbeddingProvider] = None, **kwargs):
        from .semantic_temporal_memory import SemanticTemporalMemoryStore
        return cls(memory_store=SemanticTemporalMemoryStore(path=db_path, embedding_provider=embedding_provider), **kwargs)

    def process(self, text: str, top_k: int = 5, *, context_budget_tokens: int = 1200,
                recent_turns: Optional[Sequence[ConversationTurn]] = None) -> CognitiveDecision:
        safety = self.safety_filter.inspect(text)
        if safety.level == SafetyLevel.FORBIDDEN:
            return CognitiveDecision(input_text=text, safety=safety, gate=None, context=[], context_packet=None)

        query = safety.redacted_text if safety.redacted_text else text
        gate = self.memory_gate.evaluate(query)
        allow_persistence = safety.level == SafetyLevel.SAFE

        event_usage_before = self._usage_snapshot(self.event_extractor)
        planner = getattr(self.structured_retriever, "planner", None)
        planner_usage_before = self._usage_snapshot(planner)

        # State mutation happens before retrieval so the current turn immediately
        # becomes authoritative. Questions normally produce no events.
        events = self.event_extractor.extract(query) if allow_persistence else []
        transitions = self.temporal_state.apply(
            events,
            memory_store=self.memory_store,
            goal_engine=self.goal_engine,
        ) if events else []

        goal_types = {
            CognitiveEventType.CREATE_GOAL,
            CognitiveEventType.COMPLETE_GOAL,
            CognitiveEventType.CANCEL_GOAL,
            CognitiveEventType.RESCHEDULE_GOAL,
        }
        structured_goal = any(event.type in goal_types for event in events)
        created_goals = self.goal_engine.ingest(query) if allow_persistence and not structured_goal else []

        structured_context = list(self.structured_retriever.retrieve(query)) if self.structured_retriever else []
        semantic_context = self._search_memory(
            query,
            top_k=top_k,
            include_history=self.temporal_state.wants_history(query),
        )
        context = self._merge_context(structured_context, semantic_context)

        context_packet = self.context_builder.build(
            query,
            memory_candidates=context,
            recent_turns=recent_turns,
            max_tokens=context_budget_tokens,
        )
        if context_packet is not None:
            context_packet.diagnostics["structured_state"] = (
                self.structured_retriever.diagnostics() if self.structured_retriever else {}
            )
            context_packet.diagnostics["semantic_event_extractor"] = self._usage_delta(
                event_usage_before,
                self._usage_snapshot(self.event_extractor),
            )
            context_packet.diagnostics["structured_query_planner"] = self._usage_delta(
                planner_usage_before,
                self._usage_snapshot(planner),
            )

        structured_persistence = bool(events)
        stored_memory_id = next((memory_id for transition in transitions for memory_id in transition.memory_ids), None)
        if stored_memory_id is None and allow_persistence and gate.should_store and not structured_persistence:
            stored = self.memory_store.add(
                MemoryRecord(
                    content=query,
                    kind=gate.kind,
                    importance=gate.importance,
                    metadata={"gate_reasons": gate.reasons},
                )
            )
            stored_memory_id = stored.id

        created_goal_ids = [goal.id for goal in created_goals]
        create_ids = {event.id for event in events if event.type == CognitiveEventType.CREATE_GOAL}
        for transition in transitions:
            if transition.event_id in create_ids:
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

    @staticmethod
    def _merge_context(structured, semantic):
        merged = []
        seen = set()
        for record in list(structured) + list(semantic):
            identifier = getattr(record, "id", None)
            if identifier and identifier in seen:
                continue
            if identifier:
                seen.add(identifier)
            merged.append(record)
        return merged

    @staticmethod
    def _usage_snapshot(component):
        getter = getattr(component, "usage_snapshot", None)
        if not callable(getter):
            return {}
        try:
            return dict(getter())
        except Exception:
            return {}

    @staticmethod
    def _usage_delta(before, after):
        keys = set(before) | set(after)
        result = {}
        for key in keys:
            try:
                result[key] = int(after.get(key, 0)) - int(before.get(key, 0))
            except (TypeError, ValueError):
                continue
        return result
