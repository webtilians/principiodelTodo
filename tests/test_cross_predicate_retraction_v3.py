from datetime import datetime

from src.infinito3.cognitive_events import CognitiveEvent, CognitiveEventType
from src.infinito3.event_extractor import TemporalCognitiveEventExtractor
from src.infinito3.persistent_memory import HashEmbeddingProvider
from src.infinito3.semantic_goal_state import SemanticGoalTemporalState
from src.infinito3.semantic_temporal_memory import SemanticTemporalMemoryStore
from src.infinito3.temporal_goals import TemporalGoalEngine


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def test_unique_value_retraction_survives_predicate_drift():
    clock = Clock(datetime(2026, 9, 14, 10, 0, 0))
    store = SemanticTemporalMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    state = SemanticGoalTemporalState(now_fn=clock)
    goals = TemporalGoalEngine(now_fn=clock)
    fallback = TemporalCognitiveEventExtractor(now_fn=clock)

    state.apply(fallback.extract("Me gusta tomar kombucha."), memory_store=store, goal_engine=goals)
    assert "kombucha" in state.current_values("likes")

    drifted = CognitiveEvent(
        CognitiveEventType.RETRACT_FACT,
        "I no longer drink kombucha.",
        predicate="drinks",
        value="kombucha",
        occurred_at=clock(),
        confidence=0.95,
    )
    state.apply([drifted], memory_store=store, goal_engine=goals)

    assert "kombucha" not in state.current_values("likes")
    assert not [record for record in store.all() if record.fact_value == "kombucha"]


def test_cross_predicate_retraction_fails_closed_when_value_is_ambiguous():
    clock = Clock(datetime(2026, 9, 14, 10, 0, 0))
    store = SemanticTemporalMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    state = SemanticGoalTemporalState(now_fn=clock)
    goals = TemporalGoalEngine(now_fn=clock)

    # Two independently structured active facts share the same value: cross-slot
    # reconciliation must not guess which one to retract.
    events = [
        CognitiveEvent(CognitiveEventType.ASSERT_FACT, "note a", predicate="slot_a", value="shared", occurred_at=clock()),
        CognitiveEvent(CognitiveEventType.ASSERT_FACT, "note b", predicate="slot_b", value="shared", occurred_at=clock()),
    ]
    state.apply(events, memory_store=store, goal_engine=goals)
    drifted = CognitiveEvent(
        CognitiveEventType.RETRACT_FACT,
        "remove shared",
        predicate="other_slot",
        value="shared",
        occurred_at=clock(),
    )
    state.apply([drifted], memory_store=store, goal_engine=goals)

    assert state.current_fact("slot_a") is not None
    assert state.current_fact("slot_b") is not None
