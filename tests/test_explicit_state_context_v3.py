from datetime import datetime

from src.infinito3.explicit_state_context import ExplicitStateContextBuilder
from src.infinito3.resolved_temporal_memory import ResolvedSemanticTemporalMemoryStore
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import MemoryKind, MemoryRecord


class Clock:
    def __init__(self, current):
        self.current = current

    def __call__(self):
        return self.current


def test_goal_query_renders_authoritative_empty_state():
    clock = Clock(datetime(2026, 11, 9, 12, 0))
    store = ResolvedSemanticTemporalMemoryStore(path=":memory:")
    goals = TemporalGoalEngine(now_fn=clock)
    builder = ExplicitStateContextBuilder(store, goals, now_fn=clock)

    packet = builder.build("Do I still have any commitments open?", memory_candidates=[])

    assert "No active goals or commitments." in packet.rendered
    assert any(item.metadata.get("state") == "no_active_goals" for item in packet.items)


def test_explicit_retraction_history_is_recovered_without_vector_hit():
    clock = Clock(datetime(2026, 11, 3, 10, 0))
    store = ResolvedSemanticTemporalMemoryStore(path=":memory:")
    record = store.add(MemoryRecord(
        content="Me he aficionado a observar aves.",
        kind=MemoryKind.USER_MODEL,
        importance=0.82,
        confidence=0.95,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="observar aves",
    ))
    store.retract_memory_id(
        record.id,
        reason="retract_preference",
        source_text="Birdwatching no longer appeals to me.",
        at=clock(),
    )
    store.add(MemoryRecord(
        content="Recently I've started enjoying urban sketching.",
        kind=MemoryKind.USER_MODEL,
        importance=0.82,
        confidence=0.95,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="urban sketching",
    ))
    builder = ExplicitStateContextBuilder(store, TemporalGoalEngine(now_fn=clock), now_fn=clock)

    packet = builder.build(
        "What activity did I explicitly say no longer appeals to me?",
        memory_candidates=[],
    )

    assert "observar aves" in packet.rendered
    assert "Birdwatching no longer appeals to me" in packet.rendered
    assert "urban sketching" not in packet.rendered
