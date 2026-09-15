from datetime import datetime

from src.infinito3.context_intent import resolve_context_intent
from src.infinito3.intent_context_v4 import DeterministicIntentContextBuilder
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.temporal_goals import TemporalGoalEngine


def test_profile_query_decomposition():
    intent = resolve_context_intent("Read back my name, home city, bicycle, language and occupation.")
    assert intent.mode == "facts"
    assert intent.predicates == frozenset({"name", "location", "bike", "studying_language", "occupation"})


def test_morning_boundary_includes_1220_but_not_1310():
    now = datetime(2027, 2, 6, 11, 0)
    goals = TemporalGoalEngine(now_fn=lambda: now)
    goals.add_structured_goal("lens pickup", due_at=datetime(2027, 2, 6, 12, 20))
    goals.add_structured_goal("frame delivery", due_at=datetime(2027, 2, 6, 13, 10))
    builder = DeterministicIntentContextBuilder(InMemoryMemoryStore(), goals, now_fn=lambda: now)
    packet = builder.build("Which commitment is due this morning?")
    assert "lens pickup" in packet.rendered
    assert "frame delivery" not in packet.rendered
    assert packet.diagnostics["context_intent"]["window"] == ["2027-02-06T05:00:00", "2027-02-06T13:00:00"]
