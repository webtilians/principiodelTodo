from datetime import datetime

from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.temporal import parse_weekday_range
from src.infinito3.temporal_context import TemporalSemanticContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import MemoryKind, MemoryRecord, MemoryStatus


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def add_fact(store, predicate, value, content=None, *, status=MemoryStatus.ACTIVE, supersedes_id=None):
    record = MemoryRecord(
        content=content or f"{predicate}: {value}",
        kind=MemoryKind.USER_MODEL,
        importance=0.95,
        fact_subject="user",
        fact_predicate=predicate,
        fact_value=value,
        status=status,
        supersedes_id=supersedes_id,
    )
    return store.add(record)


def test_parse_weekday_range_english_and_spanish():
    now = datetime(2026, 10, 5, 8, 0)  # Monday
    assert parse_weekday_range("from Wednesday through Sunday", now) == (
        datetime(2026, 10, 7).date(),
        datetime(2026, 10, 11).date(),
    )
    assert parse_weekday_range("de miércoles a domingo", now) == (
        datetime(2026, 10, 7).date(),
        datetime(2026, 10, 11).date(),
    )


def test_structured_profile_query_recovers_all_requested_slots_by_predicate():
    store = InMemoryMemoryStore()
    add_fact(store, "location", "Ghent", "I live in Ghent.")
    add_fact(store, "bike", "Norco Sight", "My bike is Norco Sight.")
    add_fact(store, "studying_language", "Danish", "I am studying Danish.")
    add_fact(store, "occupation", "machine-learning engineer", "I work as machine-learning engineer.")
    builder = TemporalSemanticContextBuilder(store, TemporalGoalEngine())

    packet = builder.build(
        "What are my current city, bike, language and job?",
        memory_candidates=[],
    )

    for value in ("Ghent", "Norco Sight", "Danish", "machine-learning engineer"):
        assert value in packet.rendered


def test_historical_predecessor_is_rendered_as_explicit_temporal_relation():
    store = InMemoryMemoryStore()
    old = add_fact(store, "location", "Ghent", "I live in Ghent.", status=MemoryStatus.SUPERSEDED)
    add_fact(store, "location", "Malmö", "I live in Malmö.", supersedes_id=old.id)
    builder = TemporalSemanticContextBuilder(store, TemporalGoalEngine())

    packet = builder.build(
        "Where did I live immediately before Malmö?",
        memory_candidates=[],
    )

    assert "relation=immediately_previous" in packet.rendered
    assert "Ghent" in packet.rendered
    assert "Malmö" not in packet.rendered


def test_commitment_query_supports_weekday_range_and_open_followup():
    clock = Clock(datetime(2026, 10, 5, 8, 0))
    goals = TemporalGoalEngine(now_fn=clock)
    goals.add_structured_goal("physiotherapy", due_at=datetime(2026, 10, 7, 8, 30))
    goals.add_structured_goal("language academy", due_at=datetime(2026, 10, 8, 19, 0))
    goals.add_structured_goal("parcel", due_at=datetime(2026, 10, 9, 13, 15))
    goals.add_structured_goal("tyre", due_at=datetime(2026, 10, 10, 10, 0))
    goals.add_structured_goal("Sara", due_at=datetime(2026, 10, 11, 18, 0))
    builder = TemporalSemanticContextBuilder(InMemoryMemoryStore(), goals, now_fn=clock)

    packet = builder.build("What commitments do I have from Wednesday through Sunday?", memory_candidates=[])
    for value in ("physiotherapy", "language academy", "parcel", "tyre", "Sara"):
        assert value in packet.rendered

    open_packet = builder.build("What remains open now?", memory_candidates=[])
    assert "physiotherapy" in open_packet.rendered
    assert "Sara" in open_packet.rendered


def test_before_weekday_is_strict_upper_bound_for_goals():
    clock = Clock(datetime(2026, 10, 8, 8, 0))  # Thursday
    goals = TemporalGoalEngine(now_fn=clock)
    goals.add_structured_goal("parcel", due_at=datetime(2026, 10, 9, 13, 15))
    goals.add_structured_goal("Sara", due_at=datetime(2026, 10, 11, 18, 0))
    goals.add_structured_goal("tyre", due_at=datetime(2026, 10, 12, 9, 0))
    builder = TemporalSemanticContextBuilder(InMemoryMemoryStore(), goals, now_fn=clock)

    packet = builder.build("Which commitments are still pending before Monday?", memory_candidates=[])
    assert "parcel" in packet.rendered
    assert "Sara" in packet.rendered
    assert "tyre" not in packet.rendered
