from datetime import datetime

from src.infinito3.context_intent import resolve_context_intent
from src.infinito3.intent_context_v4 import DeterministicIntentContextBuilder
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.semantic_reranker import SemanticRerankResult
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import MemoryKind, MemoryRecord


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


def test_latest_preference_orders_after_membership():
    class Membership:
        def __init__(self):
            self.query = None

        def rerank(self, query, candidates):
            self.query = query
            selected = [
                str(item.memory_id) for item in candidates
                if item.metadata.get("fact_value") in {"linocut", "bookbinding"}
            ]
            return SemanticRerankResult(selected_ids=selected, success=True)

    store = InMemoryMemoryStore()
    for value, day in (("linocut", 2), ("urban birding", 5), ("bookbinding", 8)):
        store.add(MemoryRecord(
            f"I enjoy {value}", MemoryKind.USER_MODEL, .9,
            fact_predicate="likes", fact_value=value,
            created_at=datetime(2027, 1, day), updated_at=datetime(2027, 1, day),
        ))
    membership = Membership()
    builder = DeterministicIntentContextBuilder(
        store, TemporalGoalEngine(), reranker=membership,
        now_fn=lambda: datetime(2027, 1, 10),
    )
    query = "Which craft did I add most recently?"
    intent = resolve_context_intent(query)
    assert intent.mode == "preferences"
    assert intent.ordering == "latest"
    packet = builder.build(query)
    assert "most recently" not in membership.query.lower()
    assert "bookbinding" in packet.rendered
    assert "linocut" not in packet.rendered
    assert "urban birding" not in packet.rendered
