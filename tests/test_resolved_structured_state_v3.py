from datetime import datetime

from src.infinito3.cognitive_events import CognitiveEvent, CognitiveEventType
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.persistent_memory import HashEmbeddingProvider
from src.infinito3.resolved_temporal_memory import ResolvedSemanticTemporalMemoryStore
from src.infinito3.resolved_temporal_state import ResolvedSemanticTemporalState
from src.infinito3.semantic_reranker import SemanticRerankResult
from src.infinito3.structured_temporal_context import StructuredTemporalContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import MemoryKind, MemoryRecord


class Clock:
    def __init__(self, current):
        self.current = current

    def __call__(self):
        return self.current


class SelectingReranker:
    def __init__(self, needle):
        self.needle = needle.lower()
        self.calls = 0

    def rerank(self, query, candidates):
        self.calls += 1
        selected = [
            str(item.memory_id)
            for item in candidates
            if self.needle in str(item.metadata.get("fact_value") or item.content).lower()
        ]
        return SemanticRerankResult(
            selected_ids=selected,
            success=True,
            provider="test",
            model="fake",
            usage={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )


def test_unresolved_cross_language_retraction_can_close_one_grounded_target():
    clock = Clock(datetime(2026, 11, 2, 9, 0))
    store = ResolvedSemanticTemporalMemoryStore(path=":memory:", embedding_provider=HashEmbeddingProvider())
    resolver = SelectingReranker("observar aves")
    state = ResolvedSemanticTemporalState(now_fn=clock, retraction_reranker=resolver)

    state.apply(
        [CognitiveEvent(
            CognitiveEventType.ASSERT_PREFERENCE,
            "Me he aficionado a observar aves.",
            predicate="likes",
            value="observar aves",
            occurred_at=clock(),
        )],
        memory_store=store,
    )
    state.apply(
        [CognitiveEvent(
            CognitiveEventType.RETRACT_PREFERENCE,
            "Birdwatching no longer appeals to me.",
            predicate="likes",
            value="Birdwatching",
            occurred_at=clock(),
        )],
        memory_store=store,
    )

    assert state.current_values("likes") == []
    records = store.all(include_inactive=True)
    assert len(records) == 1
    assert records[0].status.value == "superseded"
    assert records[0].metadata["retraction_target_resolver"] == "llm_semantic_membership"
    assert state.retraction_resolution_stats()["resolved"] == 1


def test_today_morning_query_keeps_same_day_goal_even_after_due_time():
    clock = Clock(datetime(2026, 11, 9, 10, 0))
    goals = TemporalGoalEngine(now_fn=clock)
    goals.add_structured_goal("tyre change", due_at=datetime(2026, 11, 9, 9, 0))
    builder = StructuredTemporalContextBuilder(InMemoryMemoryStore(), goals, now_fn=clock)

    packet = builder.build(
        "Today is Monday. What appointment do I have this morning and when?",
        memory_candidates=[],
    )

    assert "tyre change" in packet.rendered
    assert "2026-11-09T09:00" in packet.rendered


def test_worded_arithmetic_is_self_contained_and_does_not_leak_profile_memory():
    clock = Clock(datetime(2026, 11, 2, 9, 0))
    memory = InMemoryMemoryStore()
    memory.add(MemoryRecord(
        content="My bike is a Pivot Switchblade.",
        kind=MemoryKind.USER_MODEL,
        importance=0.9,
        fact_subject="user",
        fact_predicate="bike",
        fact_value="pivot switchblade",
    ))
    builder = StructuredTemporalContextBuilder(memory, TemporalGoalEngine(now_fn=clock), now_fn=clock)

    packet = builder.build("Dime solamente cuánto es 15 por 6.", memory_candidates=memory.all())

    assert "Pivot Switchblade" not in packet.rendered
    assert packet.items == []


def test_singular_preference_query_can_rerank_full_active_preference_state():
    clock = Clock(datetime(2026, 11, 2, 9, 0))
    memory = InMemoryMemoryStore()
    radio = memory.add(MemoryRecord(
        content="I enjoy restoring old radios.",
        kind=MemoryKind.USER_MODEL,
        importance=0.82,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="restoring old radios",
    ))
    memory.add(MemoryRecord(
        content="I've gotten really into woodworking lately.",
        kind=MemoryKind.USER_MODEL,
        importance=0.82,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="woodworking",
    ))
    reranker = SelectingReranker("restoring old radios")
    builder = StructuredTemporalContextBuilder(
        memory,
        TemporalGoalEngine(now_fn=clock),
        now_fn=clock,
        reranker=reranker,
    )

    packet = builder.build("Name my radio-related hobby.", memory_candidates=[])

    assert reranker.calls == 1
    assert "restoring old radios" in packet.rendered
    assert any(item.memory_id == radio.id for item in packet.items)
