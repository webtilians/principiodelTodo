from datetime import datetime

from src.infinito3.cognitive_events import CognitiveEvent, CognitiveEventType
from src.infinito3.event_extractor import TemporalCognitiveEventExtractor
from src.infinito3.persistent_memory import HashEmbeddingProvider
from src.infinito3.temporal_context import TemporalSemanticContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.temporal_memory import TemporalAwareSQLiteMemoryStore
from src.infinito3.temporal_state import TemporalCognitiveState
from src.infinito3.types import MemoryStatus


def _clock(value):
    return lambda: value


def test_exclusive_fact_replacement_closes_old_memory_but_keeps_history():
    now = datetime(2026, 9, 14, 9, 0, 0)
    store = TemporalAwareSQLiteMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    state = TemporalCognitiveState(now_fn=_clock(now))

    first = CognitiveEvent(CognitiveEventType.ASSERT_FACT, "Vivo en Sevilla.", predicate="location", value="Sevilla", occurred_at=now, metadata={"exclusive": True})
    second = CognitiveEvent(CognitiveEventType.REPLACE_FACT, "I moved to Bilbao.", predicate="location", value="Bilbao", occurred_at=now, metadata={"exclusive": True})
    state.apply([first], memory_store=store)
    state.apply([second], memory_store=store)

    active = store.all()
    assert len(active) == 1
    assert active[0].fact_value == "bilbao"
    history = store.history_for_fact("user", "location")
    assert [record.fact_value for record in history] == ["sevilla", "bilbao"]
    assert history[0].status == MemoryStatus.SUPERSEDED
    assert history[1].supersedes_id == history[0].id


def test_preference_retraction_removes_current_but_preserves_audit_history():
    now = datetime(2026, 9, 14, 9, 0, 0)
    store = TemporalAwareSQLiteMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    state = TemporalCognitiveState(now_fn=_clock(now))
    state.apply([
        CognitiveEvent(CognitiveEventType.ASSERT_PREFERENCE, "Me gusta el kayak.", predicate="likes", value="kayak", occurred_at=now)
    ], memory_store=store)
    state.apply([
        CognitiveEvent(CognitiveEventType.RETRACT_PREFERENCE, "He dejado el kayak.", predicate="likes", value="kayak", occurred_at=now)
    ], memory_store=store)

    assert state.current_values("likes") == []
    assert not store.all()
    historical = store.search("kayak", top_k=5, include_inactive=True)
    assert historical and historical[0].fact_value == "kayak"
    assert historical[0].status == MemoryStatus.SUPERSEDED


def test_goal_lifecycle_reschedule_and_completion_are_explicit():
    now = datetime(2026, 10, 5, 8, 0, 0)
    store = TemporalAwareSQLiteMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    goals = TemporalGoalEngine(now_fn=_clock(now))
    state = TemporalCognitiveState(now_fn=_clock(now))

    create = CognitiveEvent(CognitiveEventType.CREATE_GOAL, "El miércoles a las 11 tengo dentista.", predicate="goal", value="dentista", due_at=datetime(2026, 10, 7, 11), occurred_at=now)
    state.apply([create], memory_store=store, goal_engine=goals)
    assert goals.all()[0].due_at == datetime(2026, 10, 7, 11)

    reschedule = CognitiveEvent(CognitiveEventType.RESCHEDULE_GOAL, "Movido al jueves a las 17.", predicate="goal", value="dentista", due_at=datetime(2026, 10, 8, 17), occurred_at=now)
    state.apply([reschedule], memory_store=store, goal_engine=goals)
    assert goals.all()[0].due_at == datetime(2026, 10, 8, 17)
    assert "miércoles" not in goals.all()[0].description.lower()

    complete = CognitiveEvent(CognitiveEventType.COMPLETE_GOAL, "Ya fui al dentista.", predicate="goal", value="dentista", occurred_at=now)
    state.apply([complete], memory_store=store, goal_engine=goals)
    assert goals.all()[0].completed is True
    assert goals.all()[0].metadata["lifecycle"] == "completed"


def test_temporal_context_can_surface_direct_predecessor():
    now = datetime(2026, 9, 14, 9, 0, 0)
    store = TemporalAwareSQLiteMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    goals = TemporalGoalEngine(now_fn=_clock(now))
    state = TemporalCognitiveState(now_fn=_clock(now))
    extractor = TemporalCognitiveEventExtractor(now_fn=_clock(now))
    state.apply(extractor.extract("Vivo en Sevilla."), memory_store=store, goal_engine=goals)
    state.apply(extractor.extract("I moved to Bilbao last month. Bilbao is where I live now."), memory_store=store, goal_engine=goals)

    query = "¿En qué ciudad vivía antes de Bilbao?"
    candidates = store.search(query, top_k=10, include_inactive=True)
    builder = TemporalSemanticContextBuilder(memory_store=store, goal_engine=goals, now_fn=_clock(now))
    packet = builder.build(query, memory_candidates=candidates, max_tokens=300)
    assert "Sevilla" in packet.rendered
    assert "Bilbao" not in packet.rendered
