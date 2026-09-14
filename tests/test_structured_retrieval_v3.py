from datetime import datetime

from src.infinito3.advanced_temporal_state import ResolvedTemporalCognitiveState
from src.infinito3.cognitive_events import CognitiveEvent, CognitiveEventType
from src.infinito3.persistent_memory import HashEmbeddingProvider
from src.infinito3.semantic_temporal_memory import SemanticTemporalMemoryStore
from src.infinito3.structured_context import StructuredTemporalContextBuilder
from src.infinito3.structured_retrieval import StructuredStateRetriever
from src.infinito3.temporal_goals import TemporalGoalEngine


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def apply_fact(state, store, goals, event_type, predicate, value, text):
    state.apply(
        [
            CognitiveEvent(
                event_type,
                text,
                predicate=predicate,
                value=value,
                occurred_at=datetime(2026, 9, 14, 9, 0, 0),
                metadata={"exclusive": predicate != "likes"},
            )
        ],
        memory_store=store,
        goal_engine=goals,
    )


def make_state():
    clock = Clock(datetime(2026, 9, 14, 12, 0, 0))
    store = SemanticTemporalMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    goals = TemporalGoalEngine(now_fn=clock)
    state = ResolvedTemporalCognitiveState(now_fn=clock)
    return clock, store, goals, state


def test_structured_retrieval_fetches_all_requested_current_slots_before_vector_search():
    _, store, goals, state = make_state()
    apply_fact(state, store, goals, CognitiveEventType.ASSERT_FACT, "name", "Nora", "Me llamo Nora.")
    apply_fact(state, store, goals, CognitiveEventType.ASSERT_FACT, "location", "Utrecht", "Vivo en Utrecht.")
    apply_fact(state, store, goals, CognitiveEventType.ASSERT_FACT, "bike", "Yeti SB160", "Mi bici es una Yeti SB160.")
    apply_fact(state, store, goals, CognitiveEventType.ASSERT_FACT, "studying_language", "Japanese", "Estoy estudiando Japanese.")

    retriever = StructuredStateRetriever(state, store)
    records = retriever.retrieve("What is my name, city, bike and language now?")
    found = {record.fact_predicate: record.fact_value for record in records}
    assert found == {
        "name": "nora",
        "location": "utrecht",
        "bike": "yeti sb160",
        "studying_language": "japanese",
    }
    assert all(record.metadata.get("temporal_relation") == "current" for record in records)


def test_structured_history_follows_lineage_and_renderer_states_relation_explicitly():
    clock, store, goals, state = make_state()
    apply_fact(state, store, goals, CognitiveEventType.ASSERT_FACT, "location", "Zaragoza", "Vivo en Zaragoza.")
    apply_fact(state, store, goals, CognitiveEventType.REPLACE_FACT, "location", "Lyon", "I moved to Lyon.")
    apply_fact(state, store, goals, CognitiveEventType.REPLACE_FACT, "location", "Utrecht", "I moved to Utrecht.")

    query = "¿En qué ciudad vivía justo antes de Utrecht?"
    retriever = StructuredStateRetriever(state, store)
    records = retriever.retrieve(query)
    assert len(records) == 1
    assert records[0].fact_value == "lyon"
    assert records[0].metadata["temporal_relation"] == "immediately_previous"
    assert records[0].metadata["temporal_before_value"] == "Utrecht"

    builder = StructuredTemporalContextBuilder(memory_store=store, goal_engine=goals, now_fn=clock)
    packet = builder.build(query, memory_candidates=records, max_tokens=300)
    assert "TEMPORAL FACT" in packet.rendered
    assert "relation=immediately_previous" in packet.rendered
    assert "before=Utrecht" in packet.rendered
    assert "value=lyon" in packet.rendered
