from datetime import datetime

import pytest

from src.infinito3.context_intent import resolve_context_intent
from src.infinito3.engine import CognitiveEngine
from src.infinito3.intent_context import IntentContextBuilder, IntentEventExtractor
from src.infinito3.llm_adapter import RecordingLLMAdapter
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import MemoryRecord, MemoryKind
from src.infinito3.types import MemoryStatus
from src.infinito3.semantic_reranker import SemanticRerankResult


@pytest.mark.parametrize("query", [
    "Return only 19 times 6.", "What is 120 divided by 8?",
    "Dime solo cuanto es 21 por 3.", "Name the capital of Senegal.",
    "Explain refraction briefly.", "Give one fact about seals.",
])
def test_standalone_does_not_search_or_call_extractor(query):
    class NoSearch(InMemoryMemoryStore):
        def search(self, *args, **kwargs):
            raise AssertionError("standalone request searched memory")
    store = NoSearch()
    store.add(MemoryRecord("Private unrelated note", MemoryKind.SEMANTIC, 1))
    goals = TemporalGoalEngine()
    adapter = RecordingLLMAdapter()
    extractor = IntentEventExtractor(adapter)
    builder = IntentContextBuilder(store, goals)
    engine = CognitiveEngine(memory_store=store, goal_engine=goals,
                             context_builder=builder, event_extractor=extractor)
    decision = engine.process(query)
    assert not decision.context_packet.items
    assert not adapter.requests
    assert decision.context_packet.diagnostics["context_intent"]["mode"] == "standalone"


@pytest.mark.parametrize("query", [
    "Remember this literal phrase: return only 19 times 6.",
    "What is my monthly rent times 3?", "Explain my preferences.",
    "Cancel my appointment.",
])
def test_mutations_and_personal_queries_are_not_bypassed(query):
    assert resolve_context_intent(query).retrieve


def test_calendar_window_and_past_morning():
    now = datetime(2027, 1, 4, 14)
    goals = TemporalGoalEngine(now_fn=lambda: now)
    for day in (5, 6, 7, 8, 9):
        goals.add_structured_goal(f"appointment-{day}", due_at=datetime(2027, 1, day, 9))
    builder = IntentContextBuilder(InMemoryMemoryStore(), goals, now_fn=lambda: now)
    packet = builder.build("What's on my calendar from Tuesday to Saturday?")
    assert len(packet.items) == 5
    now = datetime(2027, 1, 9, 14)
    packet = builder.build("It is Saturday now. What appointment was due this morning?")
    assert len(packet.items) == 1
    assert "appointment-9" in packet.rendered


@pytest.mark.parametrize("query", [
    "Which interest did I say doesn't interest me these days?",
    "What interest has lost its appeal for me?",
    "Which hobby did I stop?",
])
def test_history_intent(query):
    intent = resolve_context_intent(query)
    assert intent.mode == "preferences"
    assert intent.historical


def test_note_aliases_and_pet_scope():
    now = datetime(2027, 1, 4)
    store = InMemoryMemoryStore()
    store.add(MemoryRecord("harbor 42", MemoryKind.SEMANTIC, .9,
                           fact_predicate="verification_phrase", fact_value="harbor 42"))
    builder = IntentContextBuilder(store, TemporalGoalEngine(), now_fn=lambda: now)
    packet = builder.build("What verification phrase did I ask you to remember?", memory_candidates=[])
    assert "harbor 42" in packet.rendered
    assert resolve_context_intent("What is my pet name?").predicates == frozenset({"pet_name"})


def test_closure_is_evidence_not_missing_context():
    goals = TemporalGoalEngine()
    goal = goals.add_structured_goal("archive pickup")
    goal.completed = True
    goal.metadata["lifecycle"] = "completed"
    builder = IntentContextBuilder(InMemoryMemoryStore(), goals)
    packet = builder.build("Is my archive pickup commitment still open?")
    assert "status=completed" in packet.rendered
    assert not builder.build("What commitment remains open?").items


def test_retracted_preference_is_resolved_from_history():
    store = InMemoryMemoryStore()
    old = store.add(MemoryRecord("I liked pottery", MemoryKind.USER_MODEL, .9,
                                 fact_predicate="likes", fact_value="pottery",
                                 status=MemoryStatus.FORGOTTEN))
    store.add(MemoryRecord("Retraction of pottery", MemoryKind.SEMANTIC, .9,
        fact_predicate="retracted:likes", fact_value="pottery", metadata={
            "retraction_tombstone": True, "retracted_predicate": "likes",
            "resolved_memory_ids": [old.id],
            "retraction_source_text": "Pottery has lost its appeal for me."}))
    builder = IntentContextBuilder(store, TemporalGoalEngine())
    packet = builder.build("What interest has lost its appeal for me?")
    assert "pottery" in packet.rendered
    assert "status=retracted" in packet.rendered
    assert packet.diagnostics["preference_state"]["historical"]


def test_successful_empty_reranker_is_not_overridden():
    class EmptyReranker:
        def rerank(self, query, candidates):
            return SemanticRerankResult(selected_ids=[], success=True)
    store = InMemoryMemoryStore()
    for value in ("pottery", "sailing"):
        store.add(MemoryRecord(f"I like {value}", MemoryKind.USER_MODEL, .9,
                              fact_predicate="likes", fact_value=value))
    builder = IntentContextBuilder(store, TemporalGoalEngine(), reranker=EmptyReranker())
    packet = builder.build("Which musical hobbies do I enjoy?")
    assert not packet.items
    assert packet.diagnostics["semantic_reranker"]["calls"] == 1
