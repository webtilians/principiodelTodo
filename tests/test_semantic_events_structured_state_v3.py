from datetime import datetime

from src.infinito3.cognitive_events import CognitiveEvent, CognitiveEventType
from src.infinito3.event_extractor import TemporalCognitiveEventExtractor
from src.infinito3.llm_adapter import RecordingLLMAdapter
from src.infinito3.persistent_memory import HashEmbeddingProvider
from src.infinito3.semantic_goal_state import SemanticGoalTemporalState
from src.infinito3.semantic_interpreter import SemanticCognitiveEventExtractor, SemanticStateQueryAnalyzer
from src.infinito3.semantic_temporal_memory import SemanticTemporalMemoryStore
from src.infinito3.structured_temporal_context import StructuredTemporalContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import Goal


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def _last_user(request):
    return request.messages[-1].content


def test_semantic_event_extractor_covers_unseen_surface_forms():
    clock = Clock(datetime(2026, 9, 14, 10, 0, 0))

    def responder(request):
        text = _last_user(request)
        if "Yeti SB160" in text:
            return '{"events":[{"type":"replace_fact","predicate":"bike","value":"Yeti SB160","confidence":0.99,"exclusive":true}]}'
        if "kombucha" in text:
            return '{"events":[{"type":"retract_preference","predicate":"likes","value":"kombucha","confidence":0.99}]}'
        if "bouldering" in text:
            return '{"events":[{"type":"assert_preference","predicate":"likes","value":"bouldering","confidence":0.97}]}'
        if "Nube" in text:
            return '{"events":[{"type":"assert_fact","predicate":"pet_name","value":"Nube","confidence":0.99,"exclusive":true}]}'
        if "guitarra" in text:
            return '{"events":[{"type":"create_goal","predicate":"goal","value":"clase de guitarra","due_at":"2026-09-15T18:00:00","confidence":0.96}]}'
        return '{"events":[]}'

    extractor = SemanticCognitiveEventExtractor(
        RecordingLLMAdapter(responder),
        now_fn=clock,
        fallback=TemporalCognitiveEventExtractor(now_fn=clock),
    )

    cases = [
        ("He cambiado otra vez de bici: ahora uso una Yeti SB160.", CognitiveEventType.REPLACE_FACT, "bike", "Yeti SB160"),
        ("I no longer drink kombucha.", CognitiveEventType.RETRACT_PREFERENCE, "likes", "kombucha"),
        ("I have started enjoying bouldering.", CognitiveEventType.ASSERT_PREFERENCE, "likes", "bouldering"),
        ("Mi gata se llama Nube.", CognitiveEventType.ASSERT_FACT, "pet_name", "Nube"),
        ("Mañana a las 18 tengo clase de guitarra.", CognitiveEventType.CREATE_GOAL, "goal", "clase de guitarra"),
    ]
    for text, event_type, predicate, value in cases:
        events = extractor.extract(text)
        assert any(e.type == event_type and e.predicate == predicate and e.value == value for e in events)


def test_semantic_extractor_does_not_call_model_for_ordinary_questions():
    adapter = RecordingLLMAdapter(lambda request: '{"events":[{"type":"assert_fact","predicate":"bad","value":"bad"}]}')
    extractor = SemanticCognitiveEventExtractor(adapter)
    assert extractor.extract("¿Cuál es la capital de Japón?") == []
    assert adapter.requests == []


def test_state_query_analyzer_uses_closed_slot_plan():
    adapter = RecordingLLMAdapter(
        lambda request: '{"predicates":["name","location","bike","studying_language"],'
        '"history":[{"predicate":"location","before_value":"Utrecht"}],'
        '"asks_goals":false,"confidence":0.98}'
    )
    analyzer = SemanticStateQueryAnalyzer(adapter)
    plan = analyzer.analyze("What are my name, city and bike, and where did I live before Utrecht?")
    assert set(plan.predicates) == {"name", "location", "bike", "studying_language"}
    assert plan.history[0].predicate == "location"
    assert plan.history[0].before_value == "Utrecht"
    assert plan.confidence == 0.98


def test_structured_context_bypasses_vector_top_k_for_requested_slots_and_history():
    clock = Clock(datetime(2026, 9, 14, 10, 0, 0))
    store = SemanticTemporalMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    goals = TemporalGoalEngine(now_fn=clock)
    state = SemanticGoalTemporalState(now_fn=clock)
    fallback = TemporalCognitiveEventExtractor(now_fn=clock)

    state.apply(fallback.extract("Me llamo Irene."), memory_store=store, goal_engine=goals)
    state.apply(fallback.extract("Vivo en Lyon."), memory_store=store, goal_engine=goals)
    state.apply(fallback.extract("I moved to Utrecht. Utrecht is where I live now."), memory_store=store, goal_engine=goals)
    state.apply(fallback.extract("Mi bici principal es una Trek Slash."), memory_store=store, goal_engine=goals)

    def responder(request):
        query = _last_user(request)
        if "before Utrecht" in query:
            return '{"predicates":[],"history":[{"predicate":"location","before_value":"Utrecht"}],"asks_goals":false,"confidence":0.99}'
        return '{"predicates":["name","location","bike"],"history":[],"asks_goals":false,"confidence":0.99}'

    analyzer = SemanticStateQueryAnalyzer(RecordingLLMAdapter(responder))
    builder = StructuredTemporalContextBuilder(
        memory_store=store,
        goal_engine=goals,
        temporal_state=state,
        query_analyzer=analyzer,
        now_fn=clock,
    )

    current = builder.build(
        "What are my current name, city and bike?",
        memory_candidates=[],
        max_tokens=500,
    )
    assert "Irene" in current.rendered
    assert "Utrecht" in current.rendered
    assert "Trek Slash" in current.rendered

    historical = builder.build(
        "Where did I live immediately before Utrecht?",
        memory_candidates=[],
        max_tokens=400,
    )
    assert "value=lyon" in historical.rendered.lower()
    assert "relation=immediately_previous" in historical.rendered
    assert "value=utrecht" not in historical.rendered.lower()


def test_goal_resolution_uses_previous_due_time_plus_target_identity():
    clock = Clock(datetime(2026, 10, 5, 8, 0, 0))
    state = SemanticGoalTemporalState(now_fn=clock)
    old_due = datetime(2026, 10, 6, 18, 0, 0)
    target = Goal("veterinarian appointment", due_at=old_due)
    distractor = Goal("call the bank", due_at=datetime(2026, 10, 6, 18, 0, 0))
    event = CognitiveEvent(
        CognitiveEventType.RESCHEDULE_GOAL,
        "The veterinarian appointment moved from Tuesday 18:00 to Thursday 20:00.",
        predicate="goal",
        value="veterinarian appointment",
        due_at=datetime(2026, 10, 8, 20, 0, 0),
        occurred_at=clock(),
        metadata={"previous_due_at": old_due.isoformat()},
    )
    store = SemanticTemporalMemoryStore(":memory:", embedding_provider=HashEmbeddingProvider())
    resolved = state._match_goal(event, [distractor, target], memory_store=store)
    assert resolved is target
