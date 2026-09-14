import json
from datetime import datetime

from src.infinito3.cognitive_events import CognitiveEventType
from src.infinito3.llm_adapter import RecordingLLMAdapter
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.semantic_event_extractor import SemanticCognitiveEventExtractor
from src.infinito3.temporal import parse_weekday_range
from src.infinito3.temporal_context import TemporalSemanticContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.types import MemoryKind, MemoryRecord, MemoryStatus


class Clock:
    def __init__(self, value): self.value = value
    def __call__(self): return self.value


def add_fact(store, predicate, value, content=None, status=MemoryStatus.ACTIVE, supersedes_id=None):
    return store.add(MemoryRecord(
        content=content or f"{predicate}: {value}", kind=MemoryKind.USER_MODEL,
        importance=0.95, fact_subject="user", fact_predicate=predicate,
        fact_value=value, status=status, supersedes_id=supersedes_id,
    ))


def test_structured_profile_history_and_goal_range():
    now = datetime(2026, 10, 5, 8, 0)
    assert parse_weekday_range("from Wednesday through Sunday", now) == (
        datetime(2026, 10, 7).date(), datetime(2026, 10, 11).date())

    store = InMemoryMemoryStore()
    old = add_fact(store, "location", "Ghent", "I live in Ghent.", MemoryStatus.SUPERSEDED)
    add_fact(store, "location", "Malmö", "I live in Malmö.", supersedes_id=old.id)
    add_fact(store, "bike", "Norco Sight", "My bike is Norco Sight.")
    add_fact(store, "studying_language", "Danish", "I am studying Danish.")
    add_fact(store, "occupation", "engineer", "I work as engineer.")
    goals = TemporalGoalEngine(now_fn=Clock(now))
    goals.add_structured_goal("physiotherapy", due_at=datetime(2026, 10, 7, 8, 30))
    goals.add_structured_goal("Sara", due_at=datetime(2026, 10, 11, 18, 0))
    builder = TemporalSemanticContextBuilder(store, goals, now_fn=Clock(now))

    current = builder.build("What are my current city, bike, language and job?", memory_candidates=[])
    for value in ("Malmö", "Norco Sight", "Danish", "engineer"):
        assert value in current.rendered

    historical = builder.build("Where did I live immediately before Malmö?", memory_candidates=[])
    assert "relation=immediately_previous" in historical.rendered
    assert "Ghent" in historical.rendered
    assert "Malmö" not in historical.rendered

    agenda = builder.build("What commitments do I have from Wednesday through Sunday?", memory_candidates=[])
    assert "physiotherapy" in agenda.rendered and "Sara" in agenda.rendered


def test_semantic_gate_and_ambiguous_language_replacement_review():
    responses = []
    def responder(request):
        responses.append(request)
        return json.dumps({"events":[{
            "operation":"replace_fact", "predicate":"studying_language", "value":"noruego",
            "previous_value":"danés", "confidence":0.99,
            "evidence":"He dejado el danés y ahora estoy aprendiendo noruego", "exclusive":True,
        }]}, ensure_ascii=False)

    adapter = RecordingLLMAdapter(responder=responder)
    extractor = SemanticCognitiveEventExtractor(adapter, now_fn=Clock(datetime(2026, 9, 14, 9, 0)))
    assert extractor.extract("Dime solo 17 al cuadrado.") == []
    assert len(adapter.requests) == 0

    events = extractor.extract("He dejado el danés y ahora estoy aprendiendo noruego.")
    assert len(adapter.requests) == 1
    assert len(events) == 1
    assert events[0].type == CognitiveEventType.REPLACE_FACT
    assert events[0].predicate == "studying_language"
    assert events[0].value == "noruego"
