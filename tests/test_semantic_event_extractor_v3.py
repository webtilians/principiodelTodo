import json
from datetime import datetime

from src.infinito3.cognitive_events import CognitiveEventType
from src.infinito3.llm_adapter import RecordingLLMAdapter
from src.infinito3.semantic_event_extractor import SemanticCognitiveEventExtractor


class Clock:
    def __init__(self, value):
        self.value = value

    def __call__(self):
        return self.value


def adapter_for(event):
    return RecordingLLMAdapter(
        responder=lambda request: json.dumps({"events": [event]}, ensure_ascii=False)
    )


def test_semantic_extractor_catches_unseen_bike_revision():
    clock = Clock(datetime(2026, 9, 14, 12, 0, 0))
    adapter = adapter_for({
        "operation": "replace_fact",
        "predicate": "bike",
        "value": "Yeti SB160",
        "previous_value": None,
        "due_at": None,
        "confidence": 0.99,
    })
    extractor = SemanticCognitiveEventExtractor(adapter, now_fn=clock)
    events = extractor.extract("He cambiado otra vez de bici: ahora uso una Yeti SB160.")
    assert len(events) == 1
    assert events[0].type == CognitiveEventType.REPLACE_FACT
    assert events[0].predicate == "bike"
    assert events[0].value == "Yeti SB160"
    assert events[0].metadata["extractor"] == "semantic_v1"


def test_semantic_extractor_catches_no_longer_drink_retraction():
    adapter = adapter_for({
        "operation": "retract_preference",
        "predicate": "likes",
        "value": "kombucha",
        "confidence": 0.98,
    })
    extractor = SemanticCognitiveEventExtractor(adapter)
    events = extractor.extract("I no longer drink kombucha.")
    assert len(events) == 1
    assert events[0].type == CognitiveEventType.RETRACT_PREFERENCE
    assert events[0].predicate == "likes"
    assert events[0].value == "kombucha"


def test_semantic_extractor_catches_goal_without_rule_marker():
    adapter = adapter_for({
        "operation": "create_goal",
        "predicate": "goal",
        "value": "clase de guitarra",
        "due_at": "2026-09-17T19:00:00",
        "confidence": 0.97,
    })
    extractor = SemanticCognitiveEventExtractor(adapter)
    events = extractor.extract("El jueves a las 19 tengo clase de guitarra.")
    assert len(events) == 1
    assert events[0].type == CognitiveEventType.CREATE_GOAL
    assert events[0].value == "clase de guitarra"
    assert events[0].due_at == datetime(2026, 9, 17, 19, 0, 0)


def test_low_confidence_semantic_output_is_discarded():
    adapter = adapter_for({
        "operation": "assert_fact",
        "predicate": "location",
        "value": "Mars",
        "confidence": 0.2,
    })
    extractor = SemanticCognitiveEventExtractor(adapter)
    assert extractor.extract("I had a quiet afternoon.") == []


def test_semantic_usage_is_measurable():
    adapter = adapter_for({
        "operation": "assert_preference",
        "predicate": "likes",
        "value": "bouldering",
        "confidence": 0.95,
    })
    extractor = SemanticCognitiveEventExtractor(adapter)
    events = extractor.extract("I have started enjoying bouldering.")
    assert events[0].type == CognitiveEventType.ASSERT_PREFERENCE
    usage = extractor.usage_snapshot()
    assert usage["calls"] == 1
    assert usage["failures"] == 0
