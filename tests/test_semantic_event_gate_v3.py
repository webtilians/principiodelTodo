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


def test_non_mutating_imperative_requests_do_not_call_semantic_extractor():
    adapter = RecordingLLMAdapter(responder=lambda request: '{"events":[]}')
    extractor = SemanticCognitiveEventExtractor(adapter, now_fn=Clock(datetime(2026, 9, 14, 9, 0)))

    assert extractor.extract("Dime solo 17 al cuadrado.") == []
    assert extractor.extract("Explica la refracción en una frase.") == []
    assert extractor.extract("Tell me a short fact about Saturn.") == []
    assert adapter.requests == []
    assert extractor.stats()["skipped_non_mutating_requests"] == 3


def test_ambiguous_deterministic_retraction_is_semantically_reviewed_as_replacement():
    payload = {
        "events": [
            {
                "operation": "replace_fact",
                "predicate": "studying_language",
                "value": "noruego",
                "previous_value": "danés",
                "due_at": None,
                "confidence": 0.99,
                "evidence": "He dejado el danés y ahora estoy aprendiendo noruego",
                "exclusive": True,
            }
        ]
    }
    adapter = RecordingLLMAdapter(responder=lambda request: json.dumps(payload, ensure_ascii=False))
    extractor = SemanticCognitiveEventExtractor(adapter, now_fn=Clock(datetime(2026, 9, 14, 9, 0)))

    events = extractor.extract("He dejado el danés y ahora estoy aprendiendo noruego.")

    assert len(adapter.requests) == 1
    assert len(events) == 1
    assert events[0].type == CognitiveEventType.REPLACE_FACT
    assert events[0].predicate == "studying_language"
    assert events[0].value == "noruego"
    assert extractor.stats()["semantic_reviews"] == 1


def test_preferred_form_of_address_can_be_grounded_as_name_fact():
    payload = {
        "events": [
            {
                "operation": "assert_fact",
                "predicate": "name",
                "value": "Teo",
                "confidence": 0.98,
                "evidence": "prefiero que me llames Teo",
                "exclusive": True,
            }
        ]
    }
    adapter = RecordingLLMAdapter(responder=lambda request: json.dumps(payload, ensure_ascii=False))
    extractor = SemanticCognitiveEventExtractor(adapter, now_fn=Clock(datetime(2026, 9, 14, 9, 0)))

    events = extractor.extract("Entre amigos prefiero que me llames Teo.")

    assert len(events) == 1
    assert events[0].predicate == "name"
    assert events[0].metadata["exclusive"] is True
