from datetime import datetime

from src.infinito3.cognitive_events import CognitiveEvent, CognitiveEventType
from src.infinito3.goal_resolver import StateAwareGoalResolver
from src.infinito3.types import Goal


def test_reschedule_uses_old_due_time_and_entity_identity():
    resolver = StateAwareGoalResolver()
    vet = Goal("cita del veterinario", due_at=datetime(2026, 9, 15, 18, 0, 0), metadata={"source_text": "El martes a las 18 tengo cita con el veterinario."})
    guitar = Goal("clase de guitarra", due_at=datetime(2026, 9, 17, 19, 0, 0), metadata={"source_text": "El jueves a las 19 tengo clase de guitarra."})
    insurance = Goal("enviar documentación del seguro", due_at=datetime(2026, 9, 18, 16, 0, 0))
    event = CognitiveEvent(
        CognitiveEventType.RESCHEDULE_GOAL,
        "La cita del veterinario cambia del martes a las 18 al jueves a las 20.",
        predicate="goal",
        value="cita del veterinario",
        due_at=datetime(2026, 9, 17, 20, 0, 0),
        occurred_at=datetime(2026, 9, 14, 12, 0, 0),
        metadata={"previous_due_at": "2026-09-15T18:00:00"},
    )
    assert resolver.resolve(event, [guitar, insurance, vet]) is vet


def test_completion_can_use_due_now_when_text_is_pronoun_heavy():
    resolver = StateAwareGoalResolver()
    due_now = Goal("recoger portátil reparado", due_at=datetime(2026, 9, 14, 12, 0, 0))
    later = Goal("revisión del coche", due_at=datetime(2026, 9, 20, 10, 0, 0))
    event = CognitiveEvent(
        CognitiveEventType.COMPLETE_GOAL,
        "Ya está hecho; márcalo como completado.",
        predicate="goal",
        value="hecho",
        occurred_at=datetime(2026, 9, 14, 13, 0, 0),
    )
    assert resolver.resolve(event, [later, due_now]) is due_now
