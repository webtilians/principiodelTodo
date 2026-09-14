from datetime import datetime

from src.infinito3.generalized_context_builder import GeneralizedContextBuilder
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord


class Clock:
    def __init__(self, current):
        self.current = current

    def __call__(self):
        return self.current


def test_weekday_goals_share_calendar_semantics_with_weekend_query():
    clock = Clock(datetime(2026, 9, 14, 9, 0))  # Monday
    goals = SimpleGoalEngine(now_fn=clock)
    friday = goals.ingest("El viernes tengo cita con el fisioterapeuta a las 12.")[0]
    saturday = goals.ingest("El sábado tengo reunión con Ana a las 10.")[0]

    assert friday.due_at == datetime(2026, 9, 18, 12, 0)
    assert saturday.due_at == datetime(2026, 9, 19, 10, 0)

    builder = GeneralizedContextBuilder(InMemoryMemoryStore(), goals, now_fn=clock)
    packet = builder.build(
        "¿Qué compromisos tengo este fin de semana?",
        memory_candidates=[],
    )

    assert "fisioterapeuta" in packet.rendered
    assert "Ana" in packet.rendered
    assert "due=2026-09-18T12:00:00" in packet.rendered
    assert "due=2026-09-19T10:00:00" in packet.rendered


def test_explicit_spanish_date_is_scheduled_and_retrievable():
    clock = Clock(datetime(2026, 9, 17, 9, 0))
    goals = SimpleGoalEngine(now_fn=clock)
    goal = goals.ingest("El 25 de septiembre tengo que renovar el seguro a las 9.")[0]

    assert goal.due_at == datetime(2026, 9, 25, 9, 0)

    builder = GeneralizedContextBuilder(InMemoryMemoryStore(), goals, now_fn=clock)
    packet = builder.build(
        "¿Qué tengo programado para el 25 de septiembre?",
        memory_candidates=[],
    )
    assert "seguro" in packet.rendered
    assert "due=2026-09-25T09:00:00" in packet.rendered


def test_cancelled_goal_cannot_leak_back_from_episodic_memory():
    clock = Clock(datetime(2026, 9, 14, 9, 0))
    goals = SimpleGoalEngine(now_fn=clock)
    goals.ingest("El viernes tengo cita con el fisioterapeuta a las 12.")
    goals.ingest("El sábado tengo reunión con Ana a las 10.")
    goals.ingest("Cancela la cita con el fisioterapeuta del viernes; ya no la tengo.")

    stale = MemoryRecord(
        content="El viernes tengo cita con el fisioterapeuta a las 12.",
        kind=MemoryKind.EPISODIC,
        importance=0.7,
    )
    builder = GeneralizedContextBuilder(InMemoryMemoryStore(), goals, now_fn=clock)
    packet = builder.build(
        "¿Qué compromiso me queda para el sábado?",
        memory_candidates=[stale],
    )

    assert "Ana" in packet.rendered
    assert "fisioterapeuta" not in packet.rendered
