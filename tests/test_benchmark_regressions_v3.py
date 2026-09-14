from datetime import datetime

from src.infinito3 import RuleBasedMemoryGate
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.types import MemoryKind


def test_gate_stores_direct_bike_fact_as_user_model():
    decision = RuleBasedMemoryGate().evaluate("Mi bici es una Santa Cruz V10.")
    assert decision.should_store is True
    assert decision.kind == MemoryKind.USER_MODEL
    assert "structured_user_fact" in decision.reasons


def test_gate_stores_favorite_color_as_user_model():
    decision = RuleBasedMemoryGate().evaluate("Mi color favorito es azul petróleo.")
    assert decision.should_store is True
    assert decision.kind == MemoryKind.USER_MODEL
    assert "structured_user_fact" in decision.reasons


def test_gate_stores_age_as_user_model():
    decision = RuleBasedMemoryGate().evaluate("Tengo 45 años.")
    assert decision.should_store is True
    assert decision.kind == MemoryKind.USER_MODEL


def test_interrogative_future_probe_does_not_create_goal():
    engine = SimpleGoalEngine(now_fn=lambda: datetime(2026, 9, 14, 13, 0))
    assert engine.ingest("¿Qué tengo que hacer mañana?") == []
    assert engine.all() == []


def test_interrogative_day_after_tomorrow_probe_does_not_create_goal():
    engine = SimpleGoalEngine(now_fn=lambda: datetime(2026, 9, 14, 13, 0))
    assert engine.ingest("¿Qué tengo pendiente pasado mañana?") == []
    assert engine.all() == []


def test_explicit_reminder_question_form_can_still_create_goal():
    engine = SimpleGoalEngine(now_fn=lambda: datetime(2026, 9, 14, 13, 0))
    created = engine.ingest("¿Puedes recordarme mañana llamar al banco?")
    # This form is not an information-seeking interrogative prefix, so the
    # conservative parser still allows it to express an intention.
    assert len(created) == 1
