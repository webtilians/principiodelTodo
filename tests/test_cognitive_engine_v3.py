from datetime import datetime

from src.infinito3.engine import CognitiveEngine
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.memory import InMemoryMemoryStore, RuleBasedMemoryGate
from src.infinito3.safety import SensitiveInformationFilter
from src.infinito3.types import MemoryKind, SafetyLevel


def test_trivial_message_is_not_stored():
    store = InMemoryMemoryStore()
    engine = CognitiveEngine(memory_store=store)

    decision = engine.process("hola")

    assert decision.stored is False
    assert store.all() == []


def test_user_preference_is_stored_as_user_model():
    store = InMemoryMemoryStore()
    engine = CognitiveEngine(memory_store=store)

    decision = engine.process("Me gusta entrenar descenso en bicicleta")

    assert decision.stored is True
    assert len(store.all()) == 1
    assert store.all()[0].kind == MemoryKind.USER_MODEL


def test_password_is_blocked_before_gate_and_storage():
    store = InMemoryMemoryStore()
    engine = CognitiveEngine(memory_store=store)

    decision = engine.process("mi contraseña es superSecreta123")

    assert decision.safety.level == SafetyLevel.FORBIDDEN
    assert decision.gate is None
    assert decision.stored is False
    assert store.all() == []


def test_sensitive_email_is_not_persisted_by_default():
    store = InMemoryMemoryStore()
    engine = CognitiveEngine(memory_store=store)

    decision = engine.process("mi email es persona@example.com")

    assert decision.safety.level == SafetyLevel.SENSITIVE
    assert decision.stored is False
    assert store.all() == []


def test_retrieval_happens_before_current_message_is_written():
    store = InMemoryMemoryStore()
    engine = CognitiveEngine(memory_store=store)

    engine.process("Me gusta el descenso en bicicleta")
    decision = engine.process("Me encanta el descenso en bicicleta")

    assert decision.stored is True
    assert len(decision.context) == 1
    assert decision.context[0].content == "Me gusta el descenso en bicicleta"


def test_pasado_manana_is_two_days_not_one():
    fixed_now = datetime(2026, 9, 14, 9, 0)
    goals = SimpleGoalEngine(now_fn=lambda: fixed_now)

    created = goals.ingest("Pasado mañana tengo cita a las 10:30")

    assert len(created) == 1
    assert created[0].due_at == datetime(2026, 9, 16, 10, 30)


def test_memory_gate_is_a_replaceable_baseline():
    gate = RuleBasedMemoryGate()

    decision = gate.evaluate("Mi nombre es Enrique")

    assert decision.should_store is True
    assert decision.kind == MemoryKind.USER_MODEL
    assert "user_identity" in decision.reasons


def test_api_key_is_never_store():
    safety = SensitiveInformationFilter()

    decision = safety.inspect("usa sk-abcdefghijklmnop123456 para la API")

    assert decision.level == SafetyLevel.FORBIDDEN
    assert decision.reason == "api_key"
    assert "[REDACTED]" in decision.redacted_text
