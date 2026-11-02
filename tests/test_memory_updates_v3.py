"""Stated decisions and their later updates must reach the model, oldest first.

E10-B and E10-D (real LLM) found four ways they did not: the write gate dropped
statements without a known marker, the word "plan" was read as a goal request
and emptied the context, singular-question pruning kept the old value but not
its update, and "a partir de hoy" turned an update into a task due today.
"""
import re

from src.infinito3.context_builder import BalancedContextBuilder
from src.infinito3.engine import CognitiveEngine
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.memory import RuleBasedMemoryGate
from src.infinito3.persistent_memory import SimpleFactExtractor, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord

NOISE = (
    "Los proveedores han contestado a las preguntas del equipo.",
    "Se modificó una presentación para mejorar su claridad.",
    "Las cifras del año anterior siguen siendo provisionales.",
)


def _context(engine, question):
    return engine.process(question, top_k=8).context_packet.rendered


def test_plain_statement_is_stored_as_episodic_memory():
    decision = RuleBasedMemoryGate().evaluate(
        "Para el procedimiento Cobalto-323 hemos acordado utilizar Panel C como herramienta."
    )

    assert decision.should_store
    assert decision.kind == MemoryKind.EPISODIC
    assert "statement" in decision.reasons


def test_questions_and_acknowledgements_are_not_stored():
    gate = RuleBasedMemoryGate()

    assert not gate.evaluate("¿Qué herramienta acordamos para el procedimiento?").should_store
    assert not gate.evaluate("Vale, gracias, perfecto.").should_store


def test_spanish_plan_is_not_a_goal_request():
    asks = BalancedContextBuilder._asks_for_goals

    assert not asks("¿Con qué frecuencia se revisa el plan Orbe-123?")
    assert asks("What are my plans for tomorrow?")
    assert asks("What is my plan for today?")


def test_update_reaches_context_after_the_original():
    engine = CognitiveEngine.persistent(":memory:")
    try:
        engine.process("Para el procedimiento Cobalto-323 hemos acordado utilizar Panel C como herramienta.")
        for line in NOISE:
            engine.process(line)
        engine.process(
            "Al revisar el procedimiento Cobalto-323 cambiamos la herramienta elegida "
            "y desde ahora usaremos Visor A."
        )

        context = _context(engine, "¿Qué herramienta usamos en el procedimiento Cobalto-323?")

        assert context.index("Panel C") < context.index("Visor A")
        assert re.search(r'- memory \(\d{4}-\d{2}-\d{2}\): "Para el procedimiento', context)
        assert "proveedores" not in context
    finally:
        engine.memory_store.close()


def test_announced_change_is_not_turned_into_a_task():
    engine = CognitiveEngine.persistent(":memory:")
    try:
        engine.process("El resumen operativo Cedro-436 se prepara para el equipo de producto.")
        engine.process(
            "Se ha actualizado el destino del resumen Cedro-436: a partir de hoy "
            "se envía al equipo jurídico, no al anterior."
        )

        context = _context(engine, "¿A quién enviamos el resumen Cedro-436?")

        assert context.index("equipo de producto") < context.index("equipo jurídico")
        assert "[ACTIVE GOALS]" not in context
    finally:
        engine.memory_store.close()


def test_goal_question_keeps_profile_memories_it_mentions():
    engine = CognitiveEngine.persistent(":memory:")
    try:
        engine.process("Prefiero revisar el plan Orbe-123 los viernes.")
        engine.process("Me gusta el café solo.")

        context = _context(engine, "¿Qué tareas tengo pendientes con el plan Orbe-123?")

        assert "los viernes" in context
        assert "café" not in context
    finally:
        engine.memory_store.close()


def test_only_real_due_dates_create_goals():
    goals = SimpleGoalEngine()

    assert goals.ingest("A partir de hoy los avisos se envían por SMS.") == []
    assert goals.ingest("Esta mañana llovió bastante en el centro.") == []
    assert len(goals.ingest("Hoy tengo que llamar al banco.")) == 1


def test_fact_value_stops_at_the_next_clause():
    extractor = SimpleFactExtractor()

    name = extractor.extract("Me llamo Ana y vivo en Madrid.", MemoryKind.USER_MODEL)
    assert (name.predicate, name.value) == ("name", "ana")
    assert extractor.extract("Vivo en Castilla y León.", MemoryKind.USER_MODEL).value == "castilla y león"
    assert extractor.extract("Vivo en Madrid desde 2010.", MemoryKind.USER_MODEL).value == "madrid"


def test_write_times_are_strictly_increasing_across_restarts(tmp_path):
    path = str(tmp_path / "memory.db")
    store = SQLiteCognitiveMemoryStore(path)
    first = store.add(MemoryRecord("Usamos Panel C.", MemoryKind.EPISODIC, 0.3))
    second = store.add(MemoryRecord("Ahora usamos Visor A.", MemoryKind.EPISODIC, 0.3))
    store.close()

    reopened = SQLiteCognitiveMemoryStore(path)
    third = reopened.add(MemoryRecord("Volvemos a Panel C.", MemoryKind.EPISODIC, 0.3))
    reopened.close()

    assert first.updated_at < second.updated_at < third.updated_at
