from datetime import datetime, timedelta

from src.infinito3.context_builder import ApproximateTokenEstimator, BalancedContextBuilder
from src.infinito3.engine import CognitiveEngine
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.types import (
    ContextSource,
    ConversationTurn,
    Goal,
    MemoryKind,
    MemoryRecord,
)


def _memory(content, kind=MemoryKind.EPISODIC, importance=0.7, confidence=0.9, **kwargs):
    return MemoryRecord(
        content=content,
        kind=kind,
        importance=importance,
        confidence=confidence,
        **kwargs,
    )


def test_context_builder_respects_hard_budget():
    store = InMemoryMemoryStore()
    goals = SimpleGoalEngine()
    for index in range(20):
        store.add(
            _memory(
                f"Recuerdo largo sobre descenso bicicleta entrenamiento número {index} "
                "con bastante información adicional para ocupar presupuesto"
            )
        )
    builder = BalancedContextBuilder(store, goals)

    packet = builder.build("descenso bicicleta", max_tokens=120)

    assert packet.estimated_tokens <= 120
    assert packet.budget_tokens == 120
    assert packet.dropped_count > 0


def test_context_balances_goal_user_model_memory_and_recent():
    store = InMemoryMemoryStore()
    fixed_now = datetime(2026, 9, 14, 10, 0)
    goals = SimpleGoalEngine(now_fn=lambda: fixed_now)

    user = _memory(
        "Vivo en Málaga",
        kind=MemoryKind.USER_MODEL,
        importance=0.9,
        fact_subject="user",
        fact_predicate="location",
        fact_value="málaga",
    )
    relevant = _memory("Entreno descenso en bicicleta cuatro días por semana", importance=0.85)
    store.add(user)
    store.add(relevant)
    goals.ingest("Mañana tengo que entrenar a las 10")

    builder = BalancedContextBuilder(store, goals, now_fn=lambda: fixed_now)
    packet = builder.build(
        "preparar entrenamiento de descenso",
        memory_candidates=[relevant],
        recent_turns=[ConversationTurn(role="user", content="Hoy hice cuatro bajadas")],
        max_tokens=500,
    )

    sources = {item.source for item in packet.items}
    assert ContextSource.GOAL in sources
    assert ContextSource.USER_MODEL in sources
    assert ContextSource.MEMORY in sources
    assert ContextSource.RECENT in sources


def test_core_user_fact_can_be_included_even_when_retrieval_misses_it():
    store = InMemoryMemoryStore()
    goals = SimpleGoalEngine()
    identity = _memory(
        "Me llamo Enrique",
        kind=MemoryKind.USER_MODEL,
        importance=0.9,
        confidence=0.95,
        fact_subject="user",
        fact_predicate="name",
        fact_value="enrique",
    )
    store.add(identity)
    builder = BalancedContextBuilder(store, goals)

    packet = builder.build("Explícame esta arquitectura", memory_candidates=[], max_tokens=300)

    assert any(item.memory_id == identity.id for item in packet.items)


def test_unrelated_preference_is_not_forced_into_every_context():
    store = InMemoryMemoryStore()
    goals = SimpleGoalEngine()
    preference = _memory(
        "Me gusta el café",
        kind=MemoryKind.USER_MODEL,
        importance=0.8,
        confidence=0.95,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="café",
    )
    store.add(preference)
    builder = BalancedContextBuilder(store, goals)

    packet = builder.build("arquitectura de software", memory_candidates=[], max_tokens=300)

    assert all(item.memory_id != preference.id for item in packet.items)


def test_memory_is_rendered_as_untrusted_quoted_data():
    store = InMemoryMemoryStore()
    goals = SimpleGoalEngine()
    quoted_memory = _memory('Texto recordado con "comillas"\nsegunda línea')
    store.add(quoted_memory)
    builder = BalancedContextBuilder(store, goals)

    packet = builder.build(
        "Texto recordado",
        memory_candidates=[quoted_memory],
        max_tokens=300,
    )

    assert "untrusted evidence" in packet.rendered
    assert '\\n' not in packet.rendered
    assert '"Texto recordado con \\"comillas\\" segunda línea"' in packet.rendered


def test_due_goal_receives_high_priority():
    fixed_now = datetime(2026, 9, 14, 10, 0)
    goals = SimpleGoalEngine(now_fn=lambda: fixed_now)
    goals._goals.append(Goal(description="Entregar informe", due_at=fixed_now - timedelta(minutes=5)))
    goals._goals.append(Goal(description="Comprar ruedas", due_at=fixed_now + timedelta(days=30)))
    builder = BalancedContextBuilder(
        InMemoryMemoryStore(),
        goals,
        now_fn=lambda: fixed_now,
    )

    packet = builder.build("qué tengo pendiente", max_tokens=300)
    goal_items = [item for item in packet.items if item.source == ContextSource.GOAL]

    assert len(goal_items) == 2
    assert goal_items[0].content.startswith("Entregar informe")
    assert goal_items[0].score > goal_items[1].score


def test_engine_exposes_context_packet_without_self_retrieval():
    store = InMemoryMemoryStore()
    engine = CognitiveEngine(memory_store=store)

    decision = engine.process(
        "Me gusta entrenar descenso en bicicleta",
        context_budget_tokens=300,
    )

    assert decision.stored is True
    assert decision.context == []
    assert decision.context_packet is not None
    assert all(item.memory_id != decision.stored_memory_id for item in decision.context_packet.items)


def test_new_goal_is_available_to_context_same_turn():
    fixed_now = datetime(2026, 9, 14, 9, 0)
    goals = SimpleGoalEngine(now_fn=lambda: fixed_now)
    builder = BalancedContextBuilder(InMemoryMemoryStore(), goals, now_fn=lambda: fixed_now)
    engine = CognitiveEngine(goal_engine=goals, context_builder=builder)

    decision = engine.process(
        "Mañana tengo que entrenar a las 10",
        context_budget_tokens=300,
    )

    assert decision.created_goal_ids
    assert decision.context_packet is not None
    assert any(
        item.goal_id in decision.created_goal_ids
        for item in decision.context_packet.items
        if item.source == ContextSource.GOAL
    )


def test_custom_token_estimator_can_be_injected():
    class CharacterEstimator:
        def estimate(self, text):
            return len(text)

    builder = BalancedContextBuilder(
        InMemoryMemoryStore(),
        SimpleGoalEngine(),
        token_estimator=CharacterEstimator(),
    )

    packet = builder.build("hola", max_tokens=80)

    assert packet.estimated_tokens <= 80


def test_default_estimator_is_deterministic():
    estimator = ApproximateTokenEstimator()
    text = "INFINITO recuerda hechos y objetivos."

    assert estimator.estimate(text) == estimator.estimate(text)
    assert estimator.estimate(text) > 0
