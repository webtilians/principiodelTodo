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
        memory_candidates=[user, relevant],
        recent_turns=[ConversationTurn(role="user", content="Hoy hice cuatro bajadas")],
        max_tokens=500,
    )

    sources = {item.source for item in packet.items}
    assert ContextSource.GOAL in sources
    assert ContextSource.USER_MODEL in sources
    assert ContextSource.MEMORY in sources
    assert ContextSource.RECENT in sources


def test_core_user_fact_fallback_is_query_gated():
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

    relevant_packet = builder.build("¿Cómo me llamo?", memory_candidates=[], max_tokens=300)
    unrelated_packet = builder.build("Explícame esta arquitectura", memory_candidates=[], max_tokens=300)

    assert any(item.memory_id == identity.id for item in relevant_packet.items)
    assert all(item.memory_id != identity.id for item in unrelated_packet.items)


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


def test_singular_query_prunes_weaker_user_model_noise():
    store = InMemoryMemoryStore()
    goals = SimpleGoalEngine()
    bike = _memory(
        "Me gusta mi bici Specialized Demo",
        kind=MemoryKind.USER_MODEL,
        importance=0.8,
        confidence=1.0,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="mi bici specialized demo",
    )
    coffee = _memory(
        "Me gusta el café",
        kind=MemoryKind.USER_MODEL,
        importance=0.8,
        confidence=1.0,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="café",
    )
    movies = _memory(
        "Prefiero las películas de ciencia ficción",
        kind=MemoryKind.USER_MODEL,
        importance=0.8,
        confidence=1.0,
        fact_subject="user",
        fact_predicate="prefers",
        fact_value="películas de ciencia ficción",
    )
    builder = BalancedContextBuilder(store, goals)

    packet = builder.build(
        "¿Qué bici uso?",
        memory_candidates=[bike, coffee, movies],
        max_tokens=300,
    )

    ids = {item.memory_id for item in packet.items}
    assert bike.id in ids
    assert coffee.id not in ids
    assert movies.id not in ids
    assert packet.diagnostics["precision_dropped"] == 2


def test_plural_query_keeps_multiple_values_of_same_predicate():
    store = InMemoryMemoryStore()
    goals = SimpleGoalEngine()
    jazz = _memory(
        "Me gusta el jazz",
        kind=MemoryKind.USER_MODEL,
        importance=0.8,
        confidence=1.0,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="jazz",
    )
    punk = _memory(
        "Me gusta el punk",
        kind=MemoryKind.USER_MODEL,
        importance=0.8,
        confidence=1.0,
        fact_subject="user",
        fact_predicate="likes",
        fact_value="punk",
    )
    train = _memory(
        "Prefiero viajar en tren",
        kind=MemoryKind.USER_MODEL,
        importance=0.8,
        confidence=1.0,
        fact_subject="user",
        fact_predicate="prefers",
        fact_value="viajar en tren",
    )
    builder = BalancedContextBuilder(store, goals)

    packet = builder.build(
        "¿Qué estilos de música me gustan?",
        memory_candidates=[jazz, punk, train],
        max_tokens=300,
    )

    ids = {item.memory_id for item in packet.items}
    assert jazz.id in ids
    assert punk.id in ids
    assert train.id not in ids


def test_goal_duplicate_memory_is_suppressed():
    fixed_now = datetime(2026, 9, 14, 9, 0)
    goals = SimpleGoalEngine(now_fn=lambda: fixed_now)
    goals.ingest("Mañana tengo que llamar al dentista a las 09:30")
    duplicate = _memory("Mañana tengo que llamar al dentista a las 09:30")
    builder = BalancedContextBuilder(InMemoryMemoryStore(), goals, now_fn=lambda: fixed_now)

    packet = builder.build(
        "¿Qué tengo que hacer mañana?",
        memory_candidates=[duplicate],
        max_tokens=300,
    )

    assert any(item.source == ContextSource.GOAL for item in packet.items)
    assert all(item.memory_id != duplicate.id for item in packet.items)
    assert packet.diagnostics["precision_dropped"] == 1


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
