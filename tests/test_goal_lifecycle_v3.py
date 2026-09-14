from src.infinito3.generalized_context_builder import GeneralizedContextBuilder
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.trajectory_evaluation import MutableClock, TrajectoryScenario


def _clock():
    return MutableClock(TrajectoryScenario(name="clock", steps=()).start_at)


def test_temporal_reference_without_intention_does_not_create_goal():
    goals = SimpleGoalEngine(now_fn=_clock())

    created = goals.ingest("Hoy he leído un artículo sobre motores eléctricos.")

    assert created == []
    assert goals.all() == []


def test_explicit_completion_marks_matching_goal_completed():
    clock = _clock()
    goals = SimpleGoalEngine(now_fn=clock)
    created = goals.ingest("Mañana tengo que recoger un paquete a las 12.")
    assert len(created) == 1

    goals.ingest("Ya recogí el paquete; deja de considerarlo pendiente.")

    assert created[0].completed is True
    assert created[0].metadata["lifecycle"] == "completed"


def test_completion_only_changes_the_matching_open_goal():
    clock = _clock()
    goals = SimpleGoalEngine(now_fn=clock)
    bank = goals.ingest("Mañana tengo que llamar al banco a las 10.")[0]
    package = goals.ingest("Pasado mañana tengo que recoger un paquete a las 12.")[0]

    goals.ingest("Ya recogí el paquete; deja de considerarlo pendiente.")

    assert package.completed is True
    assert bank.completed is False


def test_future_remaining_state_query_is_inferred_as_goal_intent_without_noun_dictionary():
    clock = _clock()
    goals = SimpleGoalEngine(now_fn=clock)
    goals.ingest("Mañana tengo reunión con Laura a las 11.")
    builder = GeneralizedContextBuilder(InMemoryMemoryStore(), goals, now_fn=clock)

    packet = builder.build(
        "Hoy es 14 de septiembre. ¿Qué compromiso futuro me queda?",
        memory_candidates=[],
    )

    assert "Laura" in packet.rendered
    assert any(item.source.value == "goal" for item in packet.items)


def test_overdue_goal_is_not_silently_marked_completed():
    clock = _clock()
    goals = SimpleGoalEngine(now_fn=clock)
    bank = goals.ingest("Mañana tengo que llamar al banco a las 10.")[0]

    clock.advance(hours=48)

    assert bank.completed is False
    assert bank in goals.due()
