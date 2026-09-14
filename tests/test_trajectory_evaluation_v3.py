from src.infinito3.cognitive_loop import CognitiveLoop
from src.infinito3.engine import CognitiveEngine
from src.infinito3.evaluation import EvaluationExpectation
from src.infinito3.generalized_context_builder import GeneralizedContextBuilder
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.llm_adapter import RecordingLLMAdapter
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.trajectory_cases import independent_trajectory_suite
from src.infinito3.trajectory_evaluation import (
    MutableClock,
    TrajectoryEvaluationHarness,
    TrajectoryScenario,
    TrajectoryStep,
)


def test_independent_trajectories_commit_their_own_assistant_histories():
    adapters = []

    def factory(clock, scenario):
        def make_loop(label):
            adapter = RecordingLLMAdapter(
                lambda request: label,
                model=f"{label}-model",
            )
            adapters.append(adapter)
            return CognitiveLoop(CognitiveEngine(), adapter, history_limit=6)

        return make_loop("BASELINE_ONLY"), make_loop("COGNITIVE_ONLY")

    scenario = TrajectoryScenario(
        name="independent_history",
        steps=(
            TrajectoryStep("primer turno"),
            TrajectoryStep("segundo turno"),
        ),
    )
    TrajectoryEvaluationHarness(factory).run((scenario,))

    baseline_second = adapters[0].requests[1]
    cognitive_second = adapters[1].requests[1]
    baseline_text = "\n".join(message.content for message in baseline_second.messages)
    cognitive_text = "\n".join(message.content for message in cognitive_second.messages)

    assert "BASELINE_ONLY" in baseline_text
    assert "COGNITIVE_ONLY" not in baseline_text
    assert "COGNITIVE_ONLY" in cognitive_text
    assert "BASELINE_ONLY" not in cognitive_text


def test_probe_scoring_and_summary_use_full_trajectory_runs():
    def factory(clock, scenario):
        def responder(request):
            return "Clara" if request.metadata.get("infinito_cognition") else "No lo sé"

        return (
            CognitiveLoop(CognitiveEngine(), RecordingLLMAdapter(responder), history_limit=4),
            CognitiveLoop(CognitiveEngine(), RecordingLLMAdapter(responder), history_limit=4),
        )

    scenario = TrajectoryScenario(
        name="scored",
        steps=(
            TrajectoryStep("ruido"),
            TrajectoryStep(
                "¿Cómo me llamo?",
                expectation=EvaluationExpectation(answer_contains=("Clara",)),
            ),
        ),
    )
    report = TrajectoryEvaluationHarness(factory).run((scenario,))

    assert report.summary.user_turn_count == 2
    assert report.summary.probe_count == 1
    assert report.summary.cognitive_wins == 1
    assert report.summary.baseline_wins == 0
    assert report.summary.mean_cognitive_answer_score == 1.0
    assert report.summary.mean_baseline_answer_score == 0.0


def test_simulated_clock_advances_before_the_turn():
    observed = []

    def factory(clock, scenario):
        def responder(request):
            observed.append(clock())
            return "ok"

        return (
            CognitiveLoop(CognitiveEngine(), RecordingLLMAdapter(responder)),
            CognitiveLoop(CognitiveEngine(), RecordingLLMAdapter(responder)),
        )

    scenario = TrajectoryScenario(
        name="clock",
        steps=(
            TrajectoryStep("uno"),
            TrajectoryStep("dos", advance_hours=30),
        ),
    )
    TrajectoryEvaluationHarness(factory).run((scenario,))

    assert observed[0] == scenario.start_at
    assert observed[1] == scenario.start_at
    assert observed[2] == scenario.start_at.replace(day=15, hour=15)
    assert observed[3] == scenario.start_at.replace(day=15, hour=15)


def test_prefixed_informational_question_does_not_create_a_goal():
    clock = MutableClock(TrajectoryScenario(name="clock", steps=()).start_at)
    goals = SimpleGoalEngine(now_fn=clock)
    assert goals.ingest("Mañana tengo que llamar al banco a las 10.")
    before = len(goals.all())

    created = goals.ingest("Hoy es 17 de septiembre. ¿Qué tengo pendiente mañana?")

    assert created == []
    assert len(goals.all()) == before


def test_polite_reminder_question_still_creates_a_goal():
    clock = MutableClock(TrajectoryScenario(name="clock", steps=()).start_at)
    goals = SimpleGoalEngine(now_fn=clock)

    created = goals.ingest("¿Puedes recordarme mañana llamar al banco a las 10?")

    assert len(created) == 1
    assert created[0].due_at is not None


def test_future_goal_query_excludes_overdue_goals_but_keeps_future_one():
    clock = MutableClock(TrajectoryScenario(name="clock", steps=()).start_at)
    goals = SimpleGoalEngine(now_fn=clock)
    goals.ingest("Mañana tengo que llamar al banco a las 10.")
    goals.ingest("Pasado mañana tengo cita con el dentista a las 16.")
    builder = GeneralizedContextBuilder(InMemoryMemoryStore(), goals, now_fn=clock)

    clock.advance(hours=30)
    packet = builder.build(
        "Ahora es 15 de septiembre por la tarde. ¿Qué cita futura sigo teniendo?",
        memory_candidates=[],
    )

    assert "dentista" in packet.rendered
    assert "banco" not in packet.rendered


def test_frozen_long_horizon_bank_has_multiple_dozen_turn_trajectories():
    suite = independent_trajectory_suite()

    assert len(suite) == 3
    assert all(sum(step.user_text is not None for step in scenario.steps) >= 24 for scenario in suite)
    assert sum(
        step.expectation is not None
        for scenario in suite
        for step in scenario.steps
    ) >= 10
    assert {"profile_drift_under_noise", "temporal_goal_lifecycle", "memory_pressure_and_prompt_hygiene"} == {
        scenario.name for scenario in suite
    }
