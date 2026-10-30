from src.infinito3 import (
    CallablePairwiseJudge,
    CognitiveEngine,
    CognitiveLoop,
    EvaluationExpectation,
    EvaluationHarness,
    EvaluationScenario,
    PairwiseJudgeResult,
    RecordingLLMAdapter,
    standard_evaluation_suite,
)


def _context_aware_responder(request):
    reference = "\n".join(
        message.content
        for message in request.messages
        if "INFINITO REFERENCE CONTEXT" in message.content
    )
    question = request.messages[-1].content.lower()

    if "cómo me llamo" in question:
        return "Te llamas Alicia." if "Alicia" in reference else "No lo sé."
    if "dónde vivo" in question:
        return "Vives en Bilbao." if "Bilbao" in reference else "No lo sé."
    if "qué tengo pendiente" in question:
        return (
            "Tienes pendiente entrenar descenso."
            if "entrenar descenso" in reference
            else "No tengo información suficiente."
        )
    if "qué bici uso" in question:
        return (
            "Usas una Trek Session."
            if "Trek Session" in reference
            else "No lo sé."
        )
    return "ok"


def _loop_factory():
    engine = CognitiveEngine.persistent(":memory:")
    adapter = RecordingLLMAdapter(responder=_context_aware_responder)
    return CognitiveLoop(engine, adapter)


def test_standard_suite_shows_cognitive_lift():
    harness = EvaluationHarness(_loop_factory)
    report = harness.run(standard_evaluation_suite())

    assert report.summary.scenario_count == 4
    assert report.summary.comparable_answer_count == 4
    assert report.summary.cognitive_wins == 4
    assert report.summary.baseline_wins == 0
    assert report.summary.mean_cognitive_answer_score == 1.0
    assert report.summary.mean_baseline_answer_score < 1.0
    assert all(not result.setup_failures for result in report.results)


def test_each_scenario_gets_a_fresh_cognitive_loop():
    created = []

    def factory():
        loop = _loop_factory()
        created.append(loop)
        return loop

    scenarios = (
        EvaluationScenario(
            name="seed",
            setup_turns=("Me llamo Alicia.",),
            probe="¿Cómo me llamo?",
            history_limit=0,
            expectation=EvaluationExpectation(
                answer_contains=("Alicia",),
                context_contains=("Alicia",),
            ),
        ),
        EvaluationScenario(
            name="clean",
            probe="¿Cómo me llamo?",
            history_limit=0,
            expectation=EvaluationExpectation(
                answer_excludes=("Alicia",),
                context_excludes=("Alicia",),
            ),
        ),
    )

    report = EvaluationHarness(factory).run(scenarios)

    assert len(created) == 2
    assert report.results[0].cognitive_metrics.context_required_recall == 1.0
    assert report.results[1].cognitive_metrics.context_forbidden_rate == 0.0


def test_forbidden_fact_penalizes_answer_and_context():
    scenario = EvaluationScenario(
        name="contradiction",
        setup_turns=("Vivo en Sevilla.", "Vivo en Bilbao."),
        probe="¿Dónde vivo ahora?",
        history_limit=0,
        expectation=EvaluationExpectation(
            answer_contains=("Bilbao",),
            answer_excludes=("Sevilla",),
            context_contains=("Bilbao",),
            context_excludes=("Sevilla",),
        ),
    )

    result = EvaluationHarness(_loop_factory).run_scenario(scenario)

    assert result.cognitive_metrics.answer_required_recall == 1.0
    assert result.cognitive_metrics.answer_forbidden_rate == 0.0
    assert result.cognitive_metrics.context_required_recall == 1.0
    assert result.cognitive_metrics.context_forbidden_rate == 0.0
    assert result.cognitive_metrics.answer_score == 1.0
    assert result.cognitive_metrics.context_score == 1.0


def test_report_serializes_to_json_and_markdown():
    report = EvaluationHarness(_loop_factory).run(standard_evaluation_suite()[:1])

    payload = report.to_json()
    markdown = report.to_markdown()

    assert '"scenario_count": 1' in payload
    assert "INFINITO 3.0 Evaluation Report" in markdown
    assert "long_term_identity" in markdown


def test_optional_pairwise_judge_is_called():
    calls = []

    def judge_callback(scenario, comparison):
        calls.append((scenario.name, comparison.user_text))
        return PairwiseJudgeResult(
            winner="cognitive",
            score=0.9,
            rationale="context recovered the expected fact",
        )

    harness = EvaluationHarness(
        _loop_factory,
        judge=CallablePairwiseJudge(judge_callback),
    )
    result = harness.run_scenario(standard_evaluation_suite()[0])

    assert calls == [("long_term_identity", "¿Cómo me llamo?")]
    assert result.judge is not None
    assert result.judge.winner == "cognitive"
    assert result.judge.score == 0.9


def test_tag_aggregation_reports_lift():
    report = EvaluationHarness(_loop_factory).run(standard_evaluation_suite())

    assert "factual_continuity" in report.summary.by_tag
    tag = report.summary.by_tag["factual_continuity"]
    assert tag["scenario_count"] == 2.0
    assert tag["answer_lift"] > 0.0


def test_relevance_scenario_does_not_force_unrelated_preference():
    scenario = standard_evaluation_suite()[3]
    result = EvaluationHarness(_loop_factory).run_scenario(scenario)

    packet = result.comparison.cognitive.cognitive_decision.context_packet
    assert packet is not None
    assert "Trek Session" in packet.rendered
    assert "café" not in packet.rendered
