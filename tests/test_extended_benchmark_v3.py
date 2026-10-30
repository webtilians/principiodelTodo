from src.infinito3 import extended_evaluation_suite


def test_extended_suite_has_unique_broad_scenario_bank():
    suite = extended_evaluation_suite()
    names = [scenario.name for scenario in suite]

    assert len(suite) == 20
    assert len(names) == len(set(names))
    assert "long_term_name" in names
    assert "contradiction_under_noise" in names
    assert "semantic_paraphrase_gap" in names
    assert "cross_lingual_retrieval_gap" in names
    assert "instruction_like_memory_is_data" in names


def test_extended_suite_contains_controls_strengths_and_challenges():
    suite = extended_evaluation_suite()
    tags = {tag for scenario in suite for tag in scenario.tags}

    assert "expected_strength" in tags
    assert "expected_challenge" in tags
    assert "negative_control" in tags
    assert "adversarial" in tags
    assert "self_retrieval" in tags


def test_long_term_probes_do_not_leak_setup_through_short_history():
    suite = extended_evaluation_suite()
    exempt = {"current_turn_self_retrieval_guard"}

    for scenario in suite:
        if scenario.name not in exempt:
            assert scenario.history_limit == 0


def test_small_budget_case_is_actually_bounded():
    scenario = next(
        scenario
        for scenario in extended_evaluation_suite()
        if scenario.name == "small_budget_identity"
    )
    assert scenario.context_budget_tokens == 80
