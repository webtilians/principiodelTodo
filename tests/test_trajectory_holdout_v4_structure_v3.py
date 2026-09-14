from src.infinito3.trajectory_holdout_v4_cases import independent_trajectory_holdout_v4_suite


def test_v4_frozen_bank_has_expected_shape_and_unique_probes():
    scenarios = independent_trajectory_holdout_v4_suite()
    assert len(scenarios) == 4
    names = [scenario.name for scenario in scenarios]
    assert names == [
        "v4_profile_paraphrase_and_three_state_chain",
        "v4_natural_commitments_and_lifecycle",
        "v4_preference_language_without_like_verbs",
        "v4_mixed_inert_notes_profile_and_future_state",
    ]
    user_turns = sum(sum(step.user_text is not None for step in scenario.steps) for scenario in scenarios)
    probes = [step for scenario in scenarios for step in scenario.steps if step.expectation is not None]
    labels = [step.label for step in probes]
    assert 120 <= user_turns <= 180
    assert len(probes) >= 24
    assert len(labels) == len(set(labels))
    tags = {tag for scenario in scenarios for tag in scenario.tags}
    assert {"semantic_events", "profile", "goals", "preferences", "mixed"} <= tags
