from src.infinito3.trajectory_holdout_cases import independent_trajectory_holdout_suite


def test_trajectory_holdout_bank_is_frozen_and_nontrivial():
    scenarios = independent_trajectory_holdout_suite()

    assert [scenario.name for scenario in scenarios] == [
        "semantic_facets_under_dense_preferences",
        "calendar_language_and_cancellation",
        "preference_revocation_after_noise",
    ]
    assert all(len(scenario.steps) >= 15 for scenario in scenarios)

    probes = [
        step
        for scenario in scenarios
        for step in scenario.steps
        if step.expectation is not None
    ]
    assert len(probes) == 7
    assert len({step.label for step in probes}) == 7


def test_holdout_bank_covers_distinct_unseen_failure_modes():
    scenarios = independent_trajectory_holdout_suite()
    tags = {tag for scenario in scenarios for tag in scenario.tags}

    assert {"semantic", "calendar", "cancellation", "revision", "negation"} <= tags
