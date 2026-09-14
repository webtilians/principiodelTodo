from src.infinito3.trajectory_holdout_v2_cases import independent_trajectory_holdout_v2_suite


def test_second_heldout_bank_is_long_independent_and_probe_rich():
    scenarios = independent_trajectory_holdout_v2_suite()
    turns = sum(len(scenario.steps) for scenario in scenarios)
    probes = [
        step
        for scenario in scenarios
        for step in scenario.steps
        if step.expectation is not None
    ]

    assert len(scenarios) == 4
    assert 100 <= turns <= 200
    assert len(probes) >= 16
    assert all(scenario.history_limit <= 6 for scenario in scenarios)
    assert all("heldout_v2" in scenario.tags for scenario in scenarios)


def test_second_heldout_probe_labels_are_unique():
    scenarios = independent_trajectory_holdout_v2_suite()
    labels = [
        step.label
        for scenario in scenarios
        for step in scenario.steps
        if step.expectation is not None
    ]

    assert all(labels)
    assert len(labels) == len(set(labels))
