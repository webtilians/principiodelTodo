from src.infinito3.trajectory_holdout_v10_cases import independent_trajectory_holdout_v10_suite


def test_v10_frozen_structure_and_control_count():
    suite = independent_trajectory_holdout_v10_suite()
    assert len(suite) == 4
    assert sum(len(scenario.steps) for scenario in suite) == 122
    probes = [
        step
        for scenario in suite
        for step in scenario.steps
        if step.expectation is not None
    ]
    assert len(probes) == 28
    assert sum("empty_context" in step.tags for step in probes) == 4
    assert all("heldout_v10" in scenario.tags for scenario in suite)


def test_v10_does_not_reuse_v9_named_entities():
    text = "\n".join(
        step.text
        for scenario in independent_trajectory_holdout_v10_suite()
        for step in scenario.steps
    ).lower()
    for old in (
        "samira", "riga", "lugano", "basel", "jeffsy", "commencal meta",
        "thermal camera", "audiology", "binoculars", "woodblock", "canoe touring",
        "metal embossing", "harmonica", "elias", "split", "olomouc", "spectrum analyzer",
    ):
        assert old not in text


def test_v10_contains_planner_boundary_probes():
    probes = {
        step.label: step
        for scenario in independent_trajectory_holdout_v10_suite()
        for step in scenario.steps
        if step.expectation is not None
    }
    expected = {
        "v10 profile fields",
        "v10 residence predecessor",
        "v10 noon shoulder",
        "v10 latest craft",
        "v10 literal exact data",
    }
    assert expected <= set(probes)
    assert all("planner_audit" in probes[label].tags for label in expected)
