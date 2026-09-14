from src.infinito3.trajectory_holdout_v3_cases import independent_trajectory_holdout_v3_suite


def test_third_heldout_is_frozen_long_horizon_bank():
    scenarios = independent_trajectory_holdout_v3_suite()
    assert len(scenarios) == 4

    turn_count = sum(len(s.steps) for s in scenarios)
    probe_steps = [step for s in scenarios for step in s.steps if step.expectation is not None]
    labels = [step.label for step in probe_steps]

    assert turn_count == 141
    assert 100 <= turn_count <= 200
    assert len(probe_steps) == 28
    assert len(labels) == len(set(labels))
    assert all("heldout_v3" in s.tags for s in scenarios)


def test_third_heldout_covers_required_stressors():
    scenarios = independent_trajectory_holdout_v3_suite()
    tags = {tag for s in scenarios for tag in s.tags}
    assert {"profile", "historical", "cross_language", "goals", "lifecycle", "preferences", "retraction", "prompt_hygiene"} <= tags
