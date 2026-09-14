import hashlib
from pathlib import Path

from src.infinito3.trajectory_holdout_v5_cases import independent_trajectory_holdout_v5_suite


FROZEN_V5_SHA256 = "8a0a4600d9545e63ffa5da77006ed324c0a98c50c41d41d57ed11f529ecfd222"


def test_v5_bank_is_frozen_and_has_expected_shape():
    path = Path("src/infinito3/trajectory_holdout_v5_cases.py")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == FROZEN_V5_SHA256

    suite = independent_trajectory_holdout_v5_suite()
    assert len(suite) == 4
    assert sum(len(scenario.steps) for scenario in suite) == 130

    probes = [
        step
        for scenario in suite
        for step in scenario.steps
        if step.expectation is not None
    ]
    assert len(probes) == 29
    assert len({step.label for step in probes}) == 29
    assert all(step.label and step.label.startswith("v5 ") for step in probes)


def test_v5_uses_independent_long_horizon_trajectories():
    suite = independent_trajectory_holdout_v5_suite()
    assert all(scenario.history_limit <= 5 for scenario in suite)
    assert all(len(scenario.steps) >= 25 for scenario in suite)
    assert all("heldout_v5" in scenario.tags for scenario in suite)
