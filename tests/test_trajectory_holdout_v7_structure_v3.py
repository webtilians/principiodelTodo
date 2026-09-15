"""Structure-only checks: never execute V7 turns against a candidate here."""
import hashlib
from pathlib import Path

from src.infinito3.trajectory_holdout_v7_cases import independent_trajectory_holdout_v7_suite


def test_v7_frozen_source_and_structure():
    path = Path("src/infinito3/trajectory_holdout_v7_cases.py")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "0ea8d349d9789cff2f8dbef0b3d53d73bc061c7e839a7ecd250fd2df3596f3f0"
    )
    suite = independent_trajectory_holdout_v7_suite()
    assert len(suite) == 4
    assert sum(len(s.steps) for s in suite) == 122
    probes = [step for scenario in suite for step in scenario.steps if step.expectation is not None]
    assert len(probes) == 28
    assert len({step.label for step in probes}) == 28
    assert all(step.label.startswith("v7 ") for step in probes)
    assert all("heldout_v7" in scenario.tags for scenario in suite)
    assert all(scenario.history_limit <= 5 and len(scenario.steps) >= 25 for scenario in suite)
    assert sum("empty_context" in step.tags for step in probes) == 4
    assert any("closure_audit" in step.tags for step in probes)
    assert any("literal_data_audit" in step.tags for step in probes)


def test_v7_is_not_v6_relabelled():
    path = Path("src/infinito3/trajectory_holdout_v7_cases.py")
    text = path.read_text(encoding="utf-8")
    for prior_entity in (
        "Idris", "Graz", "Yeti SB140", "Petra", "bookbinding", "Selma", "Trieste", "projector"
    ):
        assert prior_entity not in text
