"""Structure-only checks: never execute V6 turns against a candidate here."""
import hashlib
from pathlib import Path

from src.infinito3.trajectory_holdout_v6_cases import independent_trajectory_holdout_v6_suite


def test_v6_frozen_source_and_structure():
    path = Path("src/infinito3/trajectory_holdout_v6_cases.py")
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "19c645820733bcf0b1a9f6341f656fe5b6a02fa653770982a87b372170265c1a")
    suite = independent_trajectory_holdout_v6_suite()
    assert len(suite) == 4
    assert sum(len(s.steps) for s in suite) == 122
    probes = [p for s in suite for p in s.steps if p.expectation is not None]
    assert len(probes) == 28
    assert len({p.label for p in probes}) == 28
    assert all(p.label.startswith("v6 ") for p in probes)
    assert all(s.history_limit <= 5 and len(s.steps) >= 25 for s in suite)
    assert sum("empty_context" in p.tags for p in probes) == 4
    assert any("closure_audit" in p.tags for p in probes)
    assert any("literal_data_audit" in p.tags for p in probes)
