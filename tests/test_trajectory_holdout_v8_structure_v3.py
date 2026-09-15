"""Structure-only checks: never execute V8 turns against a candidate here."""
import hashlib
from pathlib import Path

from src.infinito3.trajectory_holdout_v8_cases import independent_trajectory_holdout_v8_suite


def _git_blob_sha1(path):
    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode()
    return hashlib.sha1(header + data).hexdigest()


def test_v8_frozen_source_and_structure():
    path = Path("src/infinito3/trajectory_holdout_v8_cases.py")
    assert _git_blob_sha1(path) == "a7dc174fa173ef248cb732144f6a1e319dd23fc6"
    suite = independent_trajectory_holdout_v8_suite()
    assert len(suite) == 4
    assert sum(len(s.steps) for s in suite) == 122
    probes = [p for s in suite for p in s.steps if p.expectation is not None]
    assert len(probes) == 28
    assert len({p.label for p in probes}) == 28
    assert all(p.label.startswith("v8 ") for p in probes)
    assert all(s.history_limit <= 5 and len(s.steps) >= 25 for s in suite)
    assert sum("empty_context" in p.tags for p in probes) == 4
    assert any("closure_audit" in p.tags for p in probes)
    assert any("literal_data_audit" in p.tags for p in probes)
    assert sum("reschedule_identity_audit" in p.tags for p in probes) >= 2
