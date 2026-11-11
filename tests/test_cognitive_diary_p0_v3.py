"""P0 regression contract: no real models, memories, API calls or policy writes."""
import json
import subprocess
import sys

import pytest

from cognitive_diary_p0 import AuditError, Journal, fictional_day


def test_replay_is_byte_for_byte_deterministic():
    a, b = fictional_day(), fictional_day()
    assert a.snapshot() == b.snapshot()
    assert a.markdown() == b.markdown()
    assert len(a.entries) == 10
    assert all(e["model"] == "deterministic-fixture" and e["input_tokens"] == 0 and
               e["output_tokens"] == 0 and e["cost_usd"] == 0 for e in a.entries)
    assert a.active_policies() == []


def test_supervisor_sees_original_sources_and_reconsiders_opinion():
    diary = fictional_day()
    review = next(e for e in diary.entries if e["kind"] == "supervisor_review")
    assert {"e0001", "e0002", "e0003"}.issubset(set(review["sources"]))
    assert "e0003" in diary.bodies[review["id"]]["missed_by_observer"]
    later = diary.bodies["e0009"]
    assert later["earlier_reflection"] == "e0007" and later["relation"] == "reconsiders"
    assert diary.bodies["e0010"]["status"] == "pending_owner_approval"
    assert diary.bodies["e0006"]["result"] == "rejected"


def test_chained_audit_headers_detect_changes():
    diary = fictional_day()
    snap = diary.snapshot()
    snap["entries"][4]["sources"][0] = "e9999"
    with pytest.raises(AuditError, match="hash chain broken"):
        Journal.load(snap)
    snap = diary.snapshot()
    snap["bodies"]["e0010"]["suggested_text"] = "different instruction"
    with pytest.raises(AuditError, match="payload missing or modified"):
        Journal.load(snap)


def test_proposal_is_never_applied_without_explicit_owner_approval():
    diary = fictional_day()
    with pytest.raises(AuditError, match="policy application needs"):
        diary.append("policy_applied", "system", {}, at="2026-01-15T18:15:00Z", sources=("e0010",))
    with pytest.raises(AuditError, match="owner"):
        diary.append("policy_decision", "supervisor", {"decision": "approved"},
                     at="2026-01-15T18:15:00Z", sources=("e0010",))
    rejection = diary.append("policy_decision", "owner", {"decision": "rejected"},
                             at="2026-01-15T18:15:00Z", sources=("e0010",))
    with pytest.raises(AuditError, match="not user-approved"):
        diary.append("policy_applied", "system", {}, at="2026-01-15T18:17:00Z", sources=(rejection,))
    assert diary.active_policies() == []


def test_scope_and_owner_must_match():
    diary = fictional_day()
    with pytest.raises(AuditError, match="cross-tenant"):
        diary.append("observation", "observer", {}, at="2026-01-15T19:00:00Z",
                     owner="other-user", sources=("e0001",))
    with pytest.raises(AuditError, match="cross-tenant"):
        diary.append("observation", "observer", {}, at="2026-01-15T19:00:00Z",
                     scope="unrelated-project", sources=("e0001",))
    other = Journal(owner="other-user", scope="fictional-lab")
    with pytest.raises(AuditError, match="unknown reference"):
        other.append("observation", "observer", {}, at="2026-01-15T19:00:00Z", sources=("e0001",))


def test_forget_erases_source_and_all_transitive_derived_payloads():
    diary = fictional_day()
    source_text = diary.bodies["e0003"]["summary"]
    diary.forget("e0003", at="2026-01-15T20:00:00Z")
    diary.verify()
    for event_id in ("e0003", "e0008", "e0009", "e0010"):
        assert event_id not in diary.bodies
    assert "e0001" in diary.bodies
    assert source_text not in json.dumps(diary.snapshot())
    assert diary.active_policies() == []
    with pytest.raises(AuditError, match="forgotten content"):
        diary.append("observation", "observer", {}, at="2026-01-15T20:01:00Z", sources=("e0003",))
    Journal.load(diary.snapshot())


def test_verification_requires_referee():
    diary = fictional_day()
    with pytest.raises(AuditError, match="independent referee"):
        diary.append("verification", "investigator", {"result": "approved"},
                     at="2026-01-15T19:00:00Z", sources=("e0005",))


def test_cli_run_and_replay(tmp_path):
    root = str(tmp_path / "diary")
    command = [sys.executable, "-m", "cognitive_diary_p0"]
    result = subprocess.run(command + ["--run", "--output-dir", root], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    result = subprocess.run(command + ["--replay", "--output-dir", root], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "REPLAY PASS" in result.stdout and "model_calls=0" in result.stdout
    result = subprocess.run(command + ["--export", "--format", "markdown"], capture_output=True, text=True)
    assert result.returncode == 0 and "SIMULACIÓN DETERMINISTA" in result.stdout
