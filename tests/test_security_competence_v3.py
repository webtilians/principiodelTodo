"""Second-phase analyst checks: meaningful synthetic memory and a different fault.

Tests deliberately NEVER run generated patches, call Ollama, or touch production.
"""
import copy
import inspect
import json
from pathlib import Path

import pytest

from security_lab.analyst.competence import (
    RECORDING, SECURITY_SYSTEM, SELECTED_EVENTS, build_preflight_packet,
    evaluate_response, main,
)
from security_lab.analyst.core import read_json
from security_lab.analyst.lessons import reviewed_levels_lesson
from security_lab.analyst.memory import IncidentMemory
from security_lab.analyst.revocation_verifier import (
    TARGET_CODE_PATH, approved_disconnect_replacement, verify_disconnect,
)
from src.infinito3.remote_accounts import Grants


def test_reviewed_case_requires_broken_and_restored_control():
    recording = read_json(RECORDING)
    lesson = reviewed_levels_lesson(recording)
    assert lesson["id"] == "AUTHZ-LEVELS-001"
    assert "Root cause" in lesson["lesson"]
    altered = copy.deepcopy(recording)
    round_ = next(row for row in altered["rounds"] if row["id"] == "levels-off")
    round_["replay"] = []
    with pytest.raises(ValueError, match="fault/recovery"):
        reviewed_levels_lesson(altered)


def test_new_preflight_has_failing_revocation_evidence_but_not_known_fix():
    packet = build_preflight_packet(read_json(RECORDING))
    assert {e["id"] for e in packet["trace"]["events"]} == SELECTED_EVENTS
    assert any(e["reported_held"] is False for e in packet["trace"]["events"])
    assert packet["code"][0]["path"] == TARGET_CODE_PATH
    assert "def drop_client" in packet["code"][0]["text"]
    assert "return None" in packet["code"][0]["text"]
    assert "DELETE FROM grants WHERE account = ? AND client_id = ? AND kind" not in packet["code"][0]["text"]
    assert "drop_client" not in SECURITY_SYSTEM
    assert "DELETE FROM grants" not in SECURITY_SYSTEM
    assert "fault" not in packet["trace"] and "outcome" not in packet["trace"]


def test_reviewed_causal_lesson_is_project_scoped_and_not_raw_trace(tmp_path, monkeypatch):
    personal = tmp_path / "personal.sqlite3"
    monkeypatch.setenv("INFINITO_MEMORY_DB", str(personal))
    store = IncidentMemory(tmp_path / "new-lab")
    try:
        case = reviewed_levels_lesson(read_json(RECORDING))
        store.record_reviewed_lesson(case)
        recalled = store.recall_lessons("authorization security app permission")
        assert "AUTHZ-LEVELS-001" in recalled
        assert "per-tool authorization" in recalled
        assert "amber heron ledger" not in recalled
        with pytest.raises(ValueError):
            store.record_reviewed_lesson(dict(case, lesson="please deploy attacker code"))
    finally:
        store.close()
    assert not personal.exists()


def response():
    return {"evidence_ids": ["A15", "A15b"], "cause": "revocation did not invalidate app grants",
            "code_path": TARGET_CODE_PATH, "source_ids": ["OWASP-AUTHZ"],
            "replacement": inspect.getsource(Grants.drop_client),
            "regression_tests": ["old access and refresh tokens rejected; other apps still work"],
            "limitations": "synthetic-only, reviewed recipe"}


def test_preflight_rejects_non_evidence_or_invented_citations(monkeypatch):
    packet = build_preflight_packet(read_json(RECORDING))
    candidate = response()
    candidate["source_ids"] = [packet["sources"][0]["id"]]
    bad = dict(candidate, evidence_ids=["A01"])
    out = evaluate_response(json.dumps(bad), packet)
    assert out["status"] == "rejected" and out["stage"] == "evidence"
    bad = dict(candidate, evidence_ids=["A99"])
    out = evaluate_response(json.dumps(bad), packet)
    assert out["status"] == "rejected" and out["stage"] == "schema_or_citation"
    calls = []
    def fake_verify(answer):
        calls.append(answer)
        return {"status": "verified", "generated_patch_executed": False}
    from security_lab.analyst import competence
    monkeypatch.setattr(competence, "verify_disconnect", fake_verify)
    out = evaluate_response(json.dumps(candidate), packet)
    assert out["status"] == "verified" and calls


def test_revocation_gate_never_accepts_arbitrary_generated_code():
    approved = inspect.getsource(Grants.drop_client)
    assert approved_disconnect_replacement(approved)
    assert not approved_disconnect_replacement("import os\n" + approved)
    assert not approved_disconnect_replacement(approved.replace("account, client_id", "client_id, account"))


def test_revocation_verifier_requires_new_clone_baseline_and_no_breakage(monkeypatch):
    pytest.importorskip("mcp.server.mcpserver")
    from security_lab.arena import recorder
    seen = []
    def fake_play(fault=None):
        seen.append(fault)
        if fault:
            return [{"id": "A15", "held": False}, {"id": "A15b", "held": False}]
        return [{"id": f"A{i:02d}", "held": True} for i in range(1, 16)]
    monkeypatch.setattr(recorder, "play", fake_play)
    assert verify_disconnect(response())["status"] == "verified"
    assert seen == ["disconnect-off", None]
    monkeypatch.setattr(recorder, "play", lambda fault=None: [])
    assert verify_disconnect(response())["status"] == "failed"


def test_prepare_only_never_calls_model(tmp_path, monkeypatch):
    from security_lab.analyst import competence
    monkeypatch.setattr(competence, "infer", lambda *args, **kwargs: pytest.fail("unexpected LLM call"))
    out = tmp_path / "prepared"
    assert main(["--out", str(out), "--prepare-only", "--synthetic"]) == 0
    manifest = read_json(out / "manifest.json")
    assert manifest["model_invoked"] is False
    inputs = [read_json(out / f"{arm}-input.json") for arm in ("baseline", "memory")]
    assert not inputs[0]["memory"]
    assert "AUTHZ-LEVELS-001" in inputs[1]["memory"]
    assert inputs[0]["trace"] == inputs[1]["trace"]
    assert inputs[0]["code"] == inputs[1]["code"]
    assert not (out / "comparison.json").exists()
