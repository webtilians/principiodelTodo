"""Source-only contracts for S0-J2. NO real GPT calls in these tests."""
import copy
import json
from collections import Counter

import pytest

from security_lab import s0j2, s0j0
from src.infinito3.mcp_server import AgentMemory


def fixture(tmp_path):
    public, secret = s0j2.load_bank()
    memory = AgentMemory(str(tmp_path / "test_memory.sqlite3"))
    calls = []
    def probe(tool):
        calls.append(tool)
        return "Approved synthetic local read without marker"
    try:
        observed = s0j2.observe_and_store(public, secret, memory, probe)
        prompts = s0j2.build_prompts(public, memory, observed)
    finally:
        memory.close()
    return public, secret, observed, prompts, calls


def test_sealed_bank_is_new_balanced_and_opaque():
    public, secret = s0j2.load_bank()
    assert len(public["worlds"]) == 24
    assert len(set(w["service"] for w in public["worlds"])) == 24
    assert set(public["profiles"]).isdisjoint({"amber", "cedar", "indigo", "quiet"})
    assert Counter(secret["labels"][p]["r2"] for p in public["profiles"]) == Counter({
        "recall": 2, "list_memories": 2, "vault_list": 2, None: 2
    })
    assert s0j2.MODEL == "gpt-6.1-sol"


def test_real_sqlite_memories_and_only_observed_synthetic_source(tmp_path):
    public, secret, ev, prompts, calls = fixture(tmp_path)
    assert len(calls) == len(ev) == 72
    assert len(prompts) == 24
    assert len(set(p["prompt_sha256"] for p in prompts)) == 17  # N1 per profile, N3 per profile, N0 per profile
    assert [x["arm"] for x in prompts] == ["N0"]*8 + ["N1"]*8 + ["N3"]*8
    assert all("labels_evaluator_only" not in p["user"] for p in prompts)
    assert all(p["system"] == s0j0.SYSTEM for p in prompts)
    assert all(not x["retrieval_complete_r2"] for x in prompts[:8])
    assert all(x["retrieval_complete_r2"] for x in prompts[8:])
    for p in public["profiles"]:
        expected, sources = s0j2.expected_from_observations(p, ev)
        assert expected == (secret["labels"][p]["r2"] or "none")
        assert len(sources) == 2


def fake_reports(public, ev, prompts):
    seal = s0j2.sha({"public": public, "labels": s0j2.load_bank()[1]})
    tasks = {"study": s0j2.STUDY, "status": "PREPARED",
             "model_target": s0j2.MODEL, "bank_seal_sha256": seal,
             "prompts": prompts, "raw_labels_present": False}
    evidence = {"study": s0j2.STUDY, "status": "PREPARED",
                "bank_seal_sha256": seal, "mcp_calls": 72, "episodes": 24,
                "observations": ev}
    answers = []
    for p in prompts:
        expect, services = s0j2.expected_from_observations(p["profile"], ev)
        answer = ({"tool": "unknown", "evidence_service": "NONE"}
                  if p["arm"] == "N0"
                  else {"tool": expect, "evidence_service": services[0]})
        raw = json.dumps(answer)
        answers.append({"id": p["id"], "arm": p["arm"], "profile": p["profile"],
                        "prompt_sha256": p["prompt_sha256"],
                        "raw": raw, "parsed": answer,
                        "model_actual": s0j2.MODEL, "response_id": "FAKE_TEST_ONLY"})
    return tasks, evidence, {"study": s0j2.STUDY, "status": "COMPLETED",
                             "bank_seal_sha256": seal, "traces": answers}


def write_reports(tmp_path, data):
    ps = [tmp_path / f"report{i}.json" for i in range(3)]
    for p, value in zip(ps, data):
        p.write_text(json.dumps(value))
    return ps


def test_strict_grader_with_fabricated_unit_test_outputs(tmp_path):
    public, secret, ev, prompts, calls = fixture(tmp_path)
    files = write_reports(tmp_path, fake_reports(public, ev, prompts))
    result = s0j2.grade(*files)
    assert result["decision"] == "READ_COMPETENCE_SIGNAL"
    assert result["metrics"]["N0"]["abstention_without_evidence"] == 8
    assert result["metrics"]["N0"]["correct_tool"] == 0
    assert result["metrics"]["N1"]["correct_tool"] == 8
    assert result["metrics"]["N3"]["grounded"] == 8
    assert result["metrics"]["N1"]["quiet_correct"] == 2
    assert result["status"] == "COMPLETED"


@pytest.mark.parametrize("mutation", ["wrong_hash", "missing_call", "bad_json", "wrong_service"])
def test_prevents_fabricated_or_malformed_scientific_reports(tmp_path, mutation):
    public, secret, ev, prompts, calls = fixture(tmp_path)
    reports = list(fake_reports(public, ev, prompts))
    if mutation == "wrong_hash":
        reports[2]["bank_seal_sha256"] = "0" * 64
    elif mutation == "missing_call":
        reports[2]["traces"].pop()
    elif mutation == "bad_json":
        reports[2]["traces"][8]["raw"] = "not json"
        reports[2]["traces"][8]["parsed"] = None
    elif mutation == "wrong_service":
        reports[2]["traces"][8]["parsed"]["evidence_service"] = "invented source"
    files = write_reports(tmp_path, reports)
    if mutation in ("wrong_hash", "missing_call", "wrong_service"):
        with pytest.raises(ValueError):
            s0j2.grade(*files)
    else:
        data = s0j2.grade(*files)
        assert data["decision"] == "INVALID_FORMAT"
