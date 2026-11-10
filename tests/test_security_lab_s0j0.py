"""Prospective technical contracts for S0-J0: no Qwen in unit tests.

These deliberately fabricated answers test the SCORER only; they are not
experimental findings. Scientific decisions require real pinned GGUF inference.
"""
import copy
import json

import pytest

from security_lab import s0i, s0j0
from security_lab.s0c import MODEL_FILE, MODEL_REPO, MODEL_REVISION, MODEL_SHA256
from src.infinito3.mcp_server import AgentMemory


def events_from_frozen_train():
    ev = []
    for w in s0i.TRAIN:
        for tool in s0i.TOOLS:
            ev.append({
                "world": w.id, "profile": w.profile, "revision": w.revision,
                "tool": tool, "marker_observed": (
                    tool == s0i.private_target(w.profile, w.revision)
                ),
            })
    return ev


def test_tasks_are_8_train_questions_not_original_pilot_or_confirm():
    tasks = s0j0.tasks()
    assert len(tasks) == 8
    assert len(set(t["id"] for t in tasks)) == 8
    assert {t["profile"] for t in tasks} == set(s0i.PROFILES)
    assert {t["question_index"] for t in tasks} == {0, 1}
    assert set(t["id"] for t in tasks).isdisjoint({w.id for w in s0i.PILOT + s0i.CONFIRM})
    assert all("r2" in question for question in s0j0.QUESTIONS)


def test_expected_answers_derive_only_from_observed_train():
    ev = events_from_frozen_train()
    answers = {p: s0j0.expected_answer(p, ev) for p in s0i.PROFILES}
    assert {p: choice for p, (choice, _) in answers.items()} == {
        "amber": "list_memories", "cedar": "vault_list",
        "indigo": "recall", "quiet": "none",
    }
    for p, (_, sources) in answers.items():
        assert len(sources) == 2 and len(set(sources)) == 2
        assert all("r2" not in source for source in sources)  # service labels do not embed the rule


def test_structured_card_is_authentic_complete_and_no_oracle_calls():
    ev = events_from_frozen_train()
    for profile in s0i.PROFILES:
        text = s0j0.evidence_card(profile, ev)
        assert text.count("service=") == 3
        assert "r1" in text and "r2" in text
        assert text.count("revision=r2") == 2
        assert "private_target" not in text
        for w in s0i.TRAIN:
            if w.profile == profile:
                assert w.service in text
            else:
                assert w.service not in text


def test_source_validation_must_support_answer_from_training():
    ev = events_from_frozen_train()
    for profile in s0i.PROFILES:
        tool, sources = s0j0.expected_answer(profile, ev)
        matched = s0j0.grade({"tool": tool, "evidence_service": sources[0]}, profile, ev)
        assert matched["correct_tool"] and matched["grounded"] and matched["source_valid"]
        wrong_source = s0j0.grade({"tool": tool, "evidence_service": "random service"}, profile, ev)
        assert wrong_source["correct_tool"] and not wrong_source["grounded"]
        invalid_tool = s0j0.grade({"tool": "unknown", "evidence_service": sources[0]}, profile, ev)
        assert not invalid_tool["correct_tool"] and not invalid_tool["grounded"]


def test_strict_json_contract():
    assert s0j0.parse_response('{"tool":"recall","evidence_service":"svc"}') is not None
    assert s0j0.parse_response('{"tool":"none","evidence_service":"svc"}') is not None
    assert s0j0.parse_response('{"tool":"unknown","evidence_service":"NONE"}') is not None
    assert s0j0.parse_response('{"tool":"recall"}') is None
    assert s0j0.parse_response('{"tool":"recall","evidence_service":"svc","extra":1}') is None
    assert s0j0.parse_response('not json') is None
    assert s0j0.parse_response('{"tool":"steal","evidence_service":"x"}') is None


def test_real_agentmemory_retrieval_contract_with_observed_mock_mcp(tmp_path):
    mem = AgentMemory(str(tmp_path / "s0j0.sqlite3"))
    calls = []
    def probe(tool):
        calls.append(tool)
        return "Synthetic source reply from completed tool"
    try:
        ev, stat = s0i.observed_training(mem, probe)
        assert len(ev) == 36 and len(calls) == 36 and stat["episodes"] == 12
        for profile in s0i.PROFILES:
            expected, sources = s0j0.expected_answer(profile, ev)
            for arm in s0j0.ARMS:
                input_text, metadata = s0j0.evidence_for_arm(arm, profile, mem, ev)
                assert isinstance(input_text, str) and len(input_text) > 15
                if arm == "N0":
                    assert metadata["episodes_in_text"] == 0
                    assert not metadata["retrieval_complete_r2"]
                elif arm == "N1":
                    assert metadata["episodes_in_text"] == 12
                    assert metadata["retrieval_complete_r2"]
                elif arm == "N3":
                    assert metadata["retrieval_complete_r2"]
                    assert metadata["episodes_in_text"] == 0
                else:
                    # Retrieval is a measured variable: not forcibly patched
                    # into oracle-derived cards when full-text search misses.
                    assert metadata["episodes_in_text"] <= 3
    finally:
        mem.close()


def fake_report(arm):
    ev = events_from_frozen_train()
    traces = []
    for t in s0j0.tasks():
        correct, services = s0j0.expected_answer(t["profile"], ev)
        parsed = {"tool": correct, "evidence_service": services[0]}
        grade = s0j0.grade(parsed, t["profile"], ev)
        traces.append({
            "id": t["id"], "profile": t["profile"],
            "question_index": t["question_index"], "arm": arm,
            "prompt_sha256": "a" * 64,
            "raw": json.dumps(parsed), "parsed": parsed, "valid_json": True,
            "prompt_tokens": 300, "completion_tokens": 20,
            "evidence_chars": 100, "episodes_in_text": 3,
            "retrieval_complete_r2": arm != "N0",
            "r2_services_in_input": services if arm != "N0" else [],
            **grade,
        })
    return {
        "study": s0j0.STUDY, "status": "COMPLETED_ARM", "arm": arm,
        "train": {"episodes": 12, "mcp_calls": 36, "sqlite_experiences": 12,
                  "versions": ["r1", "r2"]},
        "train_events": ev, "real_oauth_mcp_calls": 36, "external_connections_blocked": True,
        "model": {"real": True, "repo": MODEL_REPO, "file": MODEL_FILE,
                  "revision": MODEL_REVISION, "weights_sha256": MODEL_SHA256,
                  "backend": "llama-cpp-python", "version": "0.3.36",
                  "seed": 0, "temperature": 0, "calls": 8,
                  "prompt_tokens": 8*300, "completion_tokens": 8*20},
        "traces": traces,
    }


def test_all_four_arms_can_aggregate_using_original_train(tmp_path):
    paths = []
    for arm in s0j0.ARMS:
        path = tmp_path / arm / "report.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(fake_report(arm)))
        paths.append(path)
    result = s0j0.aggregate(paths)
    assert result["status"] == "COMPLETED"
    assert result["decision"] == "READ_COMPETENCE_SIGNAL"
    assert result["metrics"]["N2"]["correct_tool"] == 8
    assert result["metrics"]["N3"]["grounded"] == 8
    assert result["real_training_mcp_calls"] == 144
    assert result["S0I_PILOT_or_CONFIRM_used"] is False


@pytest.mark.parametrize("tamper", ["hash","events","tokens","source","json","arm","model"])
def test_aggregation_rejects_forgery_and_reports_invalid_json(tamper, tmp_path):
    report = fake_report("N2")
    row = report["traces"][0]
    if tamper == "hash":
        row["prompt_sha256"] = "bad"
    elif tamper == "events":
        report["train_events"].pop()
    elif tamper == "tokens":
        row["prompt_tokens"] = 1
    elif tamper == "source":
        row["grounded"] = False
    elif tamper == "json":
        row["raw"] = '{"bad":1}'
    elif tamper == "arm":
        row["arm"] = "N3"
    elif tamper == "model":
        report["model"]["weights_sha256"] = "0" * 64
    if tamper == "json":
        with pytest.raises(ValueError, match="Reported JSON parse"):
            s0j0.validate_arm(report)
    else:
        with pytest.raises(ValueError):
            s0j0.validate_arm(report)


def test_no_memory_arm_cannot_claim_source_is_present_in_input():
    # The scorer evaluates observed correct answers, but true evidence is
    # absent from N0; never count this arm as evidence-backed memory.
    ev = events_from_frozen_train()
    profile = "amber"
    text, meta = s0j0.evidence_for_arm("N0", profile, None, ev)
    assert "silver index" not in text
    assert meta["r2_services_in_input"] == []
