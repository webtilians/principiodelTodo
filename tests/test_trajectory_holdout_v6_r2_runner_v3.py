"""Runner checks with synthetic data ONLY; never import/replay V6 cases."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_infinito3_trajectory_holdout_v6_r2 as runner
from src.infinito3.evaluation import EvaluationExpectation
from src.infinito3.trajectory_evaluation import TrajectoryEvaluationHarness, TrajectoryScenario, TrajectoryStep


class FakeResponse:
    def __init__(self, embedding=False, status="completed"):
        self.output_text = "42"
        self.model = "synthetic"
        self.id = "synthetic-id"
        self.status = status
        self.data = [SimpleNamespace(embedding=[1.0, 0.0])]
        self.usage = {"total_tokens": 7, "prompt_tokens": 7} if embedding else {
            "input_tokens": 5, "output_tokens": 2, "total_tokens": 7}

    def model_dump(self, **kwargs):
        return {"usage": self.usage, "status": self.status, "output_text": self.output_text}


def fake_client():
    return SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: FakeResponse()),
                           embeddings=SimpleNamespace(create=lambda **kw: FakeResponse(embedding=True)))


def test_provider_categories_include_embedding_tokens(tmp_path):
    audit = runner.AuditLog(tmp_path / "audit.jsonl")
    audit.client(fake_client(), "embeddings", "synthetic").embeddings.create(input=["example"])
    audit.client(fake_client(), "events", "synthetic").responses.create(input=[])
    assert audit.usage["embeddings"] == {"calls": 1, "total_tokens": 7, "input_tokens": 7, "output_tokens": 0}
    assert audit.usage["events"]["total_tokens"] == 7
    records = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert [r["kind"] for r in records] == ["request", "response", "request", "response"]
    assert records[0]["payload"]["input"] == ["example"]


def test_provider_failure_preserves_request_without_secret_exception(tmp_path):
    def fail(**kwargs):
        raise RuntimeError("SECRET_MUST_NOT_BE_LOGGED")
    client = SimpleNamespace(responses=SimpleNamespace(create=fail))
    audit = runner.AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(SystemExit, match="Provider failure"):
        audit.client(client, "events", "synthetic").responses.create(input=[])
    assert "provider_error" in audit.path.read_text()
    assert "SECRET_MUST_NOT_BE_LOGGED" not in audit.path.read_text()
    assert audit.calls == 1


def test_call_ceiling_stops_before_next_request(tmp_path):
    audit = runner.AuditLog(tmp_path / "audit.jsonl", max_calls=1)
    endpoint = audit.client(fake_client(), "baseline_answers", "synthetic").responses
    endpoint.create(input=[])
    with pytest.raises(SystemExit, match="ceiling"):
        endpoint.create(input=[])
    assert audit.calls == 1


def test_incomplete_response_is_not_scored(tmp_path):
    client = SimpleNamespace(responses=SimpleNamespace(create=lambda **kw: FakeResponse(status="incomplete")))
    audit = runner.AuditLog(tmp_path / "audit.jsonl")
    with pytest.raises(SystemExit, match="Incomplete"):
        audit.client(client, "cognitive_answers", "synthetic").responses.create(input=[])
    assert audit.usage["cognitive_answers"]["total_tokens"] == 7


def test_length_limited_answer_continues_with_explicit_warning(tmp_path):
    class LengthResponse(FakeResponse):
        def model_dump(self, **kw):
            return {**super().model_dump(**kw), "incomplete_details": {"reason": "max_output_tokens"}}
    received = []
    def create(**payload):
        received.append(payload)
        return LengthResponse(status="incomplete")
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    audit = runner.AuditLog(tmp_path / "audit.jsonl")
    for category in ("baseline_answers", "cognitive_answers"):
        audit.client(client, category, "synthetic").responses.create(input=[], max_output_tokens=96)
    assert [p["max_output_tokens"] for p in received] == [1024, 1024]
    assert len(audit.truncated_answers) == 2
    with pytest.raises(SystemExit, match="Incomplete"):
        audit.client(client, "events", "synthetic").responses.create(input=[], max_output_tokens=260)
    assert received[-1]["max_output_tokens"] == 2048


def test_synthetic_full_runner_wiring_and_fail_closed_audit(tmp_path):
    audit = runner.AuditLog(tmp_path / "audit.jsonl")
    engines = []
    scenario = TrajectoryScenario(name="synthetic_arithmetic", steps=tuple(
        TrajectoryStep("6 * 7", label=f"synthetic {i}",
            expectation=EvaluationExpectation(answer_contains=("42",)), tags=("empty_context",))
        for i in range(4)))
    report = TrajectoryEvaluationHarness(runner.loop_factory(fake_client(), audit, engines)).run((scenario,))
    review = runner.audit_report(report)
    assert review["empty_context_gate"] is True
    assert review["integration_status"] == "NOT_APPROVED"
    assert all(p["semantic_baseline_win"] is None for p in review["probe_reviews"])
    assert audit.usage["baseline_answers"]["calls"] == 4
    assert audit.usage["cognitive_answers"]["calls"] == 4
    assert audit.usage["embeddings"]["total_tokens"] > 0
    assert all(type(engine.context_builder).__name__ == "IntentContextBuilder" for _, _, engine in engines)
    report.results[0].steps[0].cognitive.cognitive_decision.context_packet.items.append("irrelevant")
    assert runner.audit_report(report)["empty_context_gate"] is False
    report.results[0].steps[0].cognitive.cognitive_decision.context_packet = None
    assert runner.audit_report(report)["empty_context_gate"] is False
    records = [json.loads(line) for line in audit.path.read_text().splitlines()]
    assert sum(r["kind"] == "turn_result" for r in records) == 8
    for _, _, engine in engines:
        engine.memory_store.close()


def test_default_command_does_not_load_or_replay_bank(monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda: {})
    monkeypatch.setattr(runner, "loop_factory", lambda *a: pytest.fail("No replay permitted"))
    assert runner.main([]) == 0


def test_preflight_rejects_modified_frozen_file(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "candidate.py").write_text("modified")
    (tmp_path / "docs/INFINITO_3_V6_R2_MANIFEST.json").write_text(json.dumps({
        "sha256": {"candidate.py": "wrong"}, "config": runner.CONFIG}))
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="Frozen file changed"):
        runner.preflight()


def test_main_writes_complete_evidence_using_substitute_synthetic_suite(tmp_path, monkeypatch):
    scenario = TrajectoryScenario(name="synthetic_main", steps=(TrajectoryStep(
        "Calculate 6 * 7", label="synthetic main",
        expectation=EvaluationExpectation(answer_contains=("42",), context_excludes=("irrelevant",))),))
    # Replace the module BEFORE main's guarded import. No real V6 source is loaded.
    monkeypatch.setitem(sys.modules, "src.infinito3.trajectory_holdout_v6_cases", SimpleNamespace(
        independent_trajectory_holdout_v6_suite=lambda: (scenario,)))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: fake_client()))
    monkeypatch.setattr(runner, "preflight", lambda: {"synthetic_test": True})
    monkeypatch.setattr(runner, "version", lambda name: runner.CONFIG["openai_sdk_version"])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda args, **kw: "synthetic-sha\n" if "rev-parse" in args else "")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unused-credential")
    output = tmp_path / "result"
    assert runner.main(["--authorize-live-v6", "--expected-revision", "synthetic-sha",
                        "--output-dir", str(output)]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["summary"]["user_turn_count"] == 1
    assert report["metadata"]["total_provider_tokens"] == 14
    assert len(report["metadata"]["final_states"]) == 2
    assert json.loads((output / "audit-review.json").read_text())["integration_status"] == "NOT_APPROVED"
    assert "synthetic-unused-credential" not in (output / "provider-and-turn-audit.jsonl").read_text()
    with pytest.raises(FileExistsError):
        runner.main(["--authorize-live-v6", "--expected-revision", "synthetic-sha", "--output-dir", str(output)])
