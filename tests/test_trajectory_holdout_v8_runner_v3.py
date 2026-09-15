"""V8 runner checks use synthetic inputs only; never replay the frozen V8 bank."""
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_infinito3_trajectory_holdout_v8 as runner
from src.infinito3.evaluation import EvaluationExpectation
from src.infinito3.trajectory_evaluation import TrajectoryScenario, TrajectoryStep


class FakeResponse:
    output_text = "42"
    model = "synthetic"
    id = "synthetic-v8"
    status = "completed"
    data = [SimpleNamespace(embedding=[1.0, 0.0])]

    def __init__(self, embedding=False):
        self.usage = {"total_tokens": 7, "prompt_tokens": 7} if embedding else {
            "input_tokens": 5, "output_tokens": 2, "total_tokens": 7,
        }

    def model_dump(self, **kwargs):
        return {"usage": self.usage, "status": self.status, "output_text": self.output_text}


def fake_client():
    return SimpleNamespace(
        responses=SimpleNamespace(create=lambda **kw: FakeResponse()),
        embeddings=SimpleNamespace(create=lambda **kw: FakeResponse(embedding=True)),
    )


def test_v8_preflight_accepts_frozen_manifest():
    manifest = runner.preflight()
    assert manifest["evaluation_version"] == "V8"
    assert manifest["suite_git_blob_sha1"] == runner.SUITE_BLOB_SHA1
    assert manifest["config"] == runner.CONFIG


def test_v8_default_command_is_preflight_only(monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda: {"synthetic": True})
    monkeypatch.setattr(runner, "TrajectoryEvaluationHarness",
                        lambda *a, **k: pytest.fail("No V8 replay permitted"))
    assert runner.main([]) == 0


def test_v8_preflight_rejects_modified_frozen_blob(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    (tmp_path / "candidate.py").write_text("modified", encoding="utf-8")
    manifest = {
        "config": runner.CONFIG,
        "git_blob_sha1": {"candidate.py": "wrong"},
    }
    (tmp_path / "docs/INFINITO_3_V8_RUNNER_MANIFEST.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(SystemExit, match="Frozen file changed"):
        runner.preflight()


def test_v8_live_mode_requires_exact_revision(monkeypatch):
    monkeypatch.setattr(runner, "preflight", lambda: {})
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *a, **k: "actual\n")
    with pytest.raises(SystemExit, match="Exact --expected-revision"):
        runner.main(["--authorize-live-v8", "--expected-revision", "other"])


def test_v8_main_writes_synthetic_evidence_without_real_bank(tmp_path, monkeypatch):
    scenario = TrajectoryScenario(name="synthetic_v8", steps=(TrajectoryStep(
        "Calculate 6 * 7", label="synthetic v8",
        tags=("empty_context",),
        expectation=EvaluationExpectation(answer_contains=("42",)),
    ),))
    monkeypatch.setitem(sys.modules, "src.infinito3.trajectory_holdout_v8_cases", SimpleNamespace(
        independent_trajectory_holdout_v8_suite=lambda: (scenario,)
    ))
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=lambda **kw: fake_client()))
    monkeypatch.setattr(runner, "preflight", lambda: {"synthetic": True})
    monkeypatch.setattr(runner, "version", lambda name: runner.CONFIG["openai_sdk_version"])
    monkeypatch.setattr(
        runner.subprocess, "check_output",
        lambda args, **kw: "synthetic-sha\n" if "rev-parse" in args else "",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unused-key")
    output = tmp_path / "v8-result"
    assert runner.main([
        "--authorize-live-v8", "--expected-revision", "synthetic-sha",
        "--output-dir", str(output),
    ]) == 0
    report = json.loads((output / "report.json").read_text())
    assert report["metadata"]["evaluation_version"] == "V8"
    assert report["metadata"]["first_v8_execution"] is True
    assert report["metadata"]["post_v7_diagnostic_holdout"] is True
    review = json.loads((output / "audit-review.json").read_text())
    assert review["integration_status"] == "NOT_APPROVED"
    assert review["requires_manual_mutation_boundary_review"] is True
    assert "synthetic-unused-key" not in (output / "provider-and-turn-audit.jsonl").read_text()
    with pytest.raises(FileExistsError):
        runner.main([
            "--authorize-live-v8", "--expected-revision", "synthetic-sha",
            "--output-dir", str(output),
        ])
