#!/usr/bin/env python3
"""Manual-only V9 runner. Preflight never imports or replays the frozen V9 bank."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_infinito3_trajectory_holdout_v6_r2 as audited
from src.infinito3.trajectory_evaluation import TrajectoryEvaluationHarness, _jsonable, _avg

SUITE_BLOB_SHA1 = "25d5857cd8ec8f60cb8df5e1007f46e1ba1e9453"
CASE_COMMIT = "9fcb205714cf6d4ce2b648a5c3e87cf4e7192e44"
V8_DEVELOPMENT_RUN = 34995401203
CONFIG = {
    **audited.CONFIG,
    "protocol_revision": "v9_r1_grounding_goalcanon",
}


def dump(value):
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _git_blob_sha1(path):
    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def preflight():
    manifest = json.loads((ROOT / "docs/INFINITO_3_V9_RUNNER_MANIFEST.json").read_text())
    if manifest["config"] != CONFIG:
        raise SystemExit("Frozen model/scorer configuration changed")
    for name, expected in manifest["git_blob_sha1"].items():
        if _git_blob_sha1(ROOT / name) != expected:
            raise SystemExit(f"Frozen file changed: {name}")
    suite = ROOT / "src/infinito3/trajectory_holdout_v9_cases.py"
    if _git_blob_sha1(suite) != SUITE_BLOB_SHA1:
        raise SystemExit("Frozen V9 suite mismatch")
    return manifest


def audit_report(report):
    review = audited.audit_report(report)
    review["evaluation_version"] = "V9"
    review["post_v8_diagnostic_holdout"] = True
    review["requires_manual_grounding_review"] = True
    review["requires_manual_temporal_canonicalization_review"] = True
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-live-v9", action="store_true")
    parser.add_argument("--expected-revision")
    parser.add_argument("--output-dir", default="trajectory-holdout-v9-results")
    args = parser.parse_args(argv)

    manifest = preflight()
    if not args.authorize_live_v9:
        print("Preflight passed. No V9 turns imported or replayed; no provider calls. Live authorization required.")
        return 0

    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if args.expected_revision != revision:
        raise SystemExit("Exact --expected-revision required")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise SystemExit("Live run requires a clean worktree")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is missing")
    if version("openai") != CONFIG["openai_sdk_version"]:
        raise SystemExit("OpenAI SDK version differs from frozen configuration")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    audit = audited.AuditLog(output / "provider-and-turn-audit.jsonl", CONFIG["max_provider_calls"])
    audit.write(
        "run_start",
        evaluation_version="V9",
        revision=revision,
        manifest=manifest,
        case_commit=CASE_COMMIT,
        v8_development_run=V8_DEVELOPMENT_RUN,
        python_version=sys.version,
        sdk_version=version("openai"),
    )
    engines = []
    previous_config = audited.CONFIG
    audited.CONFIG = CONFIG
    try:
        from openai import OpenAI
        from src.infinito3.trajectory_holdout_v9_cases import independent_trajectory_holdout_v9_suite

        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=CONFIG["sdk_max_retries"],
            timeout=CONFIG["timeout_seconds"],
        )
        report = TrajectoryEvaluationHarness(
            audited.loop_factory(client, audit, engines),
            win_epsilon=CONFIG["win_epsilon"],
        ).run(independent_trajectory_holdout_v9_suite())
        report.metadata.update({
            "evaluation_version": "V9",
            "runner_revision": revision,
            "manifest": manifest,
            "suite_git_blob_sha1": SUITE_BLOB_SHA1,
            "case_commit": CASE_COMMIT,
            "provider_usage_by_category": audit.usage,
            "monetary_cost": None,
            "protocol_revision": CONFIG["protocol_revision"],
            "v8_development_run": V8_DEVELOPMENT_RUN,
            "first_v9_execution": True,
            "post_v8_diagnostic_holdout": True,
            "truncated_answers": audit.truncated_answers,
            "total_provider_tokens": sum(u["total_tokens"] for u in audit.usage.values()),
            "per_trajectory_scores": [{
                "scenario": result.scenario.name,
                "baseline_answer_mean": _avg(p.baseline_metrics.answer_score for p in result.probes),
                "cognitive_answer_mean": _avg(p.cognitive_metrics.answer_score for p in result.probes),
                "context_mean": _avg(p.cognitive_metrics.context_score for p in result.probes),
            } for result in report.results],
            "cost_note": "Token usage includes embeddings. No frozen price table; monetary cost not estimated.",
            "final_states": [{
                "scenario": name,
                "arm": arm,
                "goals": _jsonable(engine.goal_engine.all()),
                "events": _jsonable(engine.temporal_state.events()),
                "event_stats": engine.event_extractor.stats(),
            } for name, arm, engine in engines],
        })
        (output / "report.json").write_text(report.to_json(), encoding="utf-8")
        (output / "report.md").write_text(report.to_markdown(), encoding="utf-8")
        review = audit_report(report)
        review["answer_completion_gate"] = not audit.truncated_answers
        review["truncated_answers"] = audit.truncated_answers
        (output / "audit-review.json").write_text(dump(review), encoding="utf-8")
        audit.write("run_completed", usage=audit.usage, integration_status="NOT_APPROVED")
    except BaseException as exc:
        audit.write("run_incomplete", error_type=type(exc).__name__, usage=audit.usage)
        raise
    finally:
        audited.CONFIG = previous_config
        for _, _, engine in engines:
            engine.memory_store.close()

    print("V9 completed. Manual semantic/final-state audit required. No integration authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
