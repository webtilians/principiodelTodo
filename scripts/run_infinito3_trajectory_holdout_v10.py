#!/usr/bin/env python3
"""Manual-only V10 runner. Preflight never imports or replays the frozen V10 bank."""
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
from src.infinito3.planned_context import PlannedIntentContextBuilder
from src.infinito3.planned_loop import PlannedLiteralGroundedCognitiveLoop
from src.infinito3.trajectory_evaluation import TrajectoryEvaluationHarness, _jsonable, _avg

SUITE_BLOB_SHA1 = "fea6f27c3f730f1797f652b6cc82f249a3663e11"
CASE_COMMIT = "83e352fe63dc4af74d2ba4bfe8b805c10fcb5af8"
V9_DEVELOPMENT_RUN = 35000202827
CONFIG = {
    **audited.CONFIG,
    "protocol_revision": "v10_r1_typed_query_planner",
}


def dump(value):
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def _git_blob_sha1(path):
    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def preflight():
    manifest = json.loads((ROOT / "docs/INFINITO_3_V10_RUNNER_MANIFEST.json").read_text())
    if manifest["config"] != CONFIG:
        raise SystemExit("Frozen model/scorer configuration changed")
    for name, expected in manifest["git_blob_sha1"].items():
        if _git_blob_sha1(ROOT / name) != expected:
            raise SystemExit(f"Frozen file changed: {name}")
    suite = ROOT / "src/infinito3/trajectory_holdout_v10_cases.py"
    if _git_blob_sha1(suite) != SUITE_BLOB_SHA1:
        raise SystemExit("Frozen V10 suite mismatch")
    return manifest


class AuditedPlannedLoop(PlannedLiteralGroundedCognitiveLoop):
    def __init__(self, *args, audit, scenario, arm, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit, self.scenario, self.arm = audit, scenario, arm
        self.turn_number = 0

    def turn(self, text, **kwargs):
        self.turn_number += 1
        self.audit.write(
            "turn_start", scenario=self.scenario, arm=self.arm,
            turn=self.turn_number, user_text=text,
        )
        result = super().turn(text, **kwargs)
        if result.blocked or result.response is None:
            raise SystemExit("Loop error; run incomplete")
        self.audit.write(
            "turn_result", scenario=self.scenario, arm=self.arm,
            turn=self.turn_number, result=result,
            events=self.engine.temporal_state.events(),
            goals=self.engine.goal_engine.all(),
        )
        return result


def loop_factory(client, audit, engines):
    def factory(clock, scenario):
        def make(arm):
            def adapter(category, model, reasoning=None):
                return audited.OpenAIResponsesAdapter(
                    audit.client(client, category, scenario.name),
                    model=model, reasoning_effort=reasoning,
                )

            store = audited.SemanticTemporalMemoryStore(
                path=":memory:",
                embedding_provider=audited.OpenAIEmbeddingProvider(
                    audit.client(client, "embeddings", scenario.name),
                    model=CONFIG["embedding_model"],
                ),
            )
            goals = audited.TemporalGoalEngine(now_fn=clock)
            builder = PlannedIntentContextBuilder(
                memory_store=store,
                goal_engine=goals,
                now_fn=clock,
                reranker=audited.LLMSemanticMembershipReranker(
                    adapter("reranker", CONFIG["reranker_model"]),
                    max_output_tokens=CONFIG["reranker_max_output_tokens"],
                ),
            )
            extractor = audited.IntentEventExtractor(
                adapter("events", CONFIG["event_model"]),
                now_fn=clock,
                max_output_tokens=CONFIG["event_max_output_tokens"],
                min_confidence=CONFIG["event_min_confidence"],
            )
            engine = audited.CognitiveEngine(
                memory_store=store,
                goal_engine=goals,
                context_builder=builder,
                event_extractor=extractor,
                temporal_state=audited.SemanticTemporalCognitiveState(now_fn=clock),
            )
            engines.append((scenario.name, arm, engine))
            return AuditedPlannedLoop(
                engine,
                adapter(arm + "_answers", CONFIG["answer_model"], CONFIG["answer_reasoning_effort"]),
                history_limit=scenario.history_limit,
                audit=audit,
                scenario=scenario.name,
                arm=arm,
            )
        return make("baseline"), make("cognitive")
    return factory


def audit_report(report):
    review = audited.audit_report(report)
    review["evaluation_version"] = "V10"
    review["post_v9_diagnostic_holdout"] = True
    review["requires_manual_planner_review"] = True
    review["requires_manual_literal_fastpath_review"] = True
    review["requires_manual_temporal_review"] = True
    return review


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-live-v10", action="store_true")
    parser.add_argument("--expected-revision")
    parser.add_argument("--output-dir", default="trajectory-holdout-v10-results")
    args = parser.parse_args(argv)

    manifest = preflight()
    if not args.authorize_live_v10:
        print("Preflight passed. No V10 turns imported or replayed; no provider calls. Live authorization required.")
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
        evaluation_version="V10",
        revision=revision,
        manifest=manifest,
        case_commit=CASE_COMMIT,
        v9_development_run=V9_DEVELOPMENT_RUN,
        python_version=sys.version,
        sdk_version=version("openai"),
    )
    engines = []
    previous_config = audited.CONFIG
    audited.CONFIG = CONFIG
    try:
        from openai import OpenAI
        from src.infinito3.trajectory_holdout_v10_cases import independent_trajectory_holdout_v10_suite

        client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            max_retries=CONFIG["sdk_max_retries"],
            timeout=CONFIG["timeout_seconds"],
        )
        report = TrajectoryEvaluationHarness(
            loop_factory(client, audit, engines),
            win_epsilon=CONFIG["win_epsilon"],
        ).run(independent_trajectory_holdout_v10_suite())
        report.metadata.update({
            "evaluation_version": "V10",
            "runner_revision": revision,
            "manifest": manifest,
            "suite_git_blob_sha1": SUITE_BLOB_SHA1,
            "case_commit": CASE_COMMIT,
            "provider_usage_by_category": audit.usage,
            "monetary_cost": None,
            "protocol_revision": CONFIG["protocol_revision"],
            "v9_development_run": V9_DEVELOPMENT_RUN,
            "first_v10_execution": True,
            "post_v9_diagnostic_holdout": True,
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

    print("V10 completed. Manual semantic/planner/final-state audit required. No integration authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
