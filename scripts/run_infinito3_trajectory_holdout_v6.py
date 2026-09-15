#!/usr/bin/env python3
"""Manual-only V6 runner. Importing/preflight never replays the held-out bank."""
import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.infinito3.cognitive_loop import CognitiveLoop
from src.infinito3.engine import CognitiveEngine
from src.infinito3.intent_context import IntentContextBuilder, IntentEventExtractor
from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.persistent_memory import OpenAIEmbeddingProvider
from src.infinito3.semantic_reranker import LLMSemanticMembershipReranker
from src.infinito3.semantic_temporal_memory import SemanticTemporalMemoryStore
from src.infinito3.semantic_temporal_state import SemanticTemporalCognitiveState
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.trajectory_evaluation import TrajectoryEvaluationHarness, _jsonable, _avg

SUITE_SHA = "19c645820733bcf0b1a9f6341f656fe5b6a02fa653770982a87b372170265c1a"
CASE_COMMIT = "efa63dcf80ce85104072a01c5fcdc8fc3feb2dbd"
CONFIG = {
    "answer_model": "gpt-5.6-luna", "answer_reasoning_effort": "none",
    "event_model": "gpt-5.6-luna", "event_max_output_tokens": 260,
    "event_min_confidence": 0.72, "reranker_model": "gpt-5.6-luna",
    "reranker_max_output_tokens": 160, "embedding_model": "text-embedding-3-small",
    "win_epsilon": 0.05, "answer_gate": 0.85, "context_gate": 0.90,
    "max_provider_calls": 1500, "sdk_max_retries": 0, "timeout_seconds": 90,
    "openai_sdk_version": "3.14.0",
}


def dump(value):
    return json.dumps(_jsonable(value), ensure_ascii=False, sort_keys=True)


def preflight():
    manifest = json.loads((ROOT / "docs/INFINITO_3_V6_RUNNER_MANIFEST.json").read_text())
    for name, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != expected:
            raise SystemExit(f"Frozen file changed: {name}")
    if manifest["config"] != CONFIG:
        raise SystemExit("Frozen model/scorer configuration changed")
    if manifest["sha256"]["src/infinito3/trajectory_holdout_v6_cases.py"] != SUITE_SHA:
        raise SystemExit("Frozen V6 suite mismatch")
    return manifest


class AuditLog:
    """Append before/after every request; retain partial evidence after failure."""
    def __init__(self, path, max_calls=1500):
        self.path = Path(path)
        self.calls = 0
        self.max_calls = max_calls
        self.usage = {}

    def write(self, kind, **data):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(dump({"kind": kind, **data}) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def client(self, client, category, scenario):
        def endpoint(name):
            def create(**payload):
                # BaseException deliberately escapes core's best-effort fallbacks.
                if self.calls >= self.max_calls:
                    raise SystemExit("Provider-call ceiling reached; run incomplete")
                self.calls += 1
                call_id = self.calls
                self.write("request", call_id=call_id, category=category,
                           scenario=scenario, endpoint=name, payload=payload)
                try:
                    response = getattr(client, name).create(**payload)
                except Exception as exc:
                    # Never serialize exception messages/headers (may contain secrets).
                    self.write("provider_error", call_id=call_id, error_type=type(exc).__name__)
                    raise SystemExit("Provider failure; preserve partial run, no automatic retry") from None
                raw = response.model_dump(mode="json")
                self.write("response", call_id=call_id, response=raw)
                usage = raw.get("usage")
                if not isinstance(usage, dict) or "total_tokens" not in usage:
                    raise SystemExit("Missing provider usage; run incomplete")
                totals = self.usage.setdefault(category, {"calls": 0, "total_tokens": 0,
                                                          "input_tokens": 0, "output_tokens": 0})
                totals["calls"] += 1
                totals["total_tokens"] += int(usage["total_tokens"])
                totals["input_tokens"] += int(usage.get("input_tokens", usage.get("prompt_tokens", 0)))
                totals["output_tokens"] += int(usage.get("output_tokens", 0))
                if name == "responses" and raw.get("status") != "completed":
                    raise SystemExit("Incomplete model response; run incomplete")
                return response
            return SimpleNamespace(create=create)
        return SimpleNamespace(responses=endpoint("responses"), embeddings=endpoint("embeddings"))


class AuditedLoop(CognitiveLoop):
    def __init__(self, *args, audit, scenario, arm, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit, self.scenario, self.arm = audit, scenario, arm
        self.turn_number = 0

    def turn(self, text, **kwargs):
        self.turn_number += 1
        self.audit.write("turn_start", scenario=self.scenario, arm=self.arm,
                         turn=self.turn_number, user_text=text)
        result = super().turn(text, **kwargs)
        if result.blocked or result.response is None:
            raise SystemExit("Loop error; run incomplete")
        self.audit.write("turn_result", scenario=self.scenario, arm=self.arm,
                         turn=self.turn_number, result=result,
                         events=self.engine.temporal_state.events(),
                         goals=self.engine.goal_engine.all())
        return result


def loop_factory(client, audit, engines):
    def factory(clock, scenario):
        def make(arm):
            def adapter(category, model, reasoning=None):
                return OpenAIResponsesAdapter(audit.client(client, category, scenario.name),
                                              model=model, reasoning_effort=reasoning)
            store = SemanticTemporalMemoryStore(path=":memory:", embedding_provider=OpenAIEmbeddingProvider(
                audit.client(client, "embeddings", scenario.name), model=CONFIG["embedding_model"]))
            goals = TemporalGoalEngine(now_fn=clock)
            builder = IntentContextBuilder(memory_store=store, goal_engine=goals, now_fn=clock,
                reranker=LLMSemanticMembershipReranker(adapter("reranker", CONFIG["reranker_model"]),
                    max_output_tokens=CONFIG["reranker_max_output_tokens"]))
            extractor = IntentEventExtractor(adapter("events", CONFIG["event_model"]), now_fn=clock,
                max_output_tokens=CONFIG["event_max_output_tokens"], min_confidence=CONFIG["event_min_confidence"])
            engine = CognitiveEngine(memory_store=store, goal_engine=goals, context_builder=builder,
                event_extractor=extractor, temporal_state=SemanticTemporalCognitiveState(now_fn=clock))
            engines.append((scenario.name, arm, engine))
            return AuditedLoop(engine, adapter(arm + "_answers", CONFIG["answer_model"],
                CONFIG["answer_reasoning_effort"]), history_limit=scenario.history_limit,
                audit=audit, scenario=scenario.name, arm=arm)
        return make("baseline"), make("cognitive")
    return factory


def audit_report(report):
    """Strict machine checks plus an unfilled, probe-by-probe semantic review."""
    controls, reviews = [], []
    for trajectory in report.results:
        for step in trajectory.probes:
            tags = trajectory.scenario.steps[step.index].tags
            decision = step.cognitive.cognitive_decision
            packet = decision.context_packet if decision else None
            if "empty_context" in tags:
                controls.append({"scenario": trajectory.scenario.name, "label": step.label,
                                 "passed": packet is not None and packet.items == []})
            reviews.append({"scenario": trajectory.scenario.name, "label": step.label,
                            "tags": tags, "baseline_answer": step.baseline.response.text,
                            "cognitive_answer": step.cognitive.response.text,
                            "semantic_baseline_win": None, "time_alias_review": None,
                            "closure_review": None, "literal_data_review": None,
                            "reviewer": None, "rationale": None})
    s = report.summary
    return {"answer_gate": s.mean_cognitive_answer_score is not None and s.mean_cognitive_answer_score >= CONFIG["answer_gate"],
            "context_gate": s.mean_context_score is not None and s.mean_context_score >= CONFIG["context_gate"],
            "empty_context_gate": len(controls) == 4 and all(c["passed"] for c in controls),
            "empty_context_controls": controls, "probe_reviews": reviews,
            "semantic_baseline_wins_gate": "pending_manual_review",
            "no_stale_final_goals_gate": "pending_manual_review",
            "integration_status": "NOT_APPROVED", "automatic_merge": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorize-live-v6", action="store_true")
    parser.add_argument("--expected-revision")
    parser.add_argument("--output-dir", default="trajectory-holdout-v6-results")
    args = parser.parse_args(argv)
    manifest = preflight()
    if not args.authorize_live_v6:
        print("Preflight passed. No V6 turns replayed; no provider calls. Live authorization required.")
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
    audit = AuditLog(output / "provider-and-turn-audit.jsonl", CONFIG["max_provider_calls"])
    audit.write("run_start", revision=revision, manifest=manifest, case_commit=CASE_COMMIT,
                python_version=sys.version, sdk_version=version("openai"))
    engines = []
    try:
        from openai import OpenAI
        from src.infinito3.trajectory_holdout_v6_cases import independent_trajectory_holdout_v6_suite
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], max_retries=0, timeout=CONFIG["timeout_seconds"])
        report = TrajectoryEvaluationHarness(loop_factory(client, audit, engines),
            win_epsilon=CONFIG["win_epsilon"]).run(independent_trajectory_holdout_v6_suite())
        report.metadata.update({"runner_revision": revision, "manifest": manifest, "suite_sha256": SUITE_SHA,
            "provider_usage_by_category": audit.usage, "monetary_cost": None,
            "total_provider_tokens": sum(u["total_tokens"] for u in audit.usage.values()),
            "per_trajectory_scores": [{"scenario": r.scenario.name,
                "baseline_answer_mean": _avg(p.baseline_metrics.answer_score for p in r.probes),
                "cognitive_answer_mean": _avg(p.cognitive_metrics.answer_score for p in r.probes),
                "context_mean": _avg(p.cognitive_metrics.context_score for p in r.probes)} for r in report.results],
            "cost_note": "Token usage includes embeddings. No frozen price table; monetary cost not estimated.",
            "final_states": [{"scenario": name, "arm": arm, "goals": _jsonable(engine.goal_engine.all()),
                              "events": _jsonable(engine.temporal_state.events()),
                              "event_stats": engine.event_extractor.stats()}
                             for name, arm, engine in engines]})
        (output / "report.json").write_text(report.to_json(), encoding="utf-8")
        (output / "report.md").write_text(report.to_markdown(), encoding="utf-8")
        (output / "audit-review.json").write_text(dump(audit_report(report)), encoding="utf-8")
        audit.write("run_completed", usage=audit.usage, integration_status="NOT_APPROVED")
    except BaseException as exc:
        audit.write("run_incomplete", error_type=type(exc).__name__, usage=audit.usage)
        raise
    finally:
        for _, _, engine in engines:
            engine.memory_store.close()
    print("V6 completed. Manual semantic/final-state audit required. No integration authorized.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
