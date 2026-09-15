#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from src.infinito3.cognitive_loop import CognitiveLoop
from src.infinito3.engine import CognitiveEngine
from src.infinito3.evaluation import DeterministicEvaluator
from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.persistent_memory import OpenAIEmbeddingProvider
from src.infinito3.resolved_temporal_memory import ResolvedSemanticTemporalMemoryStore
from src.infinito3.resolved_temporal_state import ResolvedSemanticTemporalState
from src.infinito3.semantic_event_extractor import SemanticCognitiveEventExtractor
from src.infinito3.semantic_reranker import LLMSemanticMembershipReranker
from src.infinito3.structured_temporal_context import StructuredTemporalContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.trajectory_evaluation import MutableClock
from src.infinito3.trajectory_holdout_v4_cases import independent_trajectory_holdout_v4_suite

FROZEN_CASE_COMMIT = "ef689e8adece7aa265c236ac9f50db0ebac45469"
FROZEN_SUITE_SHA256 = "8825f79527a36da808915f6b085fcee349e808da98879c8947ed807df5390979"
CANDIDATE_IMPLEMENTATION_COMMIT = "f57a1ce48dd3ed9e35c057883ab6a132fa1a8f33"
TARGET_SCENARIOS = {
    "v4_natural_commitments_and_lifecycle",
    "v4_preference_language_without_like_verbs",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run targeted frozen V4 regression on resolved temporal state.")
    parser.add_argument("--model", default="gpt-5.6-luna")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--event-model", default="gpt-5.6-luna")
    parser.add_argument("--reranker-model", default="gpt-5.6-luna")
    parser.add_argument("--output-dir", default="v4-resolved-targeted-results")
    return parser.parse_args()


def usage_int(usage, key):
    try:
        return int((usage or {}).get(key) or 0)
    except (TypeError, ValueError, AttributeError):
        return 0


def main():
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")

    case_path = REPO_ROOT / "src" / "infinito3" / "trajectory_holdout_v4_cases.py"
    suite_sha = hashlib.sha256(case_path.read_bytes()).hexdigest()
    if suite_sha != FROZEN_SUITE_SHA256:
        raise SystemExit(f"Frozen V4 suite hash mismatch: {suite_sha}")

    all_scenarios = independent_trajectory_holdout_v4_suite()
    scenarios = [scenario for scenario in all_scenarios if scenario.name in TARGET_SCENARIOS]
    if {scenario.name for scenario in scenarios} != TARGET_SCENARIOS:
        raise SystemExit("Target V4 scenarios are missing from frozen suite")

    client = OpenAI(api_key=api_key)
    evaluator = DeterministicEvaluator()
    results = []
    totals = {
        "user_turns": 0,
        "probes": 0,
        "answer_provider_tokens": 0,
        "context_reranker_calls": 0,
        "context_reranker_tokens": 0,
        "event_calls": 0,
        "event_tokens": 0,
        "event_emitted_events": 0,
        "event_skipped_non_mutating": 0,
        "event_semantic_reviews": 0,
        "retraction_resolver_calls": 0,
        "retraction_resolver_tokens": 0,
        "retraction_resolved": 0,
        "final_active_memories": 0,
        "final_open_goals": 0,
    }

    for scenario in scenarios:
        clock = MutableClock(scenario.start_at)
        store = ResolvedSemanticTemporalMemoryStore(
            path=":memory:",
            embedding_provider=OpenAIEmbeddingProvider(client, model=args.embedding_model),
        )
        goals = TemporalGoalEngine(now_fn=clock)
        context_reranker = LLMSemanticMembershipReranker(
            OpenAIResponsesAdapter(client, model=args.reranker_model, reasoning_effort=None),
            max_output_tokens=160,
        )
        retraction_reranker = LLMSemanticMembershipReranker(
            OpenAIResponsesAdapter(client, model=args.reranker_model, reasoning_effort=None),
            max_output_tokens=160,
        )
        builder = StructuredTemporalContextBuilder(
            memory_store=store,
            goal_engine=goals,
            now_fn=clock,
            reranker=context_reranker,
        )
        extractor = SemanticCognitiveEventExtractor(
            OpenAIResponsesAdapter(client, model=args.event_model, reasoning_effort=None),
            now_fn=clock,
            max_output_tokens=260,
            min_confidence=0.72,
        )
        state = ResolvedSemanticTemporalState(
            now_fn=clock,
            retraction_reranker=retraction_reranker,
        )
        engine = CognitiveEngine(
            memory_store=store,
            goal_engine=goals,
            context_builder=builder,
            event_extractor=extractor,
            temporal_state=state,
        )
        loop = CognitiveLoop(
            engine,
            OpenAIResponsesAdapter(client, model=args.model, reasoning_effort="none"),
            history_limit=scenario.history_limit,
        )
        loop.reset_history()

        scenario_result = {
            "name": scenario.name,
            "turns": 0,
            "probes": [],
        }
        for index, step in enumerate(scenario.steps):
            if step.advance_hours:
                clock.advance(hours=step.advance_hours)
            if step.user_text is None:
                continue
            run = loop.turn(
                step.user_text,
                use_cognition=True,
                context_budget_tokens=step.context_budget_tokens or scenario.context_budget_tokens,
                top_k=step.top_k or scenario.top_k,
                max_output_tokens=step.max_output_tokens if step.max_output_tokens is not None else scenario.max_output_tokens,
            )
            scenario_result["turns"] += 1
            totals["user_turns"] += 1
            totals["answer_provider_tokens"] += usage_int(run.response.usage if run.response else {}, "total_tokens")

            packet = run.cognitive_decision.context_packet if run.cognitive_decision else None
            if packet:
                reranker = packet.diagnostics.get("semantic_reranker") or {}
                totals["context_reranker_calls"] += usage_int(reranker, "calls")
                totals["context_reranker_tokens"] += usage_int(reranker, "total_tokens")

            if step.expectation is not None:
                metrics = evaluator.score(run, step.expectation)
                scenario_result["probes"].append(
                    {
                        "index": index,
                        "label": step.label,
                        "simulated_at": clock().isoformat(),
                        "answer_score": metrics.answer_score,
                        "context_score": metrics.context_score,
                        "answer": run.response.text if run.response else "",
                        "context": packet.rendered if packet else "",
                    }
                )
                totals["probes"] += 1

        event_stats = extractor.stats()
        retraction_stats = state.retraction_resolution_stats()
        totals["event_calls"] += int(event_stats.get("calls") or 0)
        totals["event_tokens"] += int(event_stats.get("total_tokens") or 0)
        totals["event_emitted_events"] += int(event_stats.get("emitted_events") or 0)
        totals["event_skipped_non_mutating"] += int(event_stats.get("skipped_non_mutating_requests") or 0)
        totals["event_semantic_reviews"] += int(event_stats.get("semantic_reviews") or 0)
        totals["retraction_resolver_calls"] += int(retraction_stats.get("calls") or 0)
        totals["retraction_resolver_tokens"] += int(retraction_stats.get("total_tokens") or 0)
        totals["retraction_resolved"] += int(retraction_stats.get("resolved") or 0)
        totals["final_active_memories"] += len(store.all())
        totals["final_open_goals"] += sum(not goal.completed for goal in goals.all())
        scenario_result["event_stats"] = event_stats
        scenario_result["retraction_resolution_stats"] = retraction_stats
        results.append(scenario_result)

    probes = [probe for scenario in results for probe in scenario["probes"]]
    answer_scores = [float(probe["answer_score"]) for probe in probes if probe["answer_score"] is not None]
    context_scores = [float(probe["context_score"]) for probe in probes if probe["context_score"] is not None]
    perfect_answer = sum(score >= 0.999999 for score in answer_scores)
    perfect_context = sum(score >= 0.999999 for score in context_scores)

    report = {
        "metadata": {
            "experiment": "frozen_v4_targeted_resolved_state_regression",
            "cognitive_only": True,
            "frozen_case_commit": FROZEN_CASE_COMMIT,
            "suite_sha256": suite_sha,
            "candidate_implementation_commit": CANDIDATE_IMPLEMENTATION_COMMIT,
            "target_scenarios": sorted(TARGET_SCENARIOS),
            "model": args.model,
            "event_model": args.event_model,
            "embedding_model": args.embedding_model,
            "reranker_model": args.reranker_model,
        },
        "summary": {
            **totals,
            "mean_answer_score": mean(answer_scores) if answer_scores else None,
            "mean_context_score": mean(context_scores) if context_scores else None,
            "perfect_answer_probes": perfect_answer,
            "perfect_context_probes": perfect_context,
        },
        "results": results,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "infinito3-v4-resolved-targeted.json"
    md_path = output_dir / "infinito3-v4-resolved-targeted.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# INFINITO 3.0 — Frozen V4 Targeted Resolved-State Regression",
        "",
        f"- Frozen suite SHA256: `{suite_sha}`",
        f"- Candidate implementation: `{CANDIDATE_IMPLEMENTATION_COMMIT}`",
        f"- Scenarios: {len(results)}",
        f"- User turns: {totals['user_turns']}",
        f"- Probes: {totals['probes']}",
        f"- Mean answer score: {report['summary']['mean_answer_score']:.3f}",
        f"- Mean context score: {report['summary']['mean_context_score']:.3f}",
        f"- Perfect answer/context probes: {perfect_answer}/{perfect_context}",
        f"- Event calls / tokens: {totals['event_calls']} / {totals['event_tokens']}",
        f"- Context reranker calls / tokens: {totals['context_reranker_calls']} / {totals['context_reranker_tokens']}",
        f"- Retraction resolver calls / resolved / tokens: {totals['retraction_resolver_calls']} / {totals['retraction_resolved']} / {totals['retraction_resolver_tokens']}",
        f"- Final active memories / open goals: {totals['final_active_memories']} / {totals['final_open_goals']}",
        "",
        "| Scenario | Probe | Answer | Context |",
        "| --- | --- | ---: | ---: |",
    ]
    for scenario in results:
        for probe in scenario["probes"]:
            lines.append(
                f"| {scenario['name']} | {probe['label']} | {probe['answer_score']:.3f} | {probe['context_score']:.3f} |"
            )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print("INFINITO 3.0 FROZEN V4 TARGETED RESOLVED-STATE REGRESSION")
    print(f"suite_sha256={suite_sha}")
    print(f"candidate_implementation_commit={CANDIDATE_IMPLEMENTATION_COMMIT}")
    print(f"scenarios={len(results)}")
    print(f"user_turns={totals['user_turns']}")
    print(f"probes={totals['probes']}")
    print(f"mean_answer_score={report['summary']['mean_answer_score']}")
    print(f"mean_context_score={report['summary']['mean_context_score']}")
    print(f"perfect_answer_probes={perfect_answer}")
    print(f"perfect_context_probes={perfect_context}")
    print(f"semantic_event_calls={totals['event_calls']}")
    print(f"semantic_event_total_tokens={totals['event_tokens']}")
    print(f"context_reranker_calls={totals['context_reranker_calls']}")
    print(f"context_reranker_total_tokens={totals['context_reranker_tokens']}")
    print(f"retraction_resolver_calls={totals['retraction_resolver_calls']}")
    print(f"retraction_resolver_resolved={totals['retraction_resolved']}")
    print(f"retraction_resolver_tokens={totals['retraction_resolver_tokens']}")
    print(f"final_active_memories={totals['final_active_memories']}")
    print(f"final_open_goals={totals['final_open_goals']}")
    print(f"json_report={json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
