#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from src.infinito3.advanced_temporal_state import ResolvedTemporalCognitiveState
from src.infinito3.cognitive_loop import CognitiveLoop
from src.infinito3.engine import CognitiveEngine
from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.persistent_memory import HashEmbeddingProvider, OpenAIEmbeddingProvider
from src.infinito3.semantic_event_extractor import SemanticCognitiveEventExtractor
from src.infinito3.semantic_reranker import LLMSemanticMembershipReranker
from src.infinito3.semantic_temporal_memory import SemanticTemporalMemoryStore
from src.infinito3.structured_context import StructuredTemporalContextBuilder
from src.infinito3.structured_retrieval import StructuredStateQueryPlanner, StructuredStateRetriever
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.trajectory_evaluation import MutableClock, TrajectoryEvaluationHarness
from src.infinito3.trajectory_holdout_v3_cases import independent_trajectory_holdout_v3_suite

FROZEN_CASE_COMMIT = "cbb05b9a803a6d0ed021f5790d43d6a87e45f426"
FROZEN_CASE_BLOB = "44262e3a1da91806d09352a9d84bf69db76178f8"
EXPECTED_SUITE_SHA256 = "58e9db6adb94fea9a1f98293d1c6a305166f3debea8b9e8d84142c7627f61ccb"
ORIGINAL_V3_IMPLEMENTATION = "9256992d2ac997bde2ab6a5851f29a9211e9eed7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the three-piece INFINITO repair on exact frozen held-out V3.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("OPENAI_REASONING_EFFORT", "none"))
    parser.add_argument("--embedding-provider", choices=("hash", "openai"), default=os.environ.get("INFINITO_EMBEDDING_PROVIDER", "openai"))
    parser.add_argument("--embedding-model", default=os.environ.get("INFINITO_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--semantic-reranker", choices=("none", "llm"), default=os.environ.get("INFINITO_SEMANTIC_RERANKER", "llm"))
    parser.add_argument("--reranker-model", default=os.environ.get("INFINITO_RERANKER_MODEL", ""))
    parser.add_argument("--event-model", default=os.environ.get("INFINITO_EVENT_MODEL", ""))
    parser.add_argument("--planner-model", default=os.environ.get("INFINITO_PLANNER_MODEL", ""))
    parser.add_argument("--output-dir", default="trajectory-v3-three-piece-repair-results")
    return parser.parse_args()


def _sum_usage(components):
    total = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "failures": 0}
    for component in components:
        getter = getattr(component, "usage_snapshot", None)
        if not callable(getter):
            continue
        usage = getter()
        for key in total:
            try:
                total[key] += int(usage.get(key) or 0)
            except (TypeError, ValueError):
                pass
    return total


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")

    case_path = REPO_ROOT / "src" / "infinito3" / "trajectory_holdout_v3_cases.py"
    suite_sha256 = hashlib.sha256(case_path.read_bytes()).hexdigest()
    if suite_sha256 != EXPECTED_SUITE_SHA256:
        raise SystemExit(
            "Frozen V3 bank mismatch: "
            f"expected {EXPECTED_SUITE_SHA256}, got {suite_sha256}. Refusing paid run."
        )

    client = OpenAI(api_key=api_key)
    model = args.model.strip()
    reranker_model = args.reranker_model.strip() or model
    event_model = args.event_model.strip() or model
    planner_model = args.planner_model.strip() or model
    event_extractors = []
    query_planners = []

    def embedding_provider():
        if args.embedding_provider == "openai":
            return OpenAIEmbeddingProvider(client, model=args.embedding_model.strip())
        return HashEmbeddingProvider()

    def semantic_reranker():
        if args.semantic_reranker != "llm":
            return None
        return LLMSemanticMembershipReranker(
            OpenAIResponsesAdapter(client, model=reranker_model, reasoning_effort=None),
            max_output_tokens=160,
        )

    def make_engine(clock: MutableClock) -> CognitiveEngine:
        store = SemanticTemporalMemoryStore(path=":memory:", embedding_provider=embedding_provider())
        goals = TemporalGoalEngine(now_fn=clock)
        state = ResolvedTemporalCognitiveState(now_fn=clock)

        event_extractor = SemanticCognitiveEventExtractor(
            OpenAIResponsesAdapter(client, model=event_model, reasoning_effort=None),
            now_fn=clock,
            max_output_tokens=320,
            min_confidence=0.72,
        )
        planner = StructuredStateQueryPlanner(
            OpenAIResponsesAdapter(client, model=planner_model, reasoning_effort=None),
            max_output_tokens=160,
        )
        retriever = StructuredStateRetriever(state, store, planner=planner)
        builder = StructuredTemporalContextBuilder(
            memory_store=store,
            goal_engine=goals,
            now_fn=clock,
            reranker=semantic_reranker(),
        )
        event_extractors.append(event_extractor)
        query_planners.append(planner)

        return CognitiveEngine(
            memory_store=store,
            goal_engine=goals,
            context_builder=builder,
            event_extractor=event_extractor,
            temporal_state=state,
            structured_retriever=retriever,
        )

    def loop_pair_factory(clock, scenario):
        baseline = CognitiveLoop(
            make_engine(clock),
            OpenAIResponsesAdapter(client, model=model, reasoning_effort=args.reasoning_effort.strip() or None),
            history_limit=scenario.history_limit,
        )
        cognitive = CognitiveLoop(
            make_engine(clock),
            OpenAIResponsesAdapter(client, model=model, reasoning_effort=args.reasoning_effort.strip() or None),
            history_limit=scenario.history_limit,
        )
        return baseline, cognitive

    scenarios = independent_trajectory_holdout_v3_suite()
    report = TrajectoryEvaluationHarness(loop_pair_factory).run(scenarios)

    event_usage = _sum_usage(event_extractors)
    planner_usage = _sum_usage(query_planners)
    auxiliary_tokens = event_usage["total_tokens"] + planner_usage["total_tokens"]
    implementation_commit = os.environ.get("GITHUB_SHA", "working-tree")

    report.metadata.update(
        {
            "provider": "openai",
            "model": model,
            "reasoning_effort": args.reasoning_effort.strip() or None,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model.strip() if args.embedding_provider == "openai" else None,
            "semantic_reranker": args.semantic_reranker,
            "reranker_model": reranker_model if args.semantic_reranker == "llm" else None,
            "semantic_event_extractor": "hybrid_closed_schema_semantic_v1",
            "semantic_event_model": event_model,
            "semantic_event_usage": event_usage,
            "goal_resolver": "state_semantic_temporal_v1",
            "structured_query_planner": "closed_schema_semantic_v1",
            "structured_query_planner_model": planner_model,
            "structured_query_planner_usage": planner_usage,
            "structured_state_retrieval": "temporal_slots_lineage_v1",
            "structured_temporal_renderer": "explicit_relation_v1",
            "auxiliary_semantic_tokens": auxiliary_tokens,
            "real_model_run": True,
            "suite": "independent_trajectory_holdout_v3_suite",
            "suite_frozen_before_repair": True,
            "frozen_case_commit": FROZEN_CASE_COMMIT,
            "frozen_case_blob": FROZEN_CASE_BLOB,
            "suite_sha256": suite_sha256,
            "original_v3_implementation": ORIGINAL_V3_IMPLEMENTATION,
            "repair_implementation_commit": implementation_commit,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "infinito3-v3-three-piece-repair.json"
    md_path = output_dir / "infinito3-v3-three-piece-repair.md"
    json_path.write_text(report.to_json(), encoding="utf-8")

    s = report.summary
    all_cognitive_tokens = s.cognitive_effective_total_tokens + auxiliary_tokens
    all_token_delta = all_cognitive_tokens - s.baseline_total_tokens
    extra = [
        "",
        "## Three-piece repair auxiliary model cost",
        "",
        f"- semantic event extractor: {event_usage['calls']} calls / {event_usage['total_tokens']} tokens / {event_usage['failures']} failures",
        f"- structured query planner: {planner_usage['calls']} calls / {planner_usage['total_tokens']} tokens / {planner_usage['failures']} failures",
        f"- auxiliary semantic tokens: {auxiliary_tokens}",
        f"- all-in cognitive effective tokens: {all_cognitive_tokens}",
        f"- all-in token delta vs baseline: {all_token_delta}",
        "",
    ]
    md_path.write_text(report.to_markdown() + "\n".join(extra), encoding="utf-8")

    summary_payload = {
        "frozen_case_commit": FROZEN_CASE_COMMIT,
        "frozen_case_blob": FROZEN_CASE_BLOB,
        "suite_sha256": suite_sha256,
        "implementation_commit": implementation_commit,
        "trajectories": s.trajectory_count,
        "user_turns": s.user_turn_count,
        "probes": s.probe_count,
        "wins": s.cognitive_wins,
        "ties": s.ties,
        "losses": s.baseline_wins,
        "mean_baseline_score": s.mean_baseline_answer_score,
        "mean_cognitive_score": s.mean_cognitive_answer_score,
        "mean_answer_lift": s.mean_answer_lift,
        "mean_context_score": s.mean_context_score,
        "baseline_total_tokens": s.baseline_total_tokens,
        "cognitive_total_tokens": s.cognitive_total_tokens,
        "reranker_calls": s.reranker_calls,
        "reranker_total_tokens": s.reranker_total_tokens,
        "event_usage": event_usage,
        "planner_usage": planner_usage,
        "all_cognitive_effective_tokens": all_cognitive_tokens,
        "all_token_delta": all_token_delta,
        "cumulative_context_tokens": s.cumulative_context_tokens,
        "final_active_memories": s.final_active_memories,
        "final_open_goals": s.final_open_goals,
    }
    print("INFINITO 3.0 V3 THREE-PIECE REPAIR")
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True))
    print(f"json_report={json_path}")
    print(f"markdown_report={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
