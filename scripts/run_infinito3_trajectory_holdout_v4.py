#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from src.infinito3.cognitive_loop import CognitiveLoop
from src.infinito3.engine import CognitiveEngine
from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.persistent_memory import HashEmbeddingProvider, OpenAIEmbeddingProvider
from src.infinito3.semantic_event_extractor import SemanticCognitiveEventExtractor
from src.infinito3.semantic_reranker import LLMSemanticMembershipReranker
from src.infinito3.semantic_temporal_memory import SemanticTemporalMemoryStore
from src.infinito3.semantic_temporal_state import SemanticTemporalCognitiveState
from src.infinito3.temporal_context import TemporalSemanticContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.trajectory_evaluation import MutableClock, TrajectoryEvaluationHarness
from src.infinito3.trajectory_holdout_v4_cases import independent_trajectory_holdout_v4_suite

FROZEN_CASE_COMMIT = "ef689e8adece7aa265c236ac9f50db0ebac45469"
IMPLEMENTATION_COMMIT = "9be85131cbabf257036bfb6742493ffe9f8d66d2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen INFINITO 3.0 held-out V4.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("OPENAI_REASONING_EFFORT", "none"))
    parser.add_argument("--embedding-provider", choices=("hash", "openai"), default=os.environ.get("INFINITO_EMBEDDING_PROVIDER", "openai"))
    parser.add_argument("--embedding-model", default=os.environ.get("INFINITO_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--semantic-reranker", choices=("none", "llm"), default=os.environ.get("INFINITO_SEMANTIC_RERANKER", "llm"))
    parser.add_argument("--reranker-model", default=os.environ.get("INFINITO_RERANKER_MODEL", ""))
    parser.add_argument("--event-model", default=os.environ.get("INFINITO_EVENT_MODEL", ""))
    parser.add_argument("--output-dir", default="trajectory-holdout-v4-results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    reranker_model = args.reranker_model.strip() or args.model.strip()
    event_model = args.event_model.strip() or args.model.strip()
    semantic_extractors = []

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
        builder = TemporalSemanticContextBuilder(
            memory_store=store,
            goal_engine=goals,
            now_fn=clock,
            reranker=semantic_reranker(),
        )
        extractor = SemanticCognitiveEventExtractor(
            OpenAIResponsesAdapter(client, model=event_model, reasoning_effort=None),
            now_fn=clock,
            max_output_tokens=420,
            min_confidence=0.72,
        )
        semantic_extractors.append(extractor)
        return CognitiveEngine(
            memory_store=store,
            goal_engine=goals,
            context_builder=builder,
            event_extractor=extractor,
            temporal_state=SemanticTemporalCognitiveState(now_fn=clock),
        )

    def loop_pair_factory(clock, scenario):
        baseline = CognitiveLoop(
            make_engine(clock),
            OpenAIResponsesAdapter(client, model=args.model.strip(), reasoning_effort=args.reasoning_effort.strip() or None),
            history_limit=scenario.history_limit,
        )
        cognitive = CognitiveLoop(
            make_engine(clock),
            OpenAIResponsesAdapter(client, model=args.model.strip(), reasoning_effort=args.reasoning_effort.strip() or None),
            history_limit=scenario.history_limit,
        )
        return baseline, cognitive

    case_path = REPO_ROOT / "src" / "infinito3" / "trajectory_holdout_v4_cases.py"
    scenarios = independent_trajectory_holdout_v4_suite()
    report = TrajectoryEvaluationHarness(loop_pair_factory).run(scenarios)

    event_stats = {"calls": 0, "successes": 0, "failures": 0, "emitted_events": 0,
                   "input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
    event_errors = {}
    for extractor in semantic_extractors:
        stats = extractor.stats()
        for key in event_stats:
            event_stats[key] += int(stats.get(key) or 0)
        for key, value in (stats.get("errors") or {}).items():
            event_errors[key] = event_errors.get(key, 0) + int(value or 0)

    s = report.summary
    effective_with_events = s.cognitive_effective_total_tokens + event_stats["total_tokens"]
    effective_delta_with_events = effective_with_events - s.baseline_total_tokens
    report.metadata.update({
        "provider": "openai",
        "model": args.model.strip(),
        "reasoning_effort": args.reasoning_effort.strip() or None,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model.strip() if args.embedding_provider == "openai" else None,
        "context_builder": "temporal_semantic_cohort_uncertainty_gated",
        "semantic_reranker": args.semantic_reranker,
        "reranker_model": reranker_model if args.semantic_reranker == "llm" else None,
        "cognitive_event_extractor": "hybrid_semantic_v1",
        "semantic_event_model": event_model,
        "semantic_event_stats": {**event_stats, "errors": event_errors},
        "temporal_cognitive_state": "semantic_event_sourced_v1",
        "temporal_memory_projection": "semantic_temporal_sqlite_v1",
        "real_model_run": True,
        "suite": "independent_trajectory_holdout_v4_suite",
        "suite_frozen_before_first_live_run": True,
        "frozen_case_commit": FROZEN_CASE_COMMIT,
        "implementation_commit": IMPLEMENTATION_COMMIT,
        "suite_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "cognitive_effective_total_tokens_including_event_extractor": effective_with_events,
        "effective_total_token_delta_including_event_extractor": effective_delta_with_events,
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "infinito3-trajectory-holdout-v4.json"
    md_path = output_dir / "infinito3-trajectory-holdout-v4.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    print("INFINITO 3.0 FOURTH FROZEN LONG-HORIZON HELD-OUT")
    print(f"implementation_commit={IMPLEMENTATION_COMMIT}")
    print(f"frozen_case_commit={FROZEN_CASE_COMMIT}")
    print(f"suite_sha256={report.metadata['suite_sha256']}")
    print(f"model={args.model.strip()}")
    print(f"event_model={event_model}")
    print(f"trajectories={s.trajectory_count}")
    print(f"user_turns={s.user_turn_count}")
    print(f"probes={s.probe_count}")
    print(f"wins/ties/losses={s.cognitive_wins}/{s.ties}/{s.baseline_wins}")
    print(f"mean_baseline_score={s.mean_baseline_answer_score}")
    print(f"mean_cognitive_score={s.mean_cognitive_answer_score}")
    print(f"mean_answer_lift={s.mean_answer_lift}")
    print(f"mean_context_score={s.mean_context_score}")
    print(f"semantic_event_calls={event_stats['calls']}")
    print(f"semantic_event_successes={event_stats['successes']}")
    print(f"semantic_event_failures={event_stats['failures']}")
    print(f"semantic_event_emitted_events={event_stats['emitted_events']}")
    print(f"semantic_event_total_tokens={event_stats['total_tokens']}")
    print(f"reranker_calls={s.reranker_calls}")
    print(f"reranker_total_tokens={s.reranker_total_tokens}")
    print(f"cognitive_effective_total_tokens_with_event_extractor={effective_with_events}")
    print(f"effective_total_token_delta_with_event_extractor={effective_delta_with_events}")
    print(f"final_active_memories={s.final_active_memories}")
    print(f"final_open_goals={s.final_open_goals}")
    print(f"json_report={json_path}")
    print(f"markdown_report={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
