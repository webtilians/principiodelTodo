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
from src.infinito3.event_extractor import TemporalCognitiveEventExtractor
from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.persistent_memory import HashEmbeddingProvider, OpenAIEmbeddingProvider
from src.infinito3.semantic_reranker import LLMSemanticMembershipReranker
from src.infinito3.semantic_temporal_memory import SemanticTemporalMemoryStore
from src.infinito3.semantic_temporal_state import SemanticTemporalCognitiveState
from src.infinito3.temporal_context import TemporalSemanticContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.trajectory_evaluation import MutableClock, TrajectoryEvaluationHarness
from src.infinito3.trajectory_holdout_v2_cases import independent_trajectory_holdout_v2_suite

FROZEN_CASE_COMMIT = "34b2b41922370ccbbcb1ba302113d5b431b22e55"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen INFINITO 3.0 long-horizon held-out v2.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("OPENAI_REASONING_EFFORT", "none"))
    parser.add_argument("--embedding-provider", choices=("hash", "openai"), default=os.environ.get("INFINITO_EMBEDDING_PROVIDER", "openai"))
    parser.add_argument("--embedding-model", default=os.environ.get("INFINITO_EMBEDDING_MODEL", "text-embedding-3-small"))
    parser.add_argument("--semantic-reranker", choices=("none", "llm"), default=os.environ.get("INFINITO_SEMANTIC_RERANKER", "llm"))
    parser.add_argument("--reranker-model", default=os.environ.get("INFINITO_RERANKER_MODEL", ""))
    parser.add_argument("--output-dir", default="trajectory-holdout-v2-results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    reranker_model = args.reranker_model.strip() or args.model.strip()

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
        return CognitiveEngine(
            memory_store=store,
            goal_engine=goals,
            context_builder=builder,
            event_extractor=TemporalCognitiveEventExtractor(now_fn=clock),
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

    case_path = REPO_ROOT / "src" / "infinito3" / "trajectory_holdout_v2_cases.py"
    scenarios = independent_trajectory_holdout_v2_suite()
    report = TrajectoryEvaluationHarness(loop_pair_factory).run(scenarios)
    report.metadata.update(
        {
            "provider": "openai",
            "model": args.model.strip(),
            "reasoning_effort": args.reasoning_effort.strip() or None,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model.strip() if args.embedding_provider == "openai" else None,
            "context_builder": "temporal_semantic_cohort_uncertainty_gated",
            "semantic_reranker": args.semantic_reranker,
            "reranker_model": reranker_model if args.semantic_reranker == "llm" else None,
            "cognitive_event_extractor": "temporal_rule_based_v1",
            "temporal_cognitive_state": "semantic_event_sourced_v1",
            "temporal_memory_projection": "semantic_temporal_sqlite_v1",
            "real_model_run": True,
            "suite": "independent_trajectory_holdout_v2_suite",
            "suite_frozen_before_first_live_run": True,
            "frozen_case_commit": FROZEN_CASE_COMMIT,
            "suite_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "infinito3-trajectory-holdout-v2.json"
    md_path = output_dir / "infinito3-trajectory-holdout-v2.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    s = report.summary
    print("INFINITO 3.0 SECOND FROZEN LONG-HORIZON HELD-OUT")
    print(f"frozen_case_commit={FROZEN_CASE_COMMIT}")
    print(f"suite_sha256={report.metadata['suite_sha256']}")
    print(f"model={args.model.strip()}")
    print(f"embedding_provider={args.embedding_provider}")
    print("context_builder=temporal_semantic_cohort_uncertainty_gated")
    print("cognitive_event_extractor=temporal_rule_based_v1")
    print("temporal_cognitive_state=semantic_event_sourced_v1")
    print(f"semantic_reranker={args.semantic_reranker}")
    print(f"trajectories={s.trajectory_count}")
    print(f"user_turns={s.user_turn_count}")
    print(f"probes={s.probe_count}")
    print(f"wins/ties/losses={s.cognitive_wins}/{s.ties}/{s.baseline_wins}")
    print(f"mean_baseline_score={s.mean_baseline_answer_score}")
    print(f"mean_cognitive_score={s.mean_cognitive_answer_score}")
    print(f"mean_answer_lift={s.mean_answer_lift}")
    print(f"mean_context_score={s.mean_context_score}")
    print(f"baseline_total_tokens={s.baseline_total_tokens}")
    print(f"cognitive_total_tokens={s.cognitive_total_tokens}")
    print(f"reranker_calls={s.reranker_calls}")
    print(f"reranker_total_tokens={s.reranker_total_tokens}")
    print(f"cognitive_effective_total_tokens={s.cognitive_effective_total_tokens}")
    print(f"effective_total_token_delta={s.effective_total_token_delta}")
    print(f"cumulative_context_tokens={s.cumulative_context_tokens}")
    print(f"final_active_memories={s.final_active_memories}")
    print(f"final_open_goals={s.final_open_goals}")
    print(f"json_report={json_path}")
    print(f"markdown_report={md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
