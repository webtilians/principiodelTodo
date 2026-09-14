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
from src.infinito3.context_builder import BalancedContextBuilder
from src.infinito3.engine import CognitiveEngine
from src.infinito3.generalized_context_builder import GeneralizedContextBuilder
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.persistent_memory import (
    HashEmbeddingProvider,
    OpenAIEmbeddingProvider,
    SQLiteCognitiveMemoryStore,
)
from src.infinito3.trajectory_cases import independent_trajectory_suite
from src.infinito3.trajectory_evaluation import MutableClock, TrajectoryEvaluationHarness


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run INFINITO 3.0 independent long-horizon conversation trajectories."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"),
        help="OpenAI API model id.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENAI_REASONING_EFFORT", "none"),
        help="Reasoning effort supported by the selected model.",
    )
    parser.add_argument(
        "--embedding-provider",
        choices=("hash", "openai"),
        default=os.environ.get("INFINITO_EMBEDDING_PROVIDER", "openai"),
    )
    parser.add_argument(
        "--embedding-model",
        default=os.environ.get("INFINITO_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    parser.add_argument(
        "--output-dir",
        default="trajectory-results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")
    if not args.model.strip():
        raise SystemExit("No model specified")

    client = OpenAI(api_key=api_key)

    def embedding_provider():
        if args.embedding_provider == "openai":
            return OpenAIEmbeddingProvider(client, model=args.embedding_model.strip())
        return HashEmbeddingProvider()

    def make_engine(clock: MutableClock) -> CognitiveEngine:
        store = SQLiteCognitiveMemoryStore(
            path=":memory:",
            embedding_provider=embedding_provider(),
        )
        goals = SimpleGoalEngine(now_fn=clock)
        context_builder = GeneralizedContextBuilder(
            memory_store=store,
            goal_engine=goals,
            now_fn=clock,
        )
        return CognitiveEngine(
            memory_store=store,
            goal_engine=goals,
            context_builder=context_builder,
        )

    def loop_pair_factory(clock, scenario):
        baseline = CognitiveLoop(
            make_engine(clock),
            OpenAIResponsesAdapter(
                client,
                model=args.model.strip(),
                reasoning_effort=args.reasoning_effort.strip() or None,
            ),
            history_limit=scenario.history_limit,
        )
        cognitive = CognitiveLoop(
            make_engine(clock),
            OpenAIResponsesAdapter(
                client,
                model=args.model.strip(),
                reasoning_effort=args.reasoning_effort.strip() or None,
            ),
            history_limit=scenario.history_limit,
        )
        return baseline, cognitive

    scenarios = independent_trajectory_suite()
    report = TrajectoryEvaluationHarness(loop_pair_factory).run(scenarios)
    report.metadata.update(
        {
            "provider": "openai",
            "model": args.model.strip(),
            "reasoning_effort": args.reasoning_effort.strip() or None,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model.strip() if args.embedding_provider == "openai" else None,
            "real_model_run": True,
            "suite": "independent_trajectory_suite",
            "suite_sha256": hashlib.sha256(
                (REPO_ROOT / "src" / "infinito3" / "trajectory_cases.py").read_bytes()
            ).hexdigest(),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "infinito3-trajectory-evaluation.json"
    markdown_path = output_dir / "infinito3-trajectory-evaluation.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")

    s = report.summary
    print("INFINITO 3.0 INDEPENDENT TRAJECTORY EVALUATION")
    print(f"model={args.model.strip()}")
    print(f"embedding_provider={args.embedding_provider}")
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
    print(f"total_token_delta={s.total_token_delta}")
    print(f"cumulative_context_tokens={s.cumulative_context_tokens}")
    print(f"final_active_memories={s.final_active_memories}")
    print(f"final_open_goals={s.final_open_goals}")
    print(f"json_report={json_path}")
    print(f"markdown_report={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
