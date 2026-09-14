#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI

from src.infinito3.benchmark_cases import extended_evaluation_suite
from src.infinito3.cognitive_loop import CognitiveLoop
from src.infinito3.engine import CognitiveEngine
from src.infinito3.evaluation import EvaluationHarness, standard_evaluation_suite
from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.persistent_memory import HashEmbeddingProvider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the INFINITO 3.0 paired A/B evaluation against a real OpenAI model."
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENAI_MODEL", ""),
        help="OpenAI API model id. Can also be provided through OPENAI_MODEL.",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENAI_REASONING_EFFORT", ""),
        help="Optional reasoning effort supported by the selected model.",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=4,
        help="Maximum number of short-term conversation turns visible to both A/B variants.",
    )
    parser.add_argument(
        "--suite",
        choices=("standard", "extended"),
        default="extended",
        help="Scenario bank to run. Extended is the default for live experiments.",
    )
    parser.add_argument(
        "--output-dir",
        default="evaluation-results",
        help="Directory where JSON and Markdown reports are written.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY is missing. Add it as a GitHub Actions secret or local environment variable."
        )
    if not args.model.strip():
        raise SystemExit(
            "No model was specified. Pass --model or set OPENAI_MODEL."
        )
    if args.history_limit < 0:
        raise SystemExit("--history-limit must be >= 0")

    client = OpenAI(api_key=api_key)

    def loop_factory() -> CognitiveLoop:
        engine = CognitiveEngine.persistent(
            db_path=":memory:",
            embedding_provider=HashEmbeddingProvider(),
        )
        llm = OpenAIResponsesAdapter(
            client,
            model=args.model.strip(),
            reasoning_effort=args.reasoning_effort.strip() or None,
        )
        return CognitiveLoop(
            engine,
            llm,
            history_limit=args.history_limit,
        )

    scenarios = (
        extended_evaluation_suite()
        if args.suite == "extended"
        else standard_evaluation_suite()
    )
    harness = EvaluationHarness(loop_factory)
    report = harness.run(scenarios)
    report.metadata.update(
        {
            "provider": "openai",
            "model": args.model.strip(),
            "reasoning_effort": args.reasoning_effort.strip() or None,
            "history_limit": args.history_limit,
            "embedding_provider": "hash_baseline",
            "real_model_run": True,
            "suite": args.suite,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "infinito3-evaluation.json"
    markdown_path = output_dir / "infinito3-evaluation.md"

    json_path.write_text(report.to_json(), encoding="utf-8")
    markdown_path.write_text(report.to_markdown(), encoding="utf-8")

    summary = report.summary
    print("INFINITO 3.0 REAL MODEL EVALUATION")
    print(f"model={args.model.strip()}")
    print(f"suite={args.suite}")
    print(f"scenarios={summary.scenario_count}")
    print(
        "wins/ties/losses="
        f"{summary.cognitive_wins}/{summary.ties}/{summary.baseline_wins}"
    )
    print(f"mean_baseline_score={summary.mean_baseline_answer_score}")
    print(f"mean_cognitive_score={summary.mean_cognitive_answer_score}")
    print(f"mean_answer_lift={summary.mean_answer_lift}")
    print(f"mean_context_score={summary.mean_context_score}")
    print(f"mean_context_tokens={summary.mean_context_tokens}")
    print(f"mean_total_token_delta={summary.mean_total_token_delta}")
    print(f"json_report={json_path}")
    print(f"markdown_report={markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
