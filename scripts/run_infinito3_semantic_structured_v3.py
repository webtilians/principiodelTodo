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
from src.infinito3.semantic_goal_state import SemanticGoalTemporalState
from src.infinito3.semantic_interpreter import SemanticCognitiveEventExtractor, SemanticStateQueryAnalyzer
from src.infinito3.semantic_reranker import LLMSemanticMembershipReranker
from src.infinito3.semantic_temporal_memory import SemanticTemporalMemoryStore
from src.infinito3.structured_temporal_context import StructuredTemporalContextBuilder
from src.infinito3.temporal_goals import TemporalGoalEngine
from src.infinito3.trajectory_evaluation import MutableClock, TrajectoryEvaluationHarness
from src.infinito3.trajectory_holdout_v3_cases import independent_trajectory_holdout_v3_suite

FROZEN_V3_CASE_COMMIT = "cbb05b9a803a6d0ed021f5790d43d6a87e45f426"
BASELINE_V3_IMPLEMENTATION = "9256992d2ac997bde2ab6a5851f29a9211e9eed7"


def parse_args():
    parser = argparse.ArgumentParser(description="Run frozen V3 with semantic events + structured temporal state.")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.6-luna"))
    parser.add_argument("--reasoning-effort", default=os.environ.get("OPENAI_REASONING_EFFORT", "none"))
    parser.add_argument("--embedding-provider", choices=("hash", "openai"), default="openai")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--semantic-reranker", choices=("none", "llm"), default="llm")
    parser.add_argument("--reranker-model", default="")
    parser.add_argument("--interpreter-model", default="")
    parser.add_argument("--output-dir", default="semantic-structured-v3-results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY is missing")

    client = OpenAI(api_key=api_key)
    model = args.model.strip()
    reranker_model = args.reranker_model.strip() or model
    interpreter_model = args.interpreter_model.strip() or model

    semantic_extractors = []
    query_analyzers = []

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
        store = SemanticTemporalMemoryStore(":memory:", embedding_provider=embedding_provider())
        goals = TemporalGoalEngine(now_fn=clock)
        state = SemanticGoalTemporalState(now_fn=clock)

        event_extractor = SemanticCognitiveEventExtractor(
            OpenAIResponsesAdapter(client, model=interpreter_model, reasoning_effort=None),
            now_fn=clock,
        )
        query_analyzer = SemanticStateQueryAnalyzer(
            OpenAIResponsesAdapter(client, model=interpreter_model, reasoning_effort=None),
            max_output_tokens=220,
        )
        semantic_extractors.append(event_extractor)
        query_analyzers.append(query_analyzer)

        builder = StructuredTemporalContextBuilder(
            memory_store=store,
            goal_engine=goals,
            temporal_state=state,
            query_analyzer=query_analyzer,
            now_fn=clock,
            reranker=semantic_reranker(),
        )
        return CognitiveEngine(
            memory_store=store,
            goal_engine=goals,
            context_builder=builder,
            event_extractor=event_extractor,
            temporal_state=state,
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

    case_path = REPO_ROOT / "src" / "infinito3" / "trajectory_holdout_v3_cases.py"
    report = TrajectoryEvaluationHarness(loop_pair_factory).run(independent_trajectory_holdout_v3_suite())

    extractor_usage = _sum_usage(component.usage_summary() for component in semantic_extractors)
    query_usage = _sum_usage(component.usage_summary() for component in query_analyzers)
    interpreter_total = extractor_usage["total_tokens"] + query_usage["total_tokens"]

    report.metadata.update({
        "provider": "openai",
        "model": model,
        "reasoning_effort": args.reasoning_effort.strip() or None,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model.strip() if args.embedding_provider == "openai" else None,
        "context_builder": "structured_temporal_semantic_v1",
        "semantic_reranker": args.semantic_reranker,
        "reranker_model": reranker_model if args.semantic_reranker == "llm" else None,
        "event_extractor": "hybrid_semantic_closed_schema_v1",
        "state_query_analyzer": "semantic_closed_schema_v1",
        "temporal_state": "semantic_goal_temporal_state_v2",
        "interpreter_model": interpreter_model,
        "event_extractor_usage": extractor_usage,
        "state_query_analyzer_usage": query_usage,
        "interpreter_total_tokens": interpreter_total,
        "suite": "independent_trajectory_holdout_v3_suite",
        "frozen_case_commit": FROZEN_V3_CASE_COMMIT,
        "baseline_v3_implementation": BASELINE_V3_IMPLEMENTATION,
        "suite_sha256": hashlib.sha256(case_path.read_bytes()).hexdigest(),
        "real_model_run": True,
    })

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "infinito3-semantic-structured-v3.json"
    md_path = output_dir / "infinito3-semantic-structured-v3.md"
    json_path.write_text(report.to_json(), encoding="utf-8")
    md_path.write_text(report.to_markdown(), encoding="utf-8")

    s = report.summary
    print("INFINITO 3.0 FROZEN V3 — SEMANTIC EVENTS + STRUCTURED STATE")
    print(f"frozen_case_commit={FROZEN_V3_CASE_COMMIT}")
    print(f"suite_sha256={report.metadata['suite_sha256']}")
    print(f"model={model}")
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
    print(f"event_extractor_calls={extractor_usage['calls']}")
    print(f"event_extractor_tokens={extractor_usage['total_tokens']}")
    print(f"state_query_calls={query_usage['calls']}")
    print(f"state_query_tokens={query_usage['total_tokens']}")
    print(f"interpreter_total_tokens={interpreter_total}")
    print(f"final_active_memories={s.final_active_memories}")
    print(f"final_open_goals={s.final_open_goals}")
    print(f"json_report={json_path}")
    print(f"markdown_report={md_path}")
    return 0


def _sum_usage(items):
    result = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "errors": []}
    for item in items:
        result["calls"] += int(item.get("calls") or 0)
        result["input_tokens"] += int(item.get("input_tokens") or 0)
        result["output_tokens"] += int(item.get("output_tokens") or 0)
        result["total_tokens"] += int(item.get("total_tokens") or 0)
        result["errors"].extend(item.get("errors") or [])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
