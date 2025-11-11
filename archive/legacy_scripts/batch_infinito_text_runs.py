#!/usr/bin/env python3
"""Batch runner for INFINITO V5.1 text experiments.

This utility executes the `InfinitoV51ConsciousnessBreakthrough` experiment
multiple times with the same input text and records summary metrics for each
run. The intent is to measure variability across repeated executions.

⚠️ WARNING: Each run produces console output, JSON logs, and breakthrough
checkpoint files. Running hundreds of trials will generate large amounts of
output and may take many hours.
"""

import argparse
import json
import math
import os
import random
import sys
import time
from argparse import Namespace
from datetime import datetime
from glob import glob
from pathlib import Path
from statistics import mean, pstdev, stdev

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from infinito_gpt_text_fixed import InfinitoV51ConsciousnessBreakthrough  # noqa: E402


DEFAULT_TEXT = (
    "Mientras exploro mi conciencia observo cada pensamiento que surge, "
    "reconozco la emoción que lo acompaña y reflexiono sobre cómo mi identidad "
    "cambia con cada memoria activada durante esta introspección presente."
)

CHECKPOINT_PATTERN = "CONSCIOUSNESS_BREAKTHROUGH*.pt"
OUTPUTS_DIR = REPO_ROOT / "outputs"


def build_args(max_iter: int, lr: float, seed: int, input_text: str) -> Namespace:
    """Create an argparse Namespace matching the runner's expectations."""
    return Namespace(
        max_iter=max_iter,
        lr=lr,
        seed=seed,
        batch_size=4,
        input_dim=257,
        hidden_dim=512,
        attention_heads=8,
        memory_slots=256,
        text_mode=False,
        input_text=input_text,
        text_examples=False,
        consciousness_boost=False,
        memory_active=False,
        comparative=False,
        comparative_iterations=100,
        bootstrap_samples=1000,
    )


def compute_output_path(experiment_data: dict, has_text: bool) -> Path:
    """Reconstruct the JSON output path created by the experiment."""
    timestamp = experiment_data.get("start_time", datetime.now().strftime("%Y%m%d_%H%M%S"))
    max_c = experiment_data.get("max_consciousness", 0.0)
    max_phi = experiment_data.get("max_phi", 0.0)
    suffix = "_TEXT" if has_text else ""
    filename = f"infinito_v5_1_consciousness{suffix}_{timestamp}_C{max_c:.3f}_PHI{max_phi:.3f}.json"
    return OUTPUTS_DIR / filename


def relative_to_repo(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def list_checkpoints() -> set:
    pattern = REPO_ROOT / CHECKPOINT_PATTERN
    return {Path(p) for p in glob(str(pattern))}


def load_existing_runs(glob_pattern: str) -> list:
    if not glob_pattern:
        return []

    pattern = glob_pattern
    if not os.path.isabs(pattern):
        pattern = str(REPO_ROOT / pattern)

    paths = sorted(Path(p) for p in glob(pattern))
    summaries = []

    for path in paths:
        try:
            with path.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception as exc:  # pragma: no cover - diagnostic output
            print(f"⚠️  Could not load {path}: {exc}")
            continue

        config = data.get("config", {})
        consciousness_values = data.get("consciousness_values", [])
        phi_values = data.get("phi_values", [])

        final_consciousness = data.get("final_consciousness")
        if final_consciousness is None and consciousness_values:
            final_consciousness = consciousness_values[-1]
        final_consciousness = float(final_consciousness or 0.0)

        final_phi = data.get("final_phi")
        if final_phi is None and phi_values:
            final_phi = phi_values[-1]
        final_phi = float(final_phi or 0.0)

        summaries.append(
            {
                "run_index": None,
                "seed": config.get("seed"),
                "runtime_seconds": data.get("total_time_seconds", 0.0),
                "breakthrough": data.get("breakthrough_achieved", False),
                "final_consciousness": final_consciousness,
                "max_consciousness": data.get("max_consciousness", final_consciousness),
                "final_phi": final_phi,
                "max_phi": data.get("max_phi", final_phi),
                "output_file": relative_to_repo(path),
                "origin": "existing",
            }
        )

    return summaries


def assign_run_indices(summaries: list) -> None:
    for idx, entry in enumerate(summaries, 1):
        entry["run_index"] = idx


def run_experiment_series(
    runs: int,
    max_iter: int,
    base_seed: int,
    input_text: str,
    cleanup_checkpoints: bool = False,
    cleanup_json: bool = False,
    start_index: int = 1,
) -> list:
    """Execute the experiment multiple times and gather per-run summaries."""
    summaries = []
    OUTPUTS_DIR.mkdir(exist_ok=True)

    for run_idx in range(1, runs + 1):
        run_seed = base_seed + run_idx - 1
        print(f"\n=== RUN {run_idx}/{runs} (seed={run_seed}) ===")

        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        random.seed(run_seed)

        args = build_args(max_iter=max_iter, lr=0.001, seed=run_seed, input_text=input_text)
        runner = InfinitoV51ConsciousnessBreakthrough(args)

        checkpoints_before = list_checkpoints()
        start_time = time.time()
        runner.run_experiment(max_iter)
        elapsed = time.time() - start_time

        exp_data = runner.experiment_data
        output_path = compute_output_path(exp_data, has_text=bool(input_text))
        checkpoints_after = list_checkpoints()
        new_checkpoints = sorted(checkpoints_after - checkpoints_before)

        breakthrough_targets = set(new_checkpoints)
        for br in exp_data.get("breakthroughs", []):
            iteration = br.get("iteration")
            conc = br.get("consciousness", 0.0)
            phi_value = br.get("phi", 0.0)
            filename = (
                f"CONSCIOUSNESS_BREAKTHROUGH_V51_TEXT_iter_{iteration}_C_{conc:.3f}_PHI_{phi_value:.3f}.pt"
            )
            path = REPO_ROOT / filename
            if path.exists():
                breakthrough_targets.add(path)

        checkpoint_records = [relative_to_repo(path) for path in sorted(breakthrough_targets)]

        if cleanup_checkpoints:
            for checkpoint_path in breakthrough_targets:
                if checkpoint_path.exists():
                    try:
                        checkpoint_path.unlink()
                    except OSError as exc:
                        print(f"⚠️  Could not delete checkpoint {checkpoint_path}: {exc}")

        output_file_rel = relative_to_repo(output_path)
        if cleanup_json and output_path.exists():
            try:
                output_path.unlink()
            except OSError as exc:
                print(f"⚠️  Could not delete output JSON {output_path}: {exc}")

        summaries.append(
            {
                "run_index": start_index + run_idx - 1,
                "seed": run_seed,
                "runtime_seconds": elapsed,
                "breakthrough": exp_data.get("breakthrough_achieved", False),
                "final_consciousness": exp_data.get("final_consciousness", 0.0),
                "max_consciousness": exp_data.get("max_consciousness", 0.0),
                "final_phi": exp_data.get("final_phi", 0.0),
                "max_phi": exp_data.get("max_phi", 0.0),
                "output_file": output_file_rel,
                "output_file_retained": not cleanup_json,
                "checkpoints": checkpoint_records,
                "checkpoints_retained": not cleanup_checkpoints,
                "baseline_locked": runner.phi_baseline_stats.get("locked", False),
                "origin": "fresh",
            }
        )

    return summaries


def summarize_results(summaries: list) -> dict:
    """Compute aggregate statistics across runs."""
    if not summaries:
        return {}

    def collect(key: str) -> list:
        return [entry[key] for entry in summaries if key in entry and entry[key] is not None]

    def ci95(values: list) -> list:
        if len(values) < 2:
            return [values[0], values[0]] if values else [0.0, 0.0]
        mu = mean(values)
        sigma = stdev(values)
        stderr = sigma / math.sqrt(len(values)) if len(values) > 0 else 0.0
        delta = 1.96 * stderr
        return [mu - delta, mu + delta]

    agg = {
        "runs": len(summaries),
        "breakthrough_rate": sum(1 for entry in summaries if entry["breakthrough"]) / len(summaries),
        "final_consciousness_mean": mean(collect("final_consciousness")),
        "final_consciousness_std": pstdev(collect("final_consciousness")) if len(summaries) > 1 else 0.0,
        "final_consciousness_ci95": ci95(collect("final_consciousness")),
        "max_consciousness_mean": mean(collect("max_consciousness")),
        "max_consciousness_std": pstdev(collect("max_consciousness")) if len(summaries) > 1 else 0.0,
        "max_consciousness_ci95": ci95(collect("max_consciousness")),
        "final_phi_mean": mean(collect("final_phi")),
        "final_phi_std": pstdev(collect("final_phi")) if len(summaries) > 1 else 0.0,
        "final_phi_ci95": ci95(collect("final_phi")),
        "max_phi_mean": mean(collect("max_phi")),
        "max_phi_std": pstdev(collect("max_phi")) if len(summaries) > 1 else 0.0,
        "max_phi_ci95": ci95(collect("max_phi")),
        "runtime_seconds_mean": mean(collect("runtime_seconds")) if collect("runtime_seconds") else 0.0,
        "origins": {
            "fresh": sum(1 for entry in summaries if entry.get("origin") == "fresh"),
            "existing": sum(1 for entry in summaries if entry.get("origin") == "existing"),
        },
    }

    return agg


def save_summary(summaries: list, aggregate: dict, label: str) -> Path:
    """Persist per-run summaries and aggregates to JSON."""
    OUTPUTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = OUTPUTS_DIR / f"batch_infinito_summary_{label}_{timestamp}.json"

    payload = {
        "metadata": {
            "label": label,
            "generated_at": datetime.now().isoformat(),
        },
        "aggregate": aggregate,
        "runs": summaries,
    }

    with open(filename, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)

    return filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch runner for INFINITO V5.1 text experiments")
    parser.add_argument("--runs", type=int, default=10, help="Number of experiments to execute")
    parser.add_argument("--max_iter", type=int, default=100, help="Iterations per experiment run")
    parser.add_argument("--seed", type=int, default=12345, help="Base random seed")
    parser.add_argument("--input_text", type=str, default=DEFAULT_TEXT, help="Input text for the experiment")
    parser.add_argument("--label", type=str, default="default", help="Label used in the summary filename")
    parser.add_argument(
        "--cleanup_checkpoints",
        action="store_true",
        help="Delete breakthrough checkpoint .pt files generated during new runs",
    )
    parser.add_argument(
        "--cleanup_json",
        action="store_true",
        help="Delete per-run JSON logs after incorporating them into the batch summary",
    )
    parser.add_argument(
        "--aggregate_glob",
        type=str,
        default=None,
        help="Glob pattern (relative to repo root) of existing experiment JSON files to aggregate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("📊 Batch INFINITO V5.1 text experiments")
    print(f"   Runs: {args.runs}")
    print(f"   Iterations per run: {args.max_iter}")
    print(f"   Base seed: {args.seed}")
    print(f"   Label: {args.label}")
    print(f"   Text length: {len(args.input_text)} characters | {len(args.input_text.split())} words")
    if args.aggregate_glob:
        print(f"   Aggregate glob: {args.aggregate_glob}")
    if args.cleanup_checkpoints or args.cleanup_json:
        print(f"   Cleanup: checkpoints={args.cleanup_checkpoints}, json={args.cleanup_json}")
    print("=" * 70)

    if args.runs >= 100:
        print("⚠️  High run count detected. Expect long runtimes and many output files.")

    existing_summaries = load_existing_runs(args.aggregate_glob) if args.aggregate_glob else []
    if existing_summaries:
        print(f"   → Loaded {len(existing_summaries)} existing runs from pattern")

    summaries = existing_summaries

    if args.runs > 0:
        new_summaries = run_experiment_series(
            runs=args.runs,
            max_iter=args.max_iter,
            base_seed=args.seed,
            input_text=args.input_text,
            cleanup_checkpoints=args.cleanup_checkpoints,
            cleanup_json=args.cleanup_json,
            start_index=len(existing_summaries) + 1,
        )
        summaries += new_summaries
    else:
        print("   → No new experiments requested (--runs 0)")

    if not summaries:
        print("❌ No runs available for aggregation. Exiting.")
        return

    assign_run_indices(summaries)

    aggregate = summarize_results(summaries)
    summary_path = save_summary(summaries, aggregate, label=args.label)

    print("\n✅ Batch execution completed")
    print(f"   Aggregate summary: {aggregate}")
    print(f"   Detailed log saved to: {summary_path}")
    print("   Per-run output files listed in summary JSON")


if __name__ == "__main__":
    main()
