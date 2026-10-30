"""Causal diagnostic: does replacing memory with distractors erase the initial cue?

Reuse the original, verified V5.1 forward probe without changing source
or the established baseline. Monkeypatch only external memory.write during
intervention episodes, restoring the method after every episode.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import torch

import scripts.infinito_v51_memory_feedback as experiment

NEW_MODES = (
    "historical",
    "historical_memory_cut",
    "historical_write_frozen",
    "early_feedback",
    "early_feedback_memory_cut",
    "early_feedback_write_frozen",
)

original_one_episode = experiment.one_episode


def with_freeze(model, wrapper, frames, *, mode, stochastic_seed):
    if not mode.endswith("write_frozen"):
        return original_one_episode(model, wrapper, frames,
                                    mode=mode, stochastic_seed=stochastic_seed)
    store = model.memory
    original_write = store.write
    count = 0

    def first_write_only(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 1:
            return original_write(*args, **kwargs)
        return None

    store.write = first_write_only
    try:
        return original_one_episode(model, wrapper, frames,
                                    mode=mode, stochastic_seed=stochastic_seed)
    finally:
        store.write = original_write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path,
                        default=Path("observatory_results/memory_overwrite"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--train", type=int, default=16)
    parser.add_argument("--test", type=int, default=16)
    parser.add_argument("--steps", type=int, default=17)
    parser.add_argument("--distractor-sd", type=float, default=0.55)
    parser.add_argument("--gain", type=float, default=0.5)
    args = parser.parse_args()
    if args.steps < 3 or args.steps > 64 or args.train % 2 or args.test % 2:
        raise ValueError("invalid steps/train/test")
    experiment.MODES = NEW_MODES
    experiment.one_episode = with_freeze
    torch.set_num_threads(1)
    results = []
    for seed in args.seeds:
        print("RUN", seed, flush=True)
        x = experiment.run_seed(seed, args.train, args.test, args.gain,
                                args.steps, args.distractor_sd, 0.7)
        results.append(x)
        print("DONE", seed, {
            mode: round(result["heldout_cue_accuracy"], 4)
            for mode, result in x["modes"].items()
        }, flush=True)
    payload = {
        "protocol": "frozen_memory_write_after_first_cue",
        "intervention": (
            "First memory write enabled; all subsequent memory.write calls "
            "return without mutation. Read and LSTM evolution remain active."
        ),
        "parameters": vars(args) | {"output": str(args.output)},
        "results": results,
        "mean_accuracy": {
            mode: mean(x["modes"][mode]["heldout_cue_accuracy"] for x in results)
            for mode in NEW_MODES
        },
        "limitations": (
            "Original model untrained; linear probe learns only a decoder. "
            "Memory freezing modifies the software's write policy on purpose, "
            "not a spontaneous adaptive response. No claims of consciousness."
        ),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(payload, indent=2))
    print("RESULT", json.dumps(payload["mean_accuracy"]), flush=True)


if __name__ == "__main__":
    main()
