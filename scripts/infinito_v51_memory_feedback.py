"""INFINITO V5.1 — early memory feedback and held-out delayed-cue probes.

Imports the ORIGINAL ConsciousnessBoostNet; never modifies its source.
No conscious-experience claims. A trainable *readout probe* is used, not a
retrained brain; readout accuracy means decodability, not spontaneous recall.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.infinito_replay_probes import load_version

MODES = ("historical", "historical_memory_cut", "early_feedback", "early_feedback_memory_cut")


class EarlyMemoryFeedback:
    """Read the historical memory and inject a fixed projection before the 4 modules.

    Controls share all original weights and inputs. No new reward or targets
    enter the historical model. Both interventions and observations are hooks.
    """
    def __init__(self, model, *, gain: float):
        self.model = model
        self.gain = float(gain)
        self.read = None
        self.workspace = None
        self.handles = []
        hidden = model.visual_module.hidden_dim
        memory_dim = model.memory.slot_size
        projection = torch.zeros(hidden, memory_dim)
        for i in range(min(hidden, memory_dim)):
            projection[i, i] = 1.0
        self.projection = projection

    def __enter__(self):
        def capture_memory(module, arguments, output):
            self.read = output[0].detach()
        def capture_workspace(module, arguments, output):
            self.workspace = output.detach()
        def prepend_memory(module, args):
            if self.gain == 0 or self.read is None:
                return None
            projected = torch.nn.functional.linear(
                self.read, self.projection.to(self.read.device))
            return (args[0] + self.gain * projected[:, None, :], *args[1:])
        self.handles.append(self.model.memory.register_forward_hook(capture_memory))
        self.handles.append(
            self.model.consciousness_processor[-1].register_forward_hook(capture_workspace))
        for name in ("visual_module", "auditory_module", "motor_module", "executive_module"):
            self.handles.append(getattr(self.model, name).register_forward_pre_hook(prepend_memory))
        return self

    def __exit__(self, exc_type, exc, tb):
        for handle in self.handles:
            handle.remove()


def reset_episode(model):
    """Reset actual buffers, recurrent state and historical statistics."""
    with torch.no_grad():
        for name in ("visual_module", "auditory_module", "motor_module", "executive_module"):
            module = getattr(model, name)
            module.h_state.zero_()
            module.c_state.zero_()
        store = model.memory
        store.memory.zero_()
        store.memory_age.zero_()
        store.memory_strength.fill_(0.1)
        store.memory_utilization_tracker.zero_()
        store.iteration_counter = 0
        store.prev_phi = 0.0
        store.phi_delta_history.clear()
        model.phi_prev = None
        if hasattr(model, "phi_prev_scalar"):
            del model.phi_prev_scalar
        if hasattr(model, "states_prev"):
            del model.states_prev
        model.consciousness_history.clear()
        model.phi_calculator.temporal_state_history.clear() if hasattr(
            model.phi_calculator, "temporal_state_history") else None
        model.noether_proxy.prev_causal_density = None
        model.noether_proxy.prev_phi = None


def structured_input(seed: int, label: int, *, steps: int = 5, distractor_sd: float = 0.12, batch: int = 2, seq_len: int = 5):
    """One cue; three independent distractors; identical-label-free query.

    Target only appears in frame 0. The final frame has exactly the same
    distribution whatever the cue and cannot leak the answer directly.
    """
    rng = np.random.default_rng(seed)
    frames = []
    for step in range(steps):
        frame = rng.normal(0, distractor_sd, size=(batch, seq_len, 257)).astype("float32")
        if step == 0:
            frame[:, :, :64] += (1 if label else -1) * 0.7
        if step == steps - 1:
            frame[:, :, :256] = 0.0
        frame[:, :, -1] = np.linspace(0, 1, seq_len, dtype="float32")[None, :]
        frames.append(torch.from_numpy(frame))
    return frames


def one_episode(model, wrapper, frames, *, mode: str, stochastic_seed: int):
    torch.manual_seed(stochastic_seed)
    np.random.seed(stochastic_seed % (2**32 - 1))
    random.seed(stochastic_seed)
    reset_episode(model)
    features, reads, memory_norm = [], [], []
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        for step, frame in enumerate(frames):
            if mode.endswith("memory_cut") and step >= 1:
                # The early recurrent pathway remains active. Remove only
                # the memory-store contribution after the cue.
                model.memory.memory.zero_()
                model.memory.memory_age.zero_()
                model.memory.memory_strength.fill_(0.1)
            model(frame)
            assert wrapper.workspace is not None and wrapper.read is not None
            features.append(wrapper.workspace.mean(dim=0).cpu().numpy().copy())
            reads.append(wrapper.read.mean(dim=0).cpu().numpy().copy())
            memory_norm.append(float(model.memory.memory.norm().item()))
    return np.asarray(features), np.asarray(reads), memory_norm


def ridge_probe(train_x, train_y, test_x, test_y, *, reg: float = 0.5):
    """Freeze all feature generators; fit linear decoding weights on train only."""
    train_x, test_x = np.asarray(train_x), np.asarray(test_x)
    mean = train_x.mean(axis=0)
    sd = np.maximum(train_x.std(axis=0), 1e-4)
    a = (train_x - mean) / sd
    b = (test_x - mean) / sd
    # Dual ridge is fast even when the representation is wide.
    target = np.asarray(train_y, dtype=float) * 2 - 1
    kernel = a @ a.T + np.eye(len(a)) * reg
    coef = a.T @ np.linalg.solve(kernel, target)
    predictions = (b @ coef >= 0).astype(int)
    return float(np.mean(predictions == np.asarray(test_y)))


def run_seed(model_seed: int, train_count: int, test_count: int, gain: float, steps: int, distractor_sd: float):
    torch.manual_seed(model_seed)
    np.random.seed(model_seed)
    random.seed(model_seed)
    mod = load_version("v51")
    model = mod.ConsciousnessBoostNet(
        input_dim=257, hidden_dim=64, attention_heads=8, memory_slots=16).eval()
    # Same initialized model for all modes. Only the forward hooks vary.
    labels = [n % 2 for n in range(train_count + test_count)]
    episode_seeds = [model_seed * 100000 + i * 97 + 7 for i in range(len(labels))]
    # The same frame sequence is supplied to all variants for each case.
    cases = [structured_input(seed, label, steps=steps, distractor_sd=distractor_sd) for seed, label in zip(episode_seeds, labels)]
    modes = {}
    final_states = {}
    for mode in MODES:
        final, queries, store_norms = [], [], []
        with EarlyMemoryFeedback(model, gain=0.0 if mode.startswith("historical") else gain) as wrapper:
            for idx, frames in enumerate(cases):
                states, reads, norms = one_episode(
                    model, wrapper, frames, mode=mode,
                    stochastic_seed=model_seed * 100003 + idx + 17,
                )
                final.append(states[-1])
                queries.append(reads[-1])
                store_norms.append(norms[-1])
        final = np.asarray(final)
        final_states[mode] = final
        train_x, test_x = final[:train_count], final[train_count:]
        train_y, test_y = labels[:train_count], labels[train_count:]
        accuracy = ridge_probe(train_x, train_y, test_x, test_y)
        read_array = np.asarray(queries)
        memory_accuracy = ridge_probe(
            read_array[:train_count], train_y, read_array[train_count:], test_y)
        modes[mode] = {
            "heldout_cue_accuracy": accuracy,
            "heldout_memory_read_accuracy": memory_accuracy,
            "mean_final_memory_norm": float(np.mean(store_norms)),
            "final_state_sd": float(final.std()),
            "train_cases": train_count,
            "test_cases": test_count,
            "samples_per_cue_class": len(labels) // 2,
            "feature_dim": int(final.shape[-1]),
            "interpretation": "frozen-model linear cue decodability, not trained cognition",
        }
    comparisons = {}
    reference = final_states["historical"]
    for mode in MODES[1:]:
        other = final_states[mode]
        scale = np.maximum(np.linalg.norm(reference, axis=-1), 1e-8)
        delta = np.linalg.norm(reference - other, axis=-1) / scale
        comparisons[mode] = {
            "mean_pairwise_query_divergence": float(delta.mean()),
            "max_pairwise_query_divergence": float(delta.max()),
            "fraction_nonzero_at_1e-6": float((delta > 1e-6).mean()),
        }
    ablation = final_states["early_feedback"] - final_states["early_feedback_memory_cut"]
    comparisons["ablation"] = {
        "mean_l2_difference_feedback_vs_memory_cut": float(
            np.linalg.norm(ablation, axis=-1).mean()),
        "mean_l2_difference_historical_vs_memory_cut": float(
            np.linalg.norm(final_states["historical"] - final_states["historical_memory_cut"], axis=-1).mean()),
    }
    return {
        "seed": model_seed, "modes": modes, "comparisons": comparisons,
        "warning": (
            "No original V5.1 training. Probe-trained linear classifier only. "
            "Memory feedback fixed identity projection; interventions in inference."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("observatory_results/memory_feedback"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 29, 47])
    parser.add_argument("--train", type=int, default=24)
    parser.add_argument("--test", type=int, default=24)
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--distractor-sd", type=float, default=0.12)
    args = parser.parse_args()
    if args.train < 4 or args.test < 4 or args.train % 2 or args.test % 2:
        raise ValueError("train/test counts must be even and at least 4")
    if args.steps < 3 or args.steps > 64 or not (0 <= args.distractor_sd <= 2):
        raise ValueError("steps must be 3..64 and distractor_sd 0..2")
    if not (0 <= args.gain <= 2):
        raise ValueError("feedback gain must be between 0 and 2")
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in args.seeds:
        print("RUN", seed, flush=True)
        result = run_seed(seed, args.train, args.test, args.gain, args.steps, args.distractor_sd)
        results.append(result)
        print("DONE", seed, {
            key: round(value["heldout_cue_accuracy"], 3)
            for key, value in result["modes"].items()
        }, flush=True)
    summary = {
        "experiment": "V5.1 original untrained forward with controlled early memory feedback",
        "architecture": "ConsciousnessBoostNet hidden_dim 64; 4 original modules; 16 memory slots",
        "stimulus": "signed cue -> 3 distractors -> label-free query",
        "protocol": {"train": args.train, "test": args.test, "gain": args.gain,
                     "seeds": args.seeds, "steps": args.steps, "distractor_sd": args.distractor_sd,
                     "query_step": args.steps - 1},
        "results": results,
        "mean_accuracy": {
            mode: float(np.mean([result["modes"][mode]["heldout_cue_accuracy"]
                                 for result in results]))
            for mode in MODES
        },
        "limitations": (
            "Readout training only; no learned goals or intrinsic curiosity. "
            "Cue information can remain in recurrent states. "
            "A few seeds cannot establish population-general effects. "
            "A difference is not proof of functional memory improvement. "
            "V5.1 64/16 scaled down from historic 512/256."
        ),
    }
    (args.output / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    print("RESULT", json.dumps(summary["mean_accuracy"]), flush=True)


if __name__ == "__main__":
    main()
