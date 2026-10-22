"""Paired, end-to-end INFINITO V1/V5.1 runs (bounded duration).

Runs the ORIGINAL V1 run_infinite and ORIGINAL V5.1 run_experiment/train_step,
without changing either research implementation. The new *harness* uses smaller
V5.1 width/iteration count on CPU and records states via observation hooks.
No score is called a measure of subjective consciousness.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch

from scripts.infinito_replay_probes import load_version

SEEDS = (11, 29)
V1_MODES = ("reference", "laws_perturbed")
V51_MODES = ("reference", "hidden_reset", "memory_reset", "visual_causal_cut")


def initial_seed(seed):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def stats_state(state, old):
    value = np.asarray(state, dtype=np.float64).reshape(-1)
    prior = None if old is None else np.asarray(old, dtype=np.float64).reshape(-1)
    denom = max(float(np.linalg.norm(value)), 1e-12)
    return {
        "state_mean": float(value.mean()),
        "state_sd": float(value.std()),
        "state_norm": float(np.linalg.norm(value)),
        "state_change": 0.0 if prior is None else float(np.linalg.norm(value - prior) / max(np.linalg.norm(prior), 1e-9)),
        "lag_cosine": 0.0 if prior is None else float(np.dot(value, prior) / (max(np.linalg.norm(prior), 1e-12) * denom)),
    }


def run_v1(seed, mode, steps, shock, out):
    initial_seed(seed)
    scope = load_version("v1")
    original_class = scope["PrincipioTodoRecursivo"]
    model = original_class(size=16, max_depth=steps)
    old = None
    rows, states = [], []
    orig_method = model._one_recursion

    def recorder(input_grid):
        nonlocal old
        # Intervene only in one arm. No random calls; paired seeds stay aligned.
        if mode == "laws_perturbed" and model.recursion + 1 == shock:
            with torch.no_grad():
                for i in range(4):
                    model.leyes[i] = model.leyes[i] + torch.ones_like(model.leyes[i]) * 0.25
        field = orig_method(input_grid)
        arr = field.detach().float().cpu().numpy()[0, 0].copy()
        stats = stats_state(arr, old)
        hist, _ = np.histogram(arr, bins=20, range=(0.0, 1.0))
        probs = hist[hist > 0] / float(hist.sum())
        stats.update({
            "shannon": float(-(probs * np.log(probs)).sum()),
            "active_fraction": float((arr > 0.1).mean()),
            "neighbors_diff": float((np.abs(arr[1:, :] - arr[:-1, :]).mean() +
                                       np.abs(arr[:, 1:] - arr[:, :-1]).mean()) / 2),
            "law_norm": float(torch.stack(model.leyes).detach().norm().item()),
            "loss_reported": float(model.complexity_log[-1]["loss"]),
            "clusters_reported": float(model.complexity_log[-1]["clusters"]),
        })
        rows.append({"step": int(model.recursion), "metrics": stats})
        states.append(arr.ravel())
        old = arr
        return field

    model._one_recursion = recorder
    with (out / "native.log").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(log):
            model.run_infinite()
    return rows, np.asarray(states, dtype=np.float32)


def internal_module_hook(name, cache, mode, iteration, shock):
    def hook(module, args, result):
        with torch.no_grad():
            cache[name] = result[0].detach().mean(dim=(0, 1)).float().cpu().numpy()
        if mode == "visual_causal_cut" and name == "visual" and iteration[0] >= shock:
            return result[0], torch.zeros_like(result[1])
        return None
    return hook


def run_v51(seed, mode, steps, shock, out, hidden_dim):
    initial_seed(seed)
    mod = load_version("v51")
    # Full original four-module V5.1, compact width and 24-iteration horizon.
    args = argparse.Namespace(
        lr=0.001, seed=seed, batch_size=4,
        input_dim=257, hidden_dim=hidden_dim,
        attention_heads=8, memory_slots=16,
        text_mode=False, input_text=None,
    )
    trainer = mod.InfinitoV51ConsciousnessBreakthrough(args)
    # Historical constructor omits this logging key.
    trainer.metrics_history.setdefault("biological_plausibility", deque(maxlen=2000))
    # Historical constructor adds another optimizer group after scheduler init.
    if len(trainer.scheduler.base_lrs) != len(trainer.optimizer.param_groups):
        trainer.scheduler.base_lrs = [group["lr"] for group in trainer.optimizer.param_groups]
        trainer.scheduler._last_lr = list(trainer.scheduler.base_lrs)
    model = trainer.model
    state_cache, iteration = {}, [0]
    handles = []
    for name in ("visual", "auditory", "motor", "executive"):
        handles.append(getattr(model, name + "_module").register_forward_hook(
            internal_module_hook(name, state_cache, mode, iteration, shock)))
    rows, states = [], []
    old = None
    original_step = trainer.train_step

    def record_step(step):
        nonlocal old
        iteration[0] = step
        if step == shock:
            with torch.no_grad():
                if mode == "hidden_reset":
                    for name in ("visual", "auditory", "motor", "executive"):
                        module = getattr(model, name + "_module")
                        module.h_state.zero_()
                        module.c_state.zero_()
                elif mode == "memory_reset":
                    model.memory.memory.zero_()
                    model.memory.memory_age.zero_()
                    model.memory.memory_strength.fill_(0.1)
        result = original_step(step)  # all native train-step pathways execute
        states4 = np.stack([state_cache[name] for name in
                            ("visual", "auditory", "motor", "executive")])
        flat = states4.reshape(-1)
        features = stats_state(flat, old)
        norms = np.linalg.norm(states4, axis=1)
        s = states4 / np.maximum(norms[:, None], 1e-9)
        sim = s @ s.T
        features.update({
            "module_differentiation": float(np.mean(1 - sim[np.triu_indices(4, 1)])),
            "module_output_sd": float(states4.std()),
            "memory_norm": float(model.memory.memory.detach().float().norm().item()),
            "lstm_hidden_norm": float(sum(getattr(model, name + "_module").h_state.detach().float().norm().item()
                                        for name in ("visual", "auditory", "motor", "executive"))),
            "loss_reported": float(result["total_loss"]),
            "memory_utilization_reported": float(result["memory_utilization"]),
        })
        rows.append({"step": int(step), "metrics": features})
        states.append(flat.astype(np.float32))
        old = flat.copy()
        print("RECORDED", mode, seed, step, flush=True)
        return result

    trainer.train_step = record_step
    os.makedirs("outputs", exist_ok=True)
    try:
        with (out / "native.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(log):
                trainer.run_experiment(max_iterations=steps)
    finally:
        for handle in handles:
            handle.remove()
    return rows, np.asarray(states, dtype=np.float32)


def paired_analysis(paths, family, modes, seeds, shock):
    comparisons = []
    for seed in seeds:
        ref_file = paths / ("%s_reference_s%s" % (family, seed)) / "states.npy"
        if not ref_file.exists():
            continue
        ref = np.load(ref_file)
        for mode in modes[1:]:
            test_file = paths / ("%s_%s_s%s" % (family, mode, seed)) / "states.npy"
            if not test_file.exists():
                continue
            alt = np.load(test_file)
            n = min(len(ref), len(alt))
            if n < shock + 2:
                continue
            # Normalize by reference scale to compare change WITHIN each family.
            scale = max(float(np.median(np.linalg.norm(ref[:shock], axis=1))), 1e-9)
            diff = np.linalg.norm(ref[:n].astype(np.float64) - alt[:n].astype(np.float64), axis=1) / scale
            before = float(np.max(diff[:shock - 1])) if shock > 1 else 0.0
            after = diff[shock - 1:]
            peak = float(np.max(after))
            first = float(after[0])
            last = float(after[-1])
            comparisons.append({
                "family": family, "seed": seed, "intervention": mode,
                "n_paired": n, "shock": shock,
                "pre_intervention_max_divergence": before,
                "post_first_divergence": first,
                "post_peak_divergence": peak,
                "post_last_divergence": last,
                "recovered_to_under_20pct_initial": bool(first > 1e-8 and last <= 0.2 * first),
                "effect_detected": bool(peak > 1e-6),
                "warning": "recovery defined relative to a finite reference run, not proof of resilience",
            })
    return comparisons


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("observatory_results/full_comparison"))
    parser.add_argument("--v1-steps", type=int, default=200)
    parser.add_argument("--v51-steps", type=int, default=24)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--v1-only", action="store_true")
    parser.add_argument("--v51-only", action="store_true")
    opts = parser.parse_args()
    opts.output.mkdir(parents=True, exist_ok=True)
    manifests, aggregated = [], []
    experiments = []
    if not opts.v51_only:
        experiments.extend(("v1", seed, mode, opts.v1_steps, max(2, opts.v1_steps // 2))
                           for seed in SEEDS for mode in V1_MODES)
    if not opts.v1_only:
        experiments.extend(("v51", seed, mode, opts.v51_steps, max(2, opts.v51_steps // 2))
                           for seed in SEEDS for mode in V51_MODES)
    root = Path.cwd()
    for family, seed, mode, steps, shock in experiments:
        label = "%s_%s_s%d" % (family, mode, seed)
        folder = opts.output.resolve() / label
        folder.mkdir(exist_ok=True, parents=True)
        os.chdir(folder)
        try:
            if family == "v1":
                rows, states = run_v1(seed, mode, steps, shock, folder)
            else:
                rows, states = run_v51(seed, mode, steps, shock, folder, opts.hidden_dim)
            np.save(folder / "states.npy", states)
            aggregated.extend({"run_id": label, **row} for row in rows)
            manifests.append({"run_id": label, "status": "ok", "observations": len(rows),
                              "state_shape": list(states.shape), "shock": shock})
        except Exception as error:
            import traceback
            (folder / "error.txt").write_text(traceback.format_exc())
            manifests.append({"run_id": label, "status": "error",
                              "error": str(error), "shock": shock})
        finally:
            os.chdir(root)
        print("FINISHED", label, manifests[-1]["status"], flush=True)
    comparison = []
    if not opts.v51_only:
        comparison += paired_analysis(opts.output, "v1", V1_MODES, SEEDS, max(2, opts.v1_steps // 2))
    if not opts.v1_only:
        comparison += paired_analysis(opts.output, "v51", V51_MODES, SEEDS, max(2, opts.v51_steps // 2))
    result = {
        "design": "original_native_training_paths_short_horizon",
        "warning": "V5.1 width 64 (vs historical 512); 24/15000 iterations; V1 fixed input; no cross-family state similarity claims",
        "seeds": SEEDS, "v1_steps": opts.v1_steps, "v51_steps": opts.v51_steps,
        "v51_hidden_dim": opts.hidden_dim, "runs": manifests, "interventions": comparison,
    }
    (opts.output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    with (opts.output / "observations.jsonl").open("w") as handle:
        for point in aggregated:
            handle.write(json.dumps(point, allow_nan=False) + "\n")
    print("SUMMARY", json.dumps({
        "ok": sum(v["status"] == "ok" for v in manifests),
        "total": len(manifests), "observations": len(aggregated),
        "comparisons": len(comparison),
    }), flush=True)
    if any(v["status"] != "ok" for v in manifests):
        sys.exit(1)


if __name__ == "__main__":
    main()
