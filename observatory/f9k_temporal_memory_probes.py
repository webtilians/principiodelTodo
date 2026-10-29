"""F9-K: time-resolved linear probes of frozen F9-I recurrent/F4 memory.

Only S7 TRAIN teacher histories train the probe. S7 and zero-shot S13
DEVELOPMENT histories evaluate it; world IDs never cross splits. No model
weight is trained, no heldout accessor exists, and all paired cue inversions
share the same teacher-action prefix until their first divergent action.

Readouts are captured at the SAME pre-decision instant: recurrent state
before model.predict(frame[t]), F4 after encoding public frame[t]. At
post-cue checkpoints the current raw frame must be identical across both
cue variants; violations fail rather than being selectively excluded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9d_autonomous_navigation import checked_seed, demonstration
from observatory.f9i_selective_memory import SelectiveF4, DIM
from observatory.f9j_recurrent_memory_attribution import CAPACITY, frozen_action

STAGES = ("post_cue", "halfway", "fork")
FEATURES = {"context": ("H", "M", "H+M", "O"),
            "zero": ("H", "O")}
SOURCE_RUN = 40928547873
RIDGE_ALPHA = 10.0
HIDDEN_WIDTH = 128
TRAIN_PAIRS = 80
DEV_PAIRS = 10


def stage_indices(actions_a, actions_b, seen_a, seen_b):
    """Define evaluation times before inspecting a model's predictions."""
    fork = next((i for i, (a, b) in enumerate(zip(actions_a, actions_b))
                 if a != b), None)
    if fork is None:
        raise AssertionError("Cue inversion must change the teacher at a fork")
    if not np.array_equal(actions_a[:fork], actions_b[:fork]):
        raise AssertionError("Pre-fork teacher actions differ")
    if list(seen_a[:fork]) != list(seen_b[:fork]):
        raise AssertionError("Cue-visibility histories differ")
    visible = [i for i, value in enumerate(seen_a[:fork]) if value]
    if not visible:
        raise AssertionError("Teacher did not see cue before fork")
    after = max(visible) + 1
    if not after < fork:
        raise AssertionError("No post-cue interval before decision")
    return {"post_cue": after, "halfway": (after + fork) // 2, "fork": fork}


def flatten_hidden(state):
    """The frozen policy's predictive LSTM hidden/cell state, never critic labels."""
    if state is None:
        return np.zeros(HIDDEN_WIDTH, dtype=np.float32)
    out = np.concatenate([np.asarray(v, dtype=np.float32).reshape(-1)
                          for v in state])
    if out.shape != (HIDDEN_WIDTH,) or not np.isfinite(out).all():
        raise AssertionError("Unexpected frozen LSTM state dimensions")
    return out


def snapshot_history(model, mode, index, frames, times):
    if mode not in FEATURES:
        raise ValueError(mode)
    writer = SelectiveF4(mode, CAPACITY, seed=81000 + index)
    state = None
    snapshots = {}
    for t in range(max(times.values()) + 1):
        encoded = writer.encode(frames[t])
        if encoded.shape != (DIM,):
            raise AssertionError("Input dimension changed")
        if t in times.values():
            h = flatten_hidden(state)
            snapshot = {
                "H": h.copy(),
                "M": encoded[148:].copy(),
                "O": encoded[:148].copy(),
            }
            snapshot["H+M"] = np.concatenate((h, snapshot["M"]))
            for stage, pos in times.items():
                if pos == t:
                    snapshots[stage] = snapshot
        # The model cannot see the teacher action, cue variant, or stage label.
        _, state = frozen_action(model, encoded, state, first=t == 0)
    if set(snapshots) != set(STAGES):
        raise AssertionError("Incomplete time checkpoints")
    return snapshots


def paired_snapshots(model, mode, index, size, seed):
    frames_a, actions_a, vis_a = demonstration(seed, size=size, reverse=False)
    frames_b, actions_b, vis_b = demonstration(seed, size=size, reverse=True)
    times = stage_indices(actions_a, actions_b, vis_a, vis_b)
    if max(times.values()) >= min(len(frames_a), len(frames_b)):
        raise AssertionError("Missing teacher frame at fork")
    a = snapshot_history(model, mode, index, frames_a, times)
    b = snapshot_history(model, mode, index, frames_b, times)
    result = {}
    for stage in STAGES:
        if not np.array_equal(a[stage]["O"], b[stage]["O"]):
            raise AssertionError("Post-cue raw observation is not matched: " + stage)
        result[stage] = {
            "a": a[stage], "b": b[stage], "step": times[stage],
            "cue_gap": times[stage] - max(i for i,v in enumerate(vis_a[:times["fork"]]) if v),
            "raw_equal": True,
            "f4_equal": bool(np.array_equal(a[stage]["M"], b[stage]["M"])),
            "hidden_distance": float(np.linalg.norm(a[stage]["H"]-b[stage]["H"])),
        }
    return result


def build_dataset(model, mode, index, size, split, count):
    if split not in ("train", "development"):
        raise ValueError("F9-K never opens reserved worlds")
    if not 1 <= count <= (TRAIN_PAIRS if split == "train" else DEV_PAIRS):
        raise ValueError("Preassigned paired-world budget exceeded")
    per_stage = {stage: [] for stage in STAGES}
    for j in range(count):
        seed = checked_seed(split, index, j)
        pair = paired_snapshots(model, mode, index, size, seed)
        for stage in STAGES:
            per_stage[stage].append(pair[stage])
    return per_stage


def matrix(pairs, feature):
    """Paired world i contributes both inverted labels; no row-level splitting."""
    if feature not in ("H", "M", "H+M", "O"):
        raise ValueError(feature)
    x = np.stack([sample[variant][feature]
                  for sample in pairs for variant in ("a", "b")])
    y = np.tile(np.array([-1.0, 1.0], dtype=np.float64), len(pairs))
    return x.astype(np.float64), y


def fit_linear_probe(x, y, alpha=RIDGE_ALPHA):
    """Fixed regularization, trained on S7 train only; no dev threshold tuning."""
    if len(y) < 4 or len(y) % 2 or alpha <= 0:
        raise ValueError("Balanced train pairs and positive fixed ridge required")
    center = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-12] = 1.0
    z = (x - center) / scale
    # Dual ridge: work in N x N (faster for the fixed 512/640-d F4 channels).
    gram = z @ z.T
    dual = np.linalg.solve(gram + alpha * np.eye(len(y)), y)
    weight = z.T @ dual
    return center, scale, weight


def predict_probe(probe, x):
    center, scale, weight = probe
    return np.where(((x - center) / scale) @ weight >= 0., 1.0, -1.0)


def scores(probe, pairs, feature):
    x, y = matrix(pairs, feature)
    pred = predict_probe(probe, x)
    return {
        "accuracy": float(np.mean(pred == y)),
        "both_correct": int(np.sum((pred == y).reshape(-1, 2).all(axis=1))),
        "paired_flips": int(np.sum((pred.reshape(-1, 2)[:,0] !=
                                     pred.reshape(-1, 2)[:,1]))),
        "pairs": len(pairs),
        "equal_raw_pairs": int(sum(p["raw_equal"] for p in pairs)),
        "equal_f4_pairs": int(sum(p["f4_equal"] for p in pairs)),
        "mean_hidden_distance": float(np.mean([p["hidden_distance"] for p in pairs])),
        "mean_steps_since_last_cue": float(np.mean([p["cue_gap"] for p in pairs])),
    }


def evaluate_mode(model, mode, index, train_pairs, dev_pairs):
    # The auxiliary decoder is ALWAYS trained on S7; never on S13 data.
    train = build_dataset(model, mode, index, 7, "train", train_pairs)
    development = {
        str(size): build_dataset(model, mode, index, size, "development", dev_pairs)
        for size in (7, 13)
    }
    report = {}
    for stage in STAGES:
        for feature in FEATURES[mode]:
            x, y = matrix(train[stage], feature)
            probe = fit_linear_probe(x, y)
            for size in ("7", "13"):
                key = f"S{size}/{stage}/{feature}"
                report[key] = scores(probe, development[size][stage], feature)
                print("F9K_PROBE", index, mode, key, json.dumps(report[key]), flush=True)
    return report


def run(args):
    torch.set_num_threads(1)
    if args.index not in range(4):
        raise ValueError("Only original four training seeds")
    if not (1 <= args.train_pairs <= TRAIN_PAIRS and
            1 <= args.dev_pairs <= DEV_PAIRS):
        raise ValueError("Probe sample budget exceeded")
    source = Path(args.source)
    original = json.loads((source.parent / "report.json").read_text(encoding="utf8"))
    if (original["seed_index"] != args.index or
            original["event_capacity"] != CAPACITY or
            original["heldout_opened"] is not False):
        raise AssertionError("Source is not frozen F9-I capacity-8 development")
    if original["models"]["context"]["after_rl_sha256"] != file_sha(source):
        raise AssertionError("Context model SHA-256 mismatch")
    zero = source.parent / "zero_after_rl.zip"
    if original["models"]["zero"]["after_rl_sha256"] != file_sha(zero):
        raise AssertionError("Zero control SHA-256 mismatch")

    result = {
        "experiment": "F9-K frozen temporal memory linear probes",
        "source_run": SOURCE_RUN,
        "seed_index": args.index,
        "checkpoint_sha256": {"context": file_sha(source), "zero": file_sha(zero)},
        "agent_training_performed": False,
        "auxiliary_probe_training_performed": True,
        "heldout_opened": False,
        "probe_train_environment": "S7",
        "probe_train_partition": "train",
        "evaluation_partition": "development",
        "probe_regularization_alpha": RIDGE_ALPHA,
        "train_paired_worlds": args.train_pairs,
        "development_paired_worlds_per_size": args.dev_pairs,
        "modes": {},
    }
    for mode, path in (("context", source), ("zero", zero)):
        model = RecurrentPPO.load(str(path), device="cpu")
        try:
            result["modes"][mode] = evaluate_mode(
                model, mode, args.index, args.train_pairs, args.dev_pairs)
        finally:
            if model.get_env() is not None:
                model.get_env().close()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf8")
    print("F9K_COMPLETE", args.index, flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, choices=range(4), required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-pairs", type=int, default=TRAIN_PAIRS)
    p.add_argument("--dev-pairs", type=int, default=DEV_PAIRS)
    run(p.parse_args())


if __name__ == "__main__":
    main()
