"""F9-L: interrupt external F4 after a cue and quantify recurrent survival.

All policy checkpoints remain frozen. Native teacher action prefix is matched
between key/ball counterfactuals until the first cue-dependent action. Only
the 512 external-memory *actor input* channels are interrupted on steps
[cutoff, fork), with the ORIGINAL memory readout restored at the fork.

Two probes: one trained on natural S7 train histories and held fixed after
the interruption, another trained on interrupted S7 train histories.
Dev S7/S13 never trains either probe. No heldout access and no new policy.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9d_autonomous_navigation import checked_seed, demonstration
from observatory.f9i_selective_memory import SelectiveF4, DIM, FIELDS
from observatory.f9j_recurrent_memory_attribution import CAPACITY, frozen_action
from observatory.f9k_temporal_memory_probes import (
    RIDGE_ALPHA, SOURCE_RUN, TRAIN_PAIRS, DEV_PAIRS,
    flatten_hidden, fit_linear_probe, matrix, original_cue_label,
    predict_probe, scores, stage_indices,
)

CONDITIONS = (
    "normal",
    "cut_post_cue",
    "cut_halfway",
    "cut_last_step",
    "neutral_tail",
)
ZERO_CONDITIONS = ("normal",)
# Context8 only fills the first 8 * 8 channels; 448 remaining are padding.
ACTIVE_MEMORY_END = 148 + CAPACITY * FIELDS


def cutoff_for(condition, times):
    """Predeclared schedule. No dependence on predictions, rewards or labels."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    fork = times["fork"]
    cutoffs = {
        "normal": None,
        "cut_post_cue": times["post_cue"],
        "cut_halfway": times["halfway"],
        "cut_last_step": fork - 1,
        "neutral_tail": times["post_cue"],
    }
    cut = cutoffs[condition]
    if cut is not None and not 0 <= cut < fork:
        raise AssertionError("No actual pre-fork interruption")
    return cut


def run_forced_prefix(model, mode, index, frames, times, condition):
    """Read public frames; call actor.predict only, NEVER teacher labels.

    Writer sees the original public frames at all times; only the actor's
    input vector is masked. At fork, feed the natural readout (restoration).
    """
    if mode not in ("context", "zero") or (
        mode == "zero" and condition != "normal"):
        raise ValueError("Only original zero model is unaltered control")
    cutoff = cutoff_for(condition, times)
    writer = SelectiveF4(mode, CAPACITY, seed=81000 + index)
    state = None
    applied = 0
    effective = 0
    for t in range(times["fork"] + 1):
        natural = writer.encode(frames[t])
        if natural.shape != (DIM,) or not np.isfinite(natural).all():
            raise AssertionError("Frozen policy input contract changed")
        actor_obs = natural.copy()
        if cutoff is not None and cutoff <= t < times["fork"]:
            if condition == "neutral_tail":
                if np.any(natural[ACTIVE_MEMORY_END:]):
                    raise AssertionError("Supposed-zero F4 padding not zero")
                actor_obs[ACTIVE_MEMORY_END:] = 0.
            else:
                actor_obs[148:] = 0.
            applied += 1
            effective += int(not np.array_equal(actor_obs, natural))
        if t == times["fork"]:
            if not np.array_equal(actor_obs, natural):
                raise AssertionError("External F4 must be restored at fork")
            h = flatten_hidden(state)
            action, _ = frozen_action(model, actor_obs, state, first=t == 0)
            return {
                "H": h,
                "M": natural[148:].copy(),
                "O": natural[:148].copy(),
                "predicted_action": action,
                "applied_steps": applied,
                "effective_steps": effective,
                "cutoff": cutoff,
                "fork_step": times["fork"],
                "mask_duration": 0 if cutoff is None else times["fork"] - cutoff,
                "f4_restored": True,
            }
        _, state = frozen_action(model, actor_obs, state, first=t == 0)
    raise AssertionError("Fork never evaluated")


def paired_case(model, mode, index, size, seed, condition):
    """A matched original/inverted world; oracle action only in evaluator."""
    first, actions_a, visibility_a = demonstration(seed, size=size, reverse=False)
    second, actions_b, visibility_b = demonstration(seed, size=size, reverse=True)
    times = stage_indices(actions_a, actions_b, visibility_a, visibility_b)
    fork = times["fork"]
    if not fork < len(first) or not fork < len(second):
        raise AssertionError("Missing frame at first decision fork")
    a = run_forced_prefix(model, mode, index, first, times, condition)
    b = run_forced_prefix(model, mode, index, second, times, condition)
    if not np.array_equal(a["O"], b["O"]):
        raise AssertionError("Fork raw observations differ across cues")
    if not np.array_equal(a["M"], b["M"]):
        raise AssertionError("Fork external readouts differ across cues")
    if not np.array_equal(a["O"], first[fork]):
        raise AssertionError("Actor current observation is not public")
    if not a["f4_restored"] or not b["f4_restored"]:
        raise AssertionError("Fork memory not restored")
    if a["applied_steps"] != b["applied_steps"]:
        raise AssertionError("Counterfactual interruption schedule differs")
    if (condition.startswith("cut_") and
            (a["effective_steps"] == 0 or b["effective_steps"] == 0)):
        raise AssertionError("No F4 information actually removed")
    if mode == "zero" and any(np.any(v["M"]) for v in (a,b)):
        raise AssertionError("Control zero stored external F4")
    cue = original_cue_label(seed, size)
    return {
        "a": a, "b": b, "labels": (cue, -cue),
        "raw_equal": True, "f4_equal": True,
        "hidden_distance": float(np.linalg.norm(a["H"] - b["H"])),
        "cue_gap": fork - max(i for i,visible in enumerate(visibility_a[:fork])
                               if visible),
        "actions_correct": (
            int(a["predicted_action"] == int(actions_a[fork])),
            int(b["predicted_action"] == int(actions_b[fork])),
        ),
        "predicted_action_flip": int(a["predicted_action"] != b["predicted_action"]),
        "masked_steps": a["applied_steps"],
        "effective_steps": a["effective_steps"] + b["effective_steps"],
    }


def build_dataset(model, mode, index, size, split, count, condition):
    """No holdout route; pairs are indivisible, never split by individual row."""
    if split not in ("train", "development"):
        raise ValueError("Restricted to train and development partitions")
    if not 0 <= index < 4 or size not in (7,13):
        raise ValueError("Unapproved training seed/environment")
    if not 1 <= count <= (TRAIN_PAIRS if split == "train" else DEV_PAIRS):
        raise ValueError("Fixed budget exceeded")
    if split == "train" and size != 7:
        raise ValueError("No S13 examples may train a probe")
    return [
        paired_case(model, mode, index, size,
                    checked_seed(split, index, j), condition)
        for j in range(count)
    ]


def action_scores(pairs):
    return {
        "action_accuracy": float(np.mean([v for pair in pairs
                                          for v in pair["actions_correct"]])),
        "both_teacher_actions_correct": int(sum(
            all(pair["actions_correct"]) for pair in pairs)),
        "predicted_action_flips": int(sum(
            pair["predicted_action_flip"] for pair in pairs)),
        "pairs": len(pairs),
        "mean_masked_steps": statistics.mean(p["masked_steps"] for p in pairs),
        "mean_effective_masked_steps_both": statistics.mean(
            p["effective_steps"] for p in pairs),
    }


def assert_neutral_exact(normal, neutral):
    if len(normal) != len(neutral):
        raise AssertionError("Neutral matched worlds differ in count")
    for p, q in zip(normal, neutral):
        if p["labels"] != q["labels"]:
            raise AssertionError("Counterfactual cue identities changed")
        for side in ("a","b"):
            for key in ("H","M","O"):
                if not np.array_equal(p[side][key],q[side][key]):
                    raise AssertionError("Neutral padding changed "+key)
            if p[side]["predicted_action"] != q[side]["predicted_action"]:
                raise AssertionError("Neutral padding changed action")
        if q["effective_steps"] != 0:
            raise AssertionError("Control touched nonzero channels")


def evaluate_model(model, mode, index, train_pairs, dev_pairs):
    conditions = CONDITIONS if mode == "context" else ZERO_CONDITIONS
    train = {
        condition: build_dataset(model,mode,index,7,"train",train_pairs,condition)
        for condition in conditions
    }
    development = {
        str(size): {
            condition: build_dataset(model,mode,index,size,"development",
                                     dev_pairs,condition)
            for condition in conditions
        } for size in (7,13)
    }
    if mode == "context":
        assert_neutral_exact(train["normal"],train["neutral_tail"])
        for size in ("7","13"):
            assert_neutral_exact(development[size]["normal"],
                                 development[size]["neutral_tail"])

    baseline_probe = fit_linear_probe(*matrix(train["normal"],"H"))
    output = {}
    for condition in conditions:
        adapted_probe = fit_linear_probe(*matrix(train[condition],"H"))
        for size in ("7","13"):
            cases = development[size][condition]
            metric = {
                "fixed_probe": scores(baseline_probe,cases,"H"),
                "adapted_probe": scores(adapted_probe,cases,"H"),
                "policy_at_fork": action_scores(cases),
            }
            if metric["fixed_probe"]["equal_f4_pairs"] != dev_pairs:
                raise AssertionError("Not paired at fork")
            output[f"S{size}/{condition}"] = metric
            print("F9L_RESULT",index,mode,f"S{size}",condition,
                  json.dumps(metric),flush=True)
    return output


def run(args):
    torch.set_num_threads(1)
    if not 0 <= args.index < 4:
        raise ValueError("Only four original frozen models")
    if not (1 <= args.train_pairs <= TRAIN_PAIRS and
            1 <= args.dev_pairs <= DEV_PAIRS):
        raise ValueError("Preassigned probe budget exceeded")
    source = Path(args.source)
    orig = json.loads((source.parent/"report.json").read_text(encoding="utf8"))
    if (orig.get("seed_index") != args.index or
            orig.get("event_capacity") != CAPACITY or
            orig.get("heldout_opened") is not False):
        raise AssertionError("Source mismatch or heldout opened")
    zero = source.parent / "zero_after_rl.zip"
    if (orig["models"]["context"]["after_rl_sha256"] != file_sha(source) or
            orig["models"]["zero"]["after_rl_sha256"] != file_sha(zero)):
        raise AssertionError("Frozen F9-I SHA-256 verification failed")

    result = {
        "experiment":"F9-L external-memory interruption, recurrent survival",
        "source_run": SOURCE_RUN,
        "seed_index": args.index,
        "checkpoints_sha256": {"context":file_sha(source),"zero":file_sha(zero)},
        "policy_training":False,
        "auxiliary_probe_training":True,
        "heldout_opened":False,
        "train_split":"train S7",
        "eval_split":"development S7/S13",
        "train_pairs":args.train_pairs,
        "dev_pairs":args.dev_pairs,
        "ridge_alpha":RIDGE_ALPHA,
        "fork_readout_restored":True,
        "models":{},
    }
    for mode, path in (("context",source),("zero",zero)):
        model = RecurrentPPO.load(str(path),device="cpu")
        try:
            result["models"][mode]=evaluate_model(
                model,mode,args.index,args.train_pairs,args.dev_pairs)
        finally:
            env=model.get_env()
            if env is not None:
                env.close()

    # Reproduce frozen natural F9-K S7/S13 H/fork *per seed*, not cherry-picked.
    k_report=Path("docs/INFINITO_F9K_RESULTS_2026-10-29.json")
    k=json.loads(k_report.read_text(encoding="utf8"))["metrics"]
    for size in (7,13):
        for mode in ("context","zero"):
            obtained=result["models"][mode][f"S{size}/normal"]["fixed_probe"]["accuracy"]
            expected=k[f"S{size}/{mode}/fork/H"]["accuracy_by_seed"][args.index]
            if abs(obtained-expected)>1e-12:
                raise AssertionError("Natural F9-K H baseline changed: "+
                                     str((mode,size,obtained,expected)))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(
        json.dumps(result,indent=2),encoding="utf8")
    print("F9L_COMPLETE",args.index,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--train-pairs",type=int,default=TRAIN_PAIRS)
    p.add_argument("--dev-pairs",type=int,default=DEV_PAIRS)
    run(p.parse_args())


if __name__=="__main__":
    main()
