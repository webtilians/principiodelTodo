"""F9-M one-shot confirmation on a NEW world partition outside development.

Only called by the GitHub Actions confirmation job after ALL four paired
training jobs succeed and their policy/parameter budgets have been verified.
Original heldout (900000...) remains unopened. This is a distinct prospective
validation split (300000...), not F9-I/J/K/L development and not train.

Primary: paired autonomous performance with F4 normal or erased after cue.
Secondary: matched teacher forks, F9-L cutoffs, frozen/adapted linear probes.
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
from observatory.f9j_recurrent_memory_attribution import autonomous_episode
from observatory.f9k_temporal_memory_probes import fit_linear_probe, matrix, scores
from observatory.f9l_f4_interruption import paired_case, action_scores
from observatory.f9m_temporal_dropout_training import (
    ARMS, TRAIN_WORLDS, TRAIN_EPOCHS, PPO_STEPS, DIM, CAPACITY,
)

CONFIRMATION_BASE = 300_000
CONFIRMATION_PAIRS = 20
PROBE_TRAIN_PAIRS = 80
TEACHER_CONDITIONS = ("normal","cut_post_cue","cut_halfway","cut_last_step")
AUTONOMOUS_CONDITIONS = ("normal","zero_f4_after_cue")


def confirmation_seed(index,episode):
    """Prospective distinct split, never overlaps historical 900000 holdout."""
    if not (0 <= index < 4 and 0 <= episode < CONFIRMATION_PAIRS):
        raise ValueError("F9-M confirmation locked to 4x20 new worlds")
    n = CONFIRMATION_BASE + index * 10000 + episode
    if n >= 900_000 or n in range(1000, 41000) or n in range(100000,140100):
        raise AssertionError("Split collision")
    return n


def validate_training(report, source, index):
    if (report.get("seed_index")!=index or
            report.get("heldout_opened") is not False or
            report.get("confirmation_opened") is not False or
            report.get("training_only") is not True or
            report.get("training_s7_only") is not True or
            (report.get("worlds"),report.get("epochs"),report.get("requested_rl_steps"))
             != (TRAIN_WORLDS,TRAIN_EPOCHS,PPO_STEPS)):
        raise AssertionError("Incomplete preregistered training")
    a, b = [report["arms"][arm] for arm in ARMS]
    for field in ("parameters","input_width","initial_parameter_sha256",
                  "actual_rl_steps"):
        if a[field] != b[field]:
            raise AssertionError("Training inequality "+field)
    if a["input_width"] != DIM or a["actual_rl_steps"] < PPO_STEPS:
        raise AssertionError("Unmatched architecture/budget")
    if a["bc_mask_stats"]["effective_steps"] != 0 or (
        b["bc_mask_stats"]["effective_steps"] <= 0):
        raise AssertionError("Sham vs dropout BC intervention invalid")
    if a["ppo_mask_stats"]["effective_steps"] != 0 or (
        b["ppo_mask_stats"]["effective_steps"] <= 0):
        raise AssertionError("Sham vs dropout PPO intervention invalid")
    for arm in ARMS:
        model_path=source/f"{arm}_after_rl.zip"
        if not model_path.is_file() or file_sha(model_path)!=report["arms"][arm]["checkpoint_sha256"]:
            raise AssertionError("Checkpoint SHA does not match training report")


def autonomous_summary(model,index,size,condition):
    pairs=[]
    for j in range(CONFIRMATION_PAIRS):
        seed=confirmation_seed(index,j)
        a=autonomous_episode(model,"context",index,size,seed,False,condition)
        b=autonomous_episode(model,"context",index,size,seed,True,condition)
        if a["correct_side"]==b["correct_side"]:
            raise AssertionError("Confirmation pair did not flip cue")
        pairs.append((a,b))
    episodes=[x for pair in pairs for x in pair]
    return {
        "pairs":len(pairs),
        "success":statistics.mean(v["success"] for v in episodes),
        "both_correct":sum(a["success"] and b["success"] for a,b in pairs),
        "valid_side_flips":sum(
            a["chosen_side"]!=b["chosen_side"] and
            a["chosen_side"]!="none" and b["chosen_side"]!="none"
            for a,b in pairs),
        "cue_seen":statistics.mean(v["cue_seen"] for v in episodes),
        "fork_reached":statistics.mean(v["fork_reached"] for v in episodes),
        "intervention_applied":statistics.mean(v["intervention_applied"] for v in episodes),
        "timeout":statistics.mean(v["timeout"] for v in episodes),
        "length":statistics.mean(v["length"] for v in episodes),
    }


def confirm_teacher(model,index,size,condition):
    return [
        paired_case(model,"context",index,size,
                    confirmation_seed(index,j),condition)
        for j in range(CONFIRMATION_PAIRS)
    ]


def confirm_arm(model,index):
    natural_train=[
        paired_case(model,"context",index,7,
                    # Training-only samples to fit probe. No confirm label.
                    1000 + index*10000 + j,"normal")
        for j in range(PROBE_TRAIN_PAIRS)
    ]
    fixed=fit_linear_probe(*matrix(natural_train,"H"))
    adapted={}
    for condition in TEACHER_CONDITIONS:
        train=natural_train if condition=="normal" else [
            paired_case(model,"context",index,7,
                        1000 + index*10000+j,condition)
            for j in range(PROBE_TRAIN_PAIRS)
        ]
        adapted[condition]=fit_linear_probe(*matrix(train,"H"))

    results={"autonomous":{},"teacher_fork":{}}
    for size in (7,13):
        for condition in AUTONOMOUS_CONDITIONS:
            key=f"S{size}/{condition}"
            results["autonomous"][key]=autonomous_summary(model,index,size,condition)
            print("F9M_AUTONOMOUS",index,key,
                  json.dumps(results["autonomous"][key]),flush=True)
        for condition in TEACHER_CONDITIONS:
            key=f"S{size}/{condition}"
            pairs=confirm_teacher(model,index,size,condition)
            if not all(p["raw_equal"] and p["f4_equal"] for p in pairs):
                raise AssertionError("Unmatched teacher fork or un-restored F4")
            results["teacher_fork"][key]={
                "action":action_scores(pairs),
                "fixed_probe":scores(fixed,pairs,"H"),
                "adapted_probe":scores(adapted[condition],pairs,"H"),
            }
            print("F9M_FORK",index,key,
                  json.dumps(results["teacher_fork"][key]),flush=True)
    return results


def run(args):
    if args.index not in range(4) or args.pairs != CONFIRMATION_PAIRS:
        raise ValueError("Confirmation budget fixed before model training")
    torch.set_num_threads(1)
    source=Path(args.source)
    original=json.loads((source/"report.json").read_text(encoding="utf8"))
    validate_training(original,source,args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Confirmation results directory must be fresh")
    output={
        "experiment":"F9-M prospective confirmation of matched dropout/sham",
        "seed_index":args.index,
        "confirmation_partition":"new 300000 range (not historical holdout)",
        "confirmation_worlds_per_size":CONFIRMATION_PAIRS,
        "training_worlds_used_for_probe":PROBE_TRAIN_PAIRS,
        "probe_trained_on":"S7 train only",
        "historical_heldout_opened":False,
        "confirmation_opened":True,
        "policy_retraining_performed":False,
        "source_training_report_sha256":file_sha(source/"report.json"),
        "model_sha256":{
            arm:original["arms"][arm]["checkpoint_sha256"] for arm in ARMS
        },
        "arms":{},
    }
    for arm in ARMS:
        model=RecurrentPPO.load(str(source/f"{arm}_after_rl.zip"),device="cpu")
        try:
            output["arms"][arm]=confirm_arm(model,args.index)
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    (out/"report.json").write_text(json.dumps(output,indent=2),encoding="utf8")
    print("F9M_CONFIRM_COMPLETE",args.index,flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--pairs",type=int,default=CONFIRMATION_PAIRS)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
