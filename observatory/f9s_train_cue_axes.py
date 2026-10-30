"""F9-S TRAIN ONLY: cue directions of frozen H/C at last pre-fork step.

Four seeds × three frozen policy families × 80 paired new S13 worlds, with
disjoint 890000+ IDs. No confirm worlds, no actor/critic/LSTM fitting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9q_train_cue_probes import validate_sources,GROUPS
from observatory.f9q_causal_fork import matched_pair
from observatory.f9r_recurrent_state_drift import trace_history
from observatory.f9s_cue_subspace import normalize,orthogonal_control,validate_axes

TRAIN_BASE=890000
TRAIN_PAIRS=80
CONFIRM_BASE=895000
CONFIRM_PAIRS=40
ALGORITHM="mean_ball_minus_key_at_last"
CONTROL_SEED=617023


def train_seed(index,j):
    if index not in range(4) or not 0<=j<TRAIN_PAIRS:
        raise ValueError("F9-S direction fitting only on NEW train seeds")
    return TRAIN_BASE+1000*index+j


def confirm_seed(index,j):
    if index not in range(4) or not 0<=j<CONFIRM_PAIRS:
        raise ValueError("F9-S confirmation sealed")
    seed=CONFIRM_BASE+1000*index+j
    if seed>=900000:
        raise AssertionError("Would enter historical holdout")
    return seed


def fit_axes(model,index,group):
    diffs={"H":[],"C":[]}
    for j in range(TRAIN_PAIRS):
        world=matched_pair(model,index,train_seed(index,j))
        states=[
            trace_history(model,index,frames,world["times"],"normal_reference")
            for frames in world["frames"]]
        last=int(world["times"]["fork"])-1
        cue=[x["cue"] for x in world["labels"]]
        if set(cue)!={"key","ball"}:
            raise AssertionError("Need both opposite real cue labels")
        a=cue.index("key")
        b=cue.index("ball")
        for name,k in (("H",0),("C",1)):
            d=(states[b]["natural_snapshots"][last][k]-
               states[a]["natural_snapshots"][last][k]).reshape(64)
            diffs[name].append(d.astype(np.float64))
    axes={}
    for name,values in diffs.items():
        w=normalize(np.mean(np.stack(values),axis=0))
        orth=orthogonal_control(w,CONTROL_SEED+index*101+
                                GROUPS.index(group)*17+(0 if name=="H" else 1))
        axes[name]={
            "cue":w.tolist(),
            "orthogonal":orth.tolist(),
            "paired_train_worlds":TRAIN_PAIRS,
            "fit_method":ALGORITHM,
            "cue_direction_label":"ball_minus_key",
            "train_pair_contrast_mean_l2":float(np.linalg.norm(np.mean(values,axis=0))),
        }
    validate_axes(axes)
    return axes


def validate_report(report,index,shas,source):
    if (report.get("seed_index")!=index or report.get("train_base")!=TRAIN_BASE
        or report.get("confirm_base")!=CONFIRM_BASE or report.get("train_pairs")!=TRAIN_PAIRS
        or report.get("frozen_policy_sha")!=shas or report.get("source")!=source
        or report.get("model_training") is not False
        or report.get("confirmation_opened") is not False
        or report.get("historical_holdout_opened") is not False
        or set(report.get("groups",{}))!=set(GROUPS)):
        raise AssertionError("Axis provenance mismatch")
    for name in GROUPS:
        validate_axes(report["groups"][name])
    return True


def run(args):
    if args.index not in range(4) or args.train_pairs!=TRAIN_PAIRS:
        raise ValueError("F9-S direction fitting budget fixed")
    torch.set_num_threads(1)
    paths,shas,source=validate_sources(args.bc_source,args.ppo_source,args.index)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Fresh direction fitting output")
    report={
        "experiment":"F9-S cue-axis H/C trained on new disjoint paired S13",
        "seed_index":args.index,
        "train_base":TRAIN_BASE,"confirm_base":CONFIRM_BASE,
        "train_pairs":TRAIN_PAIRS,"confirm_pairs":CONFIRM_PAIRS,
        "frozen_policy_sha":shas,"source":source,
        "model_training":False,"confirmation_opened":False,
        "historical_holdout_opened":False,
        "groups":{}
    }
    for group,path in paths.items():
        model=RecurrentPPO.load(str(path),device="cpu")
        try:
            axes=fit_axes(model,args.index,group)
            report["groups"][group]=axes
            print("F9S_TRAIN_AXES",args.index,group,
                  json.dumps({name:{"norm":float(np.linalg.norm(v["cue"])),
                   "orth_dot":float(np.dot(v["cue"],v["orthogonal"])),
                   "contrast_l2":v["train_pair_contrast_mean_l2"]}
                   for name,v in axes.items()}),flush=True)
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("F9S_AXES_LOCKED",args.index,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--train-pairs",type=int,default=TRAIN_PAIRS)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
