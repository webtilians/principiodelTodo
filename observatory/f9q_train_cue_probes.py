"""F9-Q: train only tiny linear H/C identity decoders on disjoint 750000 worlds.

Actor checkpoints completely frozen. No decoder or policy fitted with 800000+
confirmation. Source hashes validated for all BC/PPO models before fitting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9k_temporal_memory_probes import fit_linear_probe, predict_probe
from observatory.f9o_confirm_long_blackout import validate_training as validate_f9o
from observatory.f9p_confirm_memory_preservation import validate_ppo
from observatory.f9q_causal_fork import REGIMES, matched_pair, history_pair

PROBE_TRAIN_BASE=750000
TRAIN_PAIRS=80
ALPHA=10.0
BC_RUN=40957255987
PPO_RUN=40960267485
GROUPS=("bc_anchor","ppo_sham","ppo_blackout")


def train_seed(index,j):
    if index not in range(4) or not 0<=j<TRAIN_PAIRS:
        raise ValueError("F9-Q probe train split only 750000+, 80 pairs/seed")
    return PROBE_TRAIN_BASE+index*10000+j


def validate_sources(bc_root,ppo_root,index):
    bc_root=Path(bc_root)
    ppo_root=Path(ppo_root)
    bc_report=json.loads((bc_root/"report.json").read_text(encoding="utf8"))
    validate_f9o(bc_report,bc_root,index)
    ppo_report=json.loads((ppo_root/"report.json").read_text(encoding="utf8"))
    validate_ppo(ppo_report,ppo_root,bc_report,bc_root,index)
    paths={
        "bc_anchor":bc_root/"continuous6_after_bc.zip",
        "ppo_sham":ppo_root/"ppo_sham_after_rl.zip",
        "ppo_blackout":ppo_root/"ppo_blackout_after_rl.zip",
    }
    shas={
        "bc_anchor":bc_report["arms"]["continuous6"]["bc_sha256"],
        "ppo_sham":ppo_report["arms"]["ppo_sham"]["checkpoint_sha256"],
        "ppo_blackout":ppo_report["arms"]["ppo_blackout"]["checkpoint_sha256"],
    }
    for group,path in paths.items():
        if not path.is_file() or file_sha(path)!=shas[group]:
            raise AssertionError("Frozen source SHA mismatch: "+str(path))
    return paths,shas,{
        "f9o_report_sha256":file_sha(bc_root/"report.json"),
        "f9p_report_sha256":file_sha(ppo_root/"report.json"),
        "f9o_run":BC_RUN,"f9p_run":PPO_RUN,
    }


def fit_and_serialize(states,feature):
    if feature not in ("H","C"):
        raise ValueError(feature)
    component=0 if feature=="H" else 1
    samples=[]
    labels=[]
    for items,cue in states:
        for j,entry in enumerate(items):
            samples.append(entry["state"][component].reshape(-1))
            labels.append(1.0 if cue[j]=="ball" else -1.0)
    x=np.asarray(samples,np.float64)
    y=np.asarray(labels,np.float64)
    if len(samples)!=2*TRAIN_PAIRS or np.sum(y==1)!=TRAIN_PAIRS or np.sum(y==-1)!=TRAIN_PAIRS:
        raise AssertionError("Probe training must be balanced true key/ball")
    if x.shape!=(2*TRAIN_PAIRS,64):
        raise AssertionError("Unexpected recurrent code width")
    probe=fit_linear_probe(x,y,alpha=ALPHA)
    pred=predict_probe(probe,x)
    return {
        "center":probe[0].tolist(),
        "scale":probe[1].tolist(),
        "weight":probe[2].tolist(),
        "training_accuracy_diagnostic":float(np.mean(pred==y)),
        "alpha":ALPHA,
        "pairs":TRAIN_PAIRS,
        "feature":feature,
        "training_worlds_only":True,
    }


def run(args):
    if args.index not in range(4) or args.train_pairs!=TRAIN_PAIRS:
        raise ValueError("F9-Q preregistered probe budget fixed")
    torch.set_num_threads(1)
    paths,shas,provenance=validate_sources(args.bc_source,args.ppo_source,args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Probe training output must be empty")
    report={
        "experiment":"F9-Q separate linear decoders of cue identity in H and C",
        "index":args.index,
        "training_only":True,
        "policy_training":False,
        "confirmation_opened":False,
        "holdout_opened":False,
        "probe_train_base":PROBE_TRAIN_BASE,
        "probe_train_pairs":TRAIN_PAIRS,
        "probe_alpha":ALPHA,
        "model_shas":shas,
        "source_provenance":provenance,
        "groups":{},
    }
    for group,path in paths.items():
        model=RecurrentPPO.load(str(path),device="cpu")
        try:
            groups={}
            for regime in REGIMES:
                states=[]
                effective=[]
                for j in range(TRAIN_PAIRS):
                    pair=matched_pair(model,args.index,train_seed(args.index,j))
                    snapshots=history_pair(model,args.index,pair,regime)
                    states.append((snapshots,[s["cue"] for s in pair["labels"]]))
                    effective.append([s["effective_prefix_masks"] for s in snapshots])
                groups[regime]={
                    "H":fit_and_serialize(states,"H"),
                    "C":fit_and_serialize(states,"C"),
                    "mean_effective_mask":float(np.mean(effective)),
                }
                print("F9Q_PROBE_TRAIN",args.index,group,regime,
                      json.dumps({"H_train":groups[regime]["H"]["training_accuracy_diagnostic"],
                                  "C_train":groups[regime]["C"]["training_accuracy_diagnostic"],
                                  "mask":groups[regime]["mean_effective_mask"]}),flush=True)
            report["groups"][group]=groups
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("F9Q_PROBES_LOCKED",args.index,flush=True)
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
