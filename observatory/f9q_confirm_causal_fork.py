"""F9-Q one-shot 800000+ confirmation: decode H/C, transplant and first exit.

The four probes are frozen on 750000+ BEFORE this script runs. Three original
policy checkpoints remain frozen, fully verified by SHA-256 provenance.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9q_causal_fork import REGIMES, INTERVENTIONS, analyze_pair, summarize
from observatory.f9q_train_cue_probes import (
    GROUPS,ALPHA,PROBE_TRAIN_BASE,TRAIN_PAIRS,validate_sources,
)

CONFIRM_BASE=800000
CONFIRM_PAIRS=40


def confirmation_seed(index,j):
    if index not in range(4) or not 0<=j<CONFIRM_PAIRS:
        raise ValueError("F9-Q one-shot prospective split only")
    seed=CONFIRM_BASE+index*10000+j
    if not 800000<=seed<840000:
        raise AssertionError("Unexpected confirm seed")
    return seed


def validate_probes(report,path,shas,source,index):
    if (report.get("index")!=index or
        report.get("probe_train_base")!=PROBE_TRAIN_BASE or
        report.get("probe_train_pairs")!=TRAIN_PAIRS or
        report.get("probe_alpha")!=ALPHA or
        report.get("training_only") is not True or
        report.get("policy_training") is not False or
        report.get("confirmation_opened") is not False or
        report.get("holdout_opened") is not False or
        report.get("model_shas")!=shas or
        report.get("source_provenance")!=source or
        set(report.get("groups",{}))!=set(GROUPS)):
        raise AssertionError("F9-Q probe provenance/partition failed")
    for group in GROUPS:
        if set(report["groups"][group])!=set(REGIMES):
            raise AssertionError("Missing preregistered probe regime")
        for regime in REGIMES:
            row=report["groups"][group][regime]
            if set(row)!={"H","C","mean_effective_mask"}:
                raise AssertionError("Unexpected fitted feature or extra probe")
            for feature in ("H","C"):
                p=row[feature]
                if p["feature"]!=feature or p["pairs"]!=TRAIN_PAIRS or p["alpha"]!=ALPHA:
                    raise AssertionError("Probe fitted out-of-split")
                if not p["training_worlds_only"]:
                    raise AssertionError("Probe training uses non-training worlds")
                for name in ("center","scale","weight"):
                    a=np.asarray(p[name],np.float32)
                    if a.shape!=(64,) or not np.isfinite(a).all():
                        raise AssertionError("Unexpected probe parameters")
                if np.any(np.asarray(p["scale"])<=0):
                    raise AssertionError("Invalid scale")
    if not Path(path).is_file():
        raise AssertionError("No probe report")
    return True


def run(args):
    if args.index not in range(4) or args.pairs!=CONFIRM_PAIRS:
        raise ValueError("F9-Q confirm scope immutable")
    torch.set_num_threads(1)
    paths,shas,source=validate_sources(args.bc_source,args.ppo_source,args.index)
    probe_file=Path(args.probes)/"report.json"
    report=json.loads(probe_file.read_text(encoding="utf8"))
    validate_probes(report,probe_file,shas,source,args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Fresh confirmation directory required")
    result={
        "experiment":"F9-Q causal recall at matched S13 decision fork",
        "seed_index":args.index,
        "confirmation_seed_base":CONFIRM_BASE,
        "probe_train_seed_base":PROBE_TRAIN_BASE,
        "world_pairs":CONFIRM_PAIRS,
        "training_performed":False,
        "policy_training":False,
        "new_probe_fit":False,
        "prior_confirmations_reused":False,
        "historical_holdout_opened":False,
        "F4_zero_at_fork_and_continuation":True,
        "checkpoint_sha256":shas,
        "probe_report_sha256":file_sha(probe_file),
        "groups":{},
    }
    for group,path in paths.items():
        model=RecurrentPPO.load(str(path),device="cpu")
        try:
            group_out={}
            for regime in REGIMES:
                probes=report["groups"][group][regime]
                cases=[
                    analyze_pair(model,args.index,confirmation_seed(args.index,j),
                                 regime,probes=probes,include_all=True)
                    for j in range(CONFIRM_PAIRS)]
                stats=summarize(cases)
                group_out[regime]={"summary":stats,"cases":cases}
                print("F9Q_RESULT",args.index,group,regime,json.dumps(stats),flush=True)
            result["groups"][group]=group_out
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    (out/"report.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("F9Q_CONFIRM_COMPLETE",args.index,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--pairs",type=int,default=CONFIRM_PAIRS)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--probes",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
