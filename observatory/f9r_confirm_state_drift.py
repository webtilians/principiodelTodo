"""F9-R one-shot 850000+ causal confirmation after independent preflight.

All checkpoints are frozen. The same 160 S13 worlds are examined under
ten preregistered recurrency interventions with deterministic teacher prefix.
No fitting, policy updates, model selection, or access to historical holdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9q_train_cue_probes import GROUPS, validate_sources
from observatory.f9r_recurrent_state_drift import CONDITIONS, SWAPS, analyze_world, summarize_worlds

CONFIRM_BASE=850000
WORLD_PAIRS=40
TRAIN_FIXTURE_BASE=840000


def confirm_seed(index,j):
    if index not in range(4) or not 0<=j<WORLD_PAIRS:
        raise ValueError("F9-R sealed new confirmation only")
    seed=CONFIRM_BASE+10000*index+j
    if not 850000<=seed<890000:
        raise AssertionError("F9-R outside reserved confirmation")
    return seed


def fixture_seed(index,j):
    if index not in range(4) or not 0<=j<10:
        raise ValueError("Only 840000 fixtures for preflight")
    return TRAIN_FIXTURE_BASE+10000*index+j


def run(args):
    if args.index not in range(4) or args.pairs!=WORLD_PAIRS:
        raise ValueError("Preregistered F9-R budget is fixed")
    torch.set_num_threads(1)
    paths,shas,source=validate_sources(args.bc_source,args.ppo_source,args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("New confirmation output must start empty")
    result={
        "experiment":"F9-R recurrent H/C update drift and targeted restore",
        "index":args.index,
        "new_confirm_base":CONFIRM_BASE,
        "preflight_fixture_base":TRAIN_FIXTURE_BASE,
        "pairs":WORLD_PAIRS,
        "policy_training":False,"probe_training":False,
        "confirm_used_as_training":False,"historical_holdout_opened":False,
        "F9Q_confirmation_reused":False,
        "source_run_ids":source,
        "frozen_model_shas":shas,
        "interventions":list(CONDITIONS),
        "swap_conditions":list(SWAPS),
        "groups":{},
    }
    for name,path in paths.items():
        model=RecurrentPPO.load(str(path),device="cpu")
        try:
            cases=[]
            for j in range(WORLD_PAIRS):
                seed=confirm_seed(args.index,j)
                cases.append(analyze_world(model,args.index,seed))
            summary=summarize_worlds(cases)
            result["groups"][name]={"summary":summary,"cases":cases}
            print("F9R_CONFIRMED",args.index,name,
                  json.dumps(summary),flush=True)
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    (out/"report.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("F9R_SEED_CONFIRMATION_COMPLETE",args.index,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--pairs",type=int,default=WORLD_PAIRS)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
