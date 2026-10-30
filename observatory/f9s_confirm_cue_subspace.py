"""F9-S one-shot 895000+ cue subspace intervention confirmation.

The 12 policy weights and 24 cue/orthogonal vectors MUST be SHA/provenance
gated BEFORE any model reads the first 895000+ paired S13 world. No fitting.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9q_train_cue_probes import validate_sources,GROUPS
from observatory.f9s_train_cue_axes import (
    validate_report, confirm_seed,TRAIN_BASE,CONFIRM_BASE,CONFIRM_PAIRS
)
from observatory.f9s_cue_subspace import CONDITIONS,MODES,analyze_world,summarize


def run(args):
    if args.index not in range(4) or args.pairs!=CONFIRM_PAIRS:
        raise ValueError("F9-S one-shot confirmation budget immutable")
    torch.set_num_threads(1)
    paths,shas,source=validate_sources(args.bc_source,args.ppo_source,args.index)
    axis_file=Path(args.axes)/"report.json"
    axis_report=json.loads(axis_file.read_text(encoding="utf8"))
    validate_report(axis_report,args.index,shas,source)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("F9-S confirmation output must start empty")
    result={
        "experiment":"F9-S causal H/C cue-direction and orthogonal control patches",
        "seed_index":args.index,
        "axis_training_base":TRAIN_BASE,"confirmation_base":CONFIRM_BASE,
        "world_pairs":CONFIRM_PAIRS,
        "policy_training":False,"new_axis_training":False,
        "previous_confirmations_reused":False,
        "historical_holdout_opened":False,
        "source_frozen_sha":shas,
        "projection_file_sha256":file_sha(axis_file),
        "conditions":list(CONDITIONS),"modes":list(MODES),
        "groups":{},
    }
    for group,path in paths.items():
        model=RecurrentPPO.load(str(path),device="cpu")
        try:
            cases=[
                analyze_world(model,args.index,confirm_seed(args.index,j),
                              axis_report["groups"][group])
                for j in range(CONFIRM_PAIRS)]
            stats=summarize(cases)
            result["groups"][group]={"summary":stats,"cases":cases}
            print("F9S_CONFIRMED",args.index,group,
                  json.dumps(stats),flush=True)
        finally:
            env=model.get_env()
            if env is not None: env.close()
    (output/"report.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("F9S_CONFIRM_COMPLETE",args.index,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--pairs",type=int,default=CONFIRM_PAIRS)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--axes",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
