"""F9-P prospective independent 700000+ confirmation, no tuning/model updates.

Gate confirms all PPO checkpoints, frozen same BC parent, 112/128 mask clock.
Primary: both autonomous first exit commitments correct in cue-inverted S13
pairs with F4 erased after cue. Secondary H/C reset at first junction.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9n_stage_autonomy import autonomous_episode,pair_summary
from observatory.f9o_confirm_long_blackout import validate_training as validate_f9o
from observatory.f9p_ppo_memory_preservation import (
    ARMS,TRAIN_BASE,TRAIN_WORLDS,PPO_STEPS,ACTUAL_STEPS,
    PERIOD,NORMAL_PREFIX,SOURCE_RUN,scheduled,
)

CONFIRM_BASE=700000
PAIRS_BY_SIZE={13:40,7:20}
CONTROLS=("cue_lost_continuous","normal","fixed_4","cue_lost_hidden_reset_at_junction")
ORIGIN_BC_RUN=SOURCE_RUN


def confirmation_seed(index,size,j):
    if index not in range(4) or size not in PAIRS_BY_SIZE or not 0<=j<PAIRS_BY_SIZE[size]:
        raise ValueError("F9-P sealed 700000 confirmation only")
    seed=CONFIRM_BASE+index*10000+j
    if not 700000<=seed<740000:
        raise AssertionError("Confirmation outside new F9-P split")
    return seed


def validate_ppo(report,root,bc_report,bc_root,index):
    if (report.get("seed_index")!=index or
        report.get("source_f9o_run")!=SOURCE_RUN or
        report.get("source_f9o_report_sha256")!=file_sha(bc_root/"report.json") or
        report.get("source_bc_sha256")!=bc_report["arms"]["continuous6"]["bc_sha256"] or
        report.get("train_base")!=TRAIN_BASE or
        report.get("train_worlds")!=TRAIN_WORLDS or
        report.get("ppo_requested_steps")!=PPO_STEPS or
        report.get("mask_period")!=PERIOD or
        report.get("normal_prefix")!=NORMAL_PREFIX or
        report.get("historical_holdout_opened") is not False or
        report.get("previous_confirmations_reused") is not False or
        report.get("new_confirmation_opened") is not False or
        report.get("training_only") is not True or
        report.get("initial_BC_retrained") is not False or
        set(report.get("arms",{}))!=set(ARMS)):
        raise AssertionError("Invalid F9-P training provenance/partitions")
    a,b=(report["arms"][arm] for arm in ARMS)
    for name in ("initial_policy_sha256","parameter_count","input_width","actual_rl_steps"):
        if a[name]!=b[name]:
            raise AssertionError("Matched PPO starts and budgets not verified")
    if a["parameter_count"]!=380552 or a["input_width"]!=660 or a["actual_rl_steps"]!=ACTUAL_STEPS:
        raise AssertionError("Unexpected architecture or PPO step count")
    expected=sum(scheduled(t) for t in range(ACTUAL_STEPS+1))
    for arm in ARMS:
        row=report["arms"][arm]
        clk=row["clock_stats"]
        if clk["native_steps"]!=ACTUAL_STEPS or clk["scheduled_unique_steps"]!=expected:
            raise AssertionError("PPO mask opportunity count differs")
        if not 1<=clk["writer_max_occupancy"]<=8:
            raise AssertionError("F4 writer was inactive/broken")
        if (arm=="ppo_sham" and clk["effective_nonzero_unique_steps"]!=0) or (
            arm=="ppo_blackout" and not clk["effective_nonzero_unique_steps"]>0):
            raise AssertionError("F4 readout treatment or sham invalid")
        path=root/f"{arm}_after_rl.zip"
        if not path.is_file() or file_sha(path)!=row["checkpoint_sha256"]:
            raise AssertionError("F9-P PPO checkpoint SHA mismatch")


def evaluate(model,index,size,control):
    if control not in CONTROLS:
        raise ValueError(control)
    reset=control=="cue_lost_hidden_reset_at_junction"
    actual="cue_lost_continuous" if reset else control
    pairs=[]
    for j in range(PAIRS_BY_SIZE[size]):
        seed=confirmation_seed(index,size,j)
        a=autonomous_episode(model,index,size,seed,False,actual,reset_hidden_on_fork=reset)
        b=autonomous_episode(model,index,size,seed,True,actual,reset_hidden_on_fork=reset)
        pairs.append((a,b))
    s=pair_summary(pairs)
    if s["pairs"]!=PAIRS_BY_SIZE[size]:
        raise AssertionError("Partial confirmation data")
    if reset:
        s["hidden_reset_applied_episodes"]=sum(
            e["hidden_reset_applied"] for pair in pairs for e in pair)
    return {"summary":s,"pairs_data":[list(pair) for pair in pairs]}


def run(args):
    if args.index not in range(4) or args.s13_pairs!=40 or args.s7_pairs!=20:
        raise ValueError("F9-P preregistered confirmation budget fixed")
    torch.set_num_threads(1)
    original_root=Path(args.bc_source)
    original_report=json.loads((original_root/"report.json").read_text(encoding="utf8"))
    validate_f9o(original_report,original_root,args.index)
    root=Path(args.source)
    report=json.loads((root/"report.json").read_text(encoding="utf8"))
    validate_ppo(report,root,original_report,original_root,args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("New confirmation output must be empty")
    output={
        "experiment":"F9-P independent confirmation of PPO long F4 blackout versus sham",
        "index":args.index,
        "source_bc_run":ORIGIN_BC_RUN,
        "training_base":TRAIN_BASE,
        "confirmation_base":CONFIRM_BASE,
        "original_holdout_opened":False,
        "prior_confirmations_reused":False,
        "training_performed":False,
        "probe_fitted":False,
        "source_bc_sha256":report["source_bc_sha256"],
        "source_ppo_report_sha256":file_sha(root/"report.json"),
        "pairs_by_size":{str(k):v for k,v in PAIRS_BY_SIZE.items()},
        "groups":{},
    }
    paths={
        "bc_anchor":original_root/"continuous6_after_bc.zip",
        "ppo_sham":root/"ppo_sham_after_rl.zip",
        "ppo_blackout":root/"ppo_blackout_after_rl.zip",
    }
    for group,path in paths.items():
        model=RecurrentPPO.load(str(path),device="cpu")
        try:
            results={}
            controls=CONTROLS if group!="bc_anchor" else CONTROLS[:-1]
            for size in (13,7):
                for condition in controls:
                    key=f"S{size}/{condition}"
                    record=evaluate(model,args.index,size,condition)
                    results[key]=record
                    print("F9P_CONFIRM",args.index,group,key,
                          json.dumps(record["summary"]),flush=True)
            output["groups"][group]=results
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    (out/"report.json").write_text(json.dumps(output,indent=2),encoding="utf8")
    print("F9P_CONFIRM_COMPLETE",args.index,flush=True)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--s13-pairs",type=int,default=40)
    p.add_argument("--s7-pairs",type=int,default=20)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
