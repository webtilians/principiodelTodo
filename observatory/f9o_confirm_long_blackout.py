"""F9-O sealed prospective 600000+ confirmation with frozen NEW S13 policies.

A separate GitHub Actions gate validates all four successful training pairs
BEFORE this module can open its new world split. Primary: both autonomous
FIRST choices correct for original/inverted cue under sustained F4 loss.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9n_stage_autonomy import autonomous_episode,pair_summary
from observatory.f9o_long_blackout_training import (
    ARMS,TRAIN_BASE,TRAIN_WORLDS,BC_EPOCHS,PPO_STEPS,DIM,
)

ORIGIN_CONFIRM_BASE=600000
PAIR_COUNTS={13:40,7:20}
CONDITIONS=("cue_lost_continuous","normal","fixed_4")


def confirmation_seed(index,size,j):
    if index not in range(4) or size not in PAIR_COUNTS or not 0<=j<PAIR_COUNTS[size]:
        raise ValueError("F9-O confirmation locked")
    seed=ORIGIN_CONFIRM_BASE+index*10000+j
    if not 600000<=seed<640000:
        raise AssertionError("Confirmation leaked outside prospective split")
    return seed


def validate_training(report,root,index):
    if (report.get("index")!=index or report.get("training_seed_base")!=TRAIN_BASE or
            report.get("worlds")!=TRAIN_WORLDS or
            report.get("epochs")!=BC_EPOCHS or
            report.get("requested_rl_steps")!=PPO_STEPS or
            report.get("training_dose_per_teacher_episode")!=6 or
            report.get("ppo_blackout") is not False or
            report.get("heldout_opened") is not False or
            report.get("confirmation_opened") is not False or
            report.get("teacher_training_only") is not True):
        raise AssertionError("F9-O source training not complete/clean")
    if set(report["arms"])!=set(ARMS):
        raise AssertionError("Both training arms required")
    a,b=(report["arms"][arm] for arm in ARMS)
    for field in ("initial_policy_sha256","parameters","input_width","actual_rl_steps"):
        if a[field]!=b[field]:
            raise AssertionError("Training not matched: "+field)
    if a["parameters"]!=380552 or a["input_width"]!=DIM or a["actual_rl_steps"]!=20096:
        raise AssertionError("Policy/steps changed")
    digests=[]
    for arm in ARMS:
        d=report["arms"][arm]
        masks=d["bc_mask_stats"]
        if (masks["episodes"]!=TRAIN_WORLDS*2 or masks["incomplete"]!=0 or
                masks["scheduled"]!=TRAIN_WORLDS*2*6 or
                masks["effective"]!=TRAIN_WORLDS*2*8):
            raise AssertionError("Training exposure not exactly matched")
        digests.append(masks["teacher_sha256"])
        for phase,key in (("after_bc","bc_sha256"),("after_rl","rl_sha256")):
            path=root/f"{arm}_{phase}.zip"
            if not path.is_file() or d[key]!=file_sha(path):
                raise AssertionError("Frozen checkpoint mismatch: "+str(path))
    if digests[0]!=digests[1]:
        raise AssertionError("Teacher demos differ")


def evaluate(model,index,size,condition):
    pairs=[]
    for j in range(PAIR_COUNTS[size]):
        seed=confirmation_seed(index,size,j)
        a=autonomous_episode(model,index,size,seed,False,condition)
        b=autonomous_episode(model,index,size,seed,True,condition)
        pairs.append((a,b))
    summary=pair_summary(pairs)
    if summary["pairs"]!=PAIR_COUNTS[size]:
        raise AssertionError("Incomplete confirmation")
    return {"summary":summary,"pairs_data":[list(p) for p in pairs]}


def run(args):
    if args.index not in range(4) or args.s13_pairs!=40 or args.s7_pairs!=20:
        raise ValueError("F9-O confirmation fixed budget")
    torch.set_num_threads(1)
    root=Path(args.source)
    original=json.loads((root/"report.json").read_text(encoding="utf8"))
    validate_training(original,root,args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Fresh confirmation output only")
    result={
        "experiment":"F9-O prospective autonomous first-choice memory comparison",
        "source_training_seed_base":TRAIN_BASE,
        "confirmation_seed_base":ORIGIN_CONFIRM_BASE,
        "index":args.index,"training_performed":False,
        "original_holdout_opened":False,
        "F9M_F9N_confirmation_reused":False,
        "pairs_by_size":{str(k):v for k,v in PAIR_COUNTS.items()},
        "source_report_sha256":file_sha(root/"report.json"),
        "checkpoint_sha256":{
            arm:{"after_bc":original["arms"][arm]["bc_sha256"],
                 "after_rl":original["arms"][arm]["rl_sha256"]}
            for arm in ARMS
        },
        "arms":{},
    }
    for arm in ARMS:
        result["arms"][arm]={}
        for phase in ("after_bc","after_rl"):
            model=RecurrentPPO.load(str(root/f"{arm}_{phase}.zip"),device="cpu")
            try:
                conditions=(("S13/cue_lost_continuous",) if phase=="after_bc"
                            else tuple(f"S{size}/{cond}" for size in (13,7)
                                       for cond in CONDITIONS))
                output={}
                for key in conditions:
                    left,right=key.split("/")
                    size=int(left[1:])
                    record=evaluate(model,args.index,size,right)
                    output[key]=record
                    print("F9O_EVAL",args.index,arm,phase,key,
                          json.dumps(record["summary"]),flush=True)
                result["arms"][arm][phase]=output
            finally:
                env=model.get_env()
                if env is not None:
                    env.close()
    (out/"report.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("F9O_CONFIRM_COMPLETE",args.index,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--s13-pairs",type=int,default=40)
    p.add_argument("--s7-pairs",type=int,default=20)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
