"""F10-A independent new-world DEV: first-fork causal READ and autonomous run."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import ensure_dependencies,file_sha
from observatory.f9o_confirm_long_blackout import validate_training
from observatory.f9b_behavior_audit import build_env,get_world_labels,swap_cue
from observatory.f9q_causal_fork import matched_pair,history_pair,REGIMES
from observatory.f9v_lexicographic_causal_f4 import one_shot_branch
from observatory.f9i_selective_memory import SelectiveF4,DIM
from observatory.f9j_recurrent_memory_attribution import frozen_action
from observatory.f9n_stage_autonomy import StageObserver
from observatory.f10a_actor_evidence_training import (
    ARMS,modify_tail,N_TRAIN,TRAIN_WORLDS,train_seed
)

DEV_BASE=12200000
DEV_WORLDS=32
ACTORS=("f9o_original_bc",)+ARMS
MODES=("skip","read_clean","read_adversarial")
CELLS=(("key",REGIMES[0]),("ball",REGIMES[0]),
       ("key",REGIMES[1]),("ball",REGIMES[1]))

def dev_seed(index,j):
    if index not in range(4) or j not in range(DEV_WORLDS):
        raise ValueError("F10-A DEV only, CONFIRM 12300000+ sealed")
    return DEV_BASE+10000*index+j

def evaluate_first_fork(actor,index,seed):
    pair=matched_pair(actor,index,seed)
    records=[]
    for regime in REGIMES:
        snapshots=history_pair(actor,index,pair,regime)
        assert np.array_equal(snapshots[0]["raw"],snapshots[1]["raw"])
        for receiver,snapshot in enumerate(snapshots):
            skip=one_shot_branch(actor,index,pair,receiver,snapshot,0)
            true=one_shot_branch(actor,index,pair,receiver,snapshot,1)
            fake=dict(snapshot,natural_f4=modify_tail(snapshot["natural_f4"],"adversarial"))
            bad=one_shot_branch(actor,index,pair,receiver,fake,1)
            assert skip["read"]==0 and true["read"]==bad["read"]==1
            assert np.array_equal(fake["raw"],snapshot["raw"])
            assert np.array_equal(fake["state"][0],snapshot["state"][0])
            assert np.array_equal(fake["state"][1],snapshot["state"][1])
            d=true["first_arm_correct"]-skip["first_arm_correct"]
            db=bad["first_arm_correct"]-skip["first_arm_correct"]
            records.append({
                "seed":seed,"cue":pair["labels"][receiver]["cue"],
                "regime":regime,
                "fork_raw_equal_between_cues":True,
                "fake_changes_identity_only":bool(
                    not np.array_equal(fake["natural_f4"],snapshot["natural_f4"])),
                "D_clean":int(d),"D_adversarial":int(db),
                "outcomes":{"skip":skip,"read_clean":true,"read_adversarial":bad}})
    assert len(records)==4
    assert {(v["cue"],v["regime"]) for v in records}==set(CELLS)
    return records

def autonomous(actor,index,seed,reverse,quality):
    """No teacher actions, real MiniGrid native reward; public observations only."""
    if quality not in ("clean","adversarial"):raise ValueError(quality)
    env=build_env(13)
    writer=SelectiveF4("context",8,seed=81000+index)
    state=None
    first_seen=False
    steps=0
    reward=0.
    ever_corrupted=0
    try:
        raw,_=env.reset(seed=seed)
        if reverse:
            raw,_=swap_cue(env,get_world_labels(env))
        labels=get_world_labels(env)
        observer=StageObserver(labels,13)
        observer.observe(env.unwrapped.agent_pos,0)
        while True:
            encoded=writer.encode(raw)
            # Only the first publicly observed object establishes exposure.
            # No oracle trajectory, time-to-fork, goal or cue label is used.
            if writer.new_events:first_seen=True
            tail=encoded[148:].copy()
            if quality=="adversarial" and first_seen and not writer.new_events:
                tail2=modify_tail(tail,"adversarial")
                ever_corrupted+=int(not np.array_equal(tail2,tail))
                tail=tail2
            x=np.concatenate((raw,tail)).astype(np.float32)
            assert x.shape==(DIM,) and np.array_equal(x[:148],raw)
            action,state=frozen_action(actor,x,state,first=steps==0)
            raw,r,terminated,truncated,_=env.step(action)
            reward+=float(r)
            steps+=1
            observer.observe(env.unwrapped.agent_pos,steps)
            if terminated or truncated:break
            if steps>1000:raise AssertionError("S13 autonomous rollout runaway")
        summary=observer.summary()
        return {"seed":seed,"quality":quality,"cue":labels["cue"],
                "seen_object":first_seen,"corrupted_frames":ever_corrupted,
                "first_correct":int(summary["first_choice_correct"]),
                "success":int(reward>0),"steps":steps}
    finally:
        env.close()

def counts(records):
    def tally(name):
        vals=[v[name] for v in records]
        return {str(d):sum(x==d for x in vals) for d in (-1,0,1)}
    return {
        "states":len(records),
        "D_clean":tally("D_clean"),
        "D_adversarial":tally("D_adversarial"),
        "first_correct":{mode:sum(v["outcomes"][mode]["first_arm_correct"]
                               for v in records) for mode in MODES},
        "identity_corruption_effective":sum(v["fake_changes_identity_only"]
                                            for v in records),
    }

def execute(args):
    if args.index not in range(4):raise ValueError("Four fixed replicas only")
    torch.set_num_threads(1)
    deps=ensure_dependencies()
    old=Path(args.source);new=Path(args.trained)
    manifest=json.loads((old/"report.json").read_text())
    validate_training(manifest,old,args.index)
    trained=json.loads((new/"report.json").read_text())
    assert trained["index"]==args.index and trained["phase"]=="TRAIN_ONLY"
    assert trained["train_worlds"]==N_TRAIN and trained["train_examples_per_arm"]==1536
    assert trained["source_bc_sha256"]==manifest["arms"]["continuous6"]["bc_sha256"]
    assert not trained["reader_head_trained"] and not trained["future_confirm_opened"]
    models={"f9o_original_bc":old/"continuous6_after_bc.zip"}
    for arm in ARMS:
        models[arm]=new/trained["arms"][arm]["checkpoint"]
        assert file_sha(models[arm])==trained["arms"][arm]["checkpoint_sha256"]
    assert file_sha(models["f9o_original_bc"])==trained["source_bc_sha256"]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    result={
        "experiment":"F10-A new-actor causal DEV before any READ head",
        "phase":"DEV_ONLY","index":args.index,
        "trained_report_sha":file_sha(new/"report.json"),
        "source_f9o_report_sha":file_sha(old/"report.json"),
        "new_train_worlds":N_TRAIN,"dev_worlds":DEV_WORLDS,
        "new_dev_seeds":[dev_seed(args.index,j) for j in range(DEV_WORLDS)],
        "actor_retrained":True,"old_checkpoints_mutated":False,
        "reader_trained":False,"future_confirm_12300000_opened":False,
        "previous_confirmations_reused":False,"dependency_versions":deps,
        "actors":{},
    }
    for label,path in models.items():
        actor=RecurrentPPO.load(str(path),device="cpu")
        actor.policy.set_training_mode(False)
        for p in actor.policy.parameters():p.requires_grad_(False)
        states=[]
        autonomy=[]
        for j in range(DEV_WORLDS):
            seed=dev_seed(args.index,j)
            states+=evaluate_first_fork(actor,args.index,seed)
            for quality in ("clean","adversarial"):
                for reverse in (False,True):
                    autonomy.append(autonomous(actor,args.index,seed,reverse,quality))
            print("F10A_DEV_NEW_WORLD",args.index,label,j,seed,flush=True)
        assert len(states)==128 and len(autonomy)==128
        strata={}
        for cue,regime in CELLS:
            subset=[v for v in states if v["cue"]==cue and v["regime"]==regime]
            assert len(subset)==DEV_WORLDS
            strata[f"{cue}|{regime}"]=counts(subset)
        result["actors"][label]={
            "checkpoint_sha256":file_sha(path),
            "all_states":counts(states),
            "by_cell":strata,
            "states":states,
            "autonomous":autonomy,
            "autonomous_summary":{
                quality:{
                  "episodes":sum(v["quality"]==quality for v in autonomy),
                  "success":sum(v["success"] for v in autonomy if v["quality"]==quality),
                  "first_correct":sum(v["first_correct"] for v in autonomy if v["quality"]==quality),
                  "corrupted_frames":sum(v["corrupted_frames"] for v in autonomy if v["quality"]==quality),
                } for quality in ("clean","adversarial")},
        }
        print("F10A_ACTOR_DEV",args.index,label,
              json.dumps({"all":result["actors"][label]["all_states"],
                          "autonomous":result["actors"][label]["autonomous_summary"]}),flush=True)
        env=actor.get_env()
        if env is not None:env.close()
    (output/"report.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("F10A_DEV_COMPLETE",args.index,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--trained",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    execute(p.parse_args())
