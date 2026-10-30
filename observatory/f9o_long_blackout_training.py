"""F9-O: equal-count F4 input blackouts, different temporal distribution.

Train eight NEW context8 LSTM policies on NEW S13-only 500000+ train worlds.
Only supervised imitation sees blackout: exactly six effective frames per
episode for both treatment and control. PPO finetuning runs CLEAN in both
arms, with the same hyperparameters/step count but not necessarily matching
on-policy transitions. No F9-M/N confirmation or historical holdout access.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import ensure_dependencies, file_sha
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9d_autonomous_navigation import demonstration
from observatory.f9e_f4_autonomous import clone
from observatory.f9i_selective_memory import SelectiveF4, SelectiveObservation, DIM
from observatory.f9m_temporal_dropout_training import parameter_signature

ARMS=("fragmented6","continuous6")
OFFSETS={
    "fragmented6":frozenset((0,1,3,4,6,7)),
    "continuous6":frozenset(range(6)),
}
TRAIN_BASE=500_000
TRAIN_WORLDS=600
BC_EPOCHS=40
PPO_STEPS=20_000
CAPACITY=8


def train_seed(index,j):
    if index not in range(4) or not 0<=j<TRAIN_WORLDS:
        raise ValueError("Only F9-O new S13 TRAIN worlds")
    seed=TRAIN_BASE+index*10_000+j
    if not 500000<=seed<540000:
        raise AssertionError("Seed escaped new training range")
    return seed


class NewS13TrainingWorlds(gym.Wrapper):
    """Finite S13 TRAIN only; paired cue variants; native rewards/actions."""
    def __init__(self,index):
        if index not in range(4):
            raise ValueError(index)
        super().__init__(build_env(13))
        self.index=index
        self.counter=0

    def reset(self,*,seed=None,options=None):
        j=(self.counter//2)%TRAIN_WORLDS
        reverse=bool(self.counter%2)
        self.counter+=1
        raw,_=self.env.reset(seed=train_seed(self.index,j))
        if reverse:
            raw,_=swap_cue(self.env,get_world_labels(self.env))
        return raw,{}


def mask_timeline(raw_frames,arm,index):
    """The same 8 F4-mask opportunities; causal public first sighting only."""
    if arm not in ARMS or index not in range(4):
        raise ValueError("Unknown F9-O arm")
    writer=SelectiveF4("context",CAPACITY,seed=81000+index)
    encoded_rows=[]
    first_public=None
    scheduled=effective=0
    changed_indices=[]
    for t,raw in enumerate(raw_frames):
        encoded=writer.encode(raw)
        if first_public is None and writer.new_events:
            first_public=t
        offset=None if first_public is None else t-(first_public+2)
        block=(offset in OFFSETS[arm]) if offset is not None else False
        actor=encoded.copy()
        if block:
            scheduled+=1
            actor[148:]=0.
            if not np.array_equal(actor,encoded):
                effective+=1
                changed_indices.append(t)
        if encoded.shape!=(DIM,) or not np.array_equal(actor[:148],raw):
            raise AssertionError("F4 ablation changed public observation")
        encoded_rows.append(actor)
    return np.stack(encoded_rows), {
        "scheduled":scheduled,"effective":effective,
        "first_public_frame":first_public,"changed_indices":changed_indices
    }


def prepare_dataset(demonstrations,arm,index):
    rows=[]
    details={"episodes":len(demonstrations),"scheduled":0,"effective":0,
             "incomplete":0,"teacher_sha256":""}
    digest=hashlib.sha256()
    for frames,actions,seen in demonstrations:
        transformed,audit=mask_timeline(frames,arm,index)
        if transformed.shape!=(len(actions),DIM):
            raise AssertionError("Mismatched teacher features")
        digest.update(np.asarray(actions).tobytes())
        digest.update(np.asarray(frames).tobytes())
        if audit["scheduled"] != 6 or audit["effective"] != 6:
            details["incomplete"]+=1
        details["scheduled"]+=audit["scheduled"]
        details["effective"]+=audit["effective"]
        rows.append((transformed,actions,seen))
    details["teacher_sha256"]=digest.hexdigest()
    return rows,details


def make_model(arm,index):
    if arm not in ARMS or index not in range(4):
        raise ValueError("Unapproved training arm/index")
    env=SelectiveObservation(NewS13TrainingWorlds(index),
                             mode="context",capacity=CAPACITY,seed=81000+index)
    return RecurrentPPO(
        "MlpLstmPolicy",env=env,
        seed=train_seed(index,0),device="cpu",
        n_steps=128,batch_size=128,n_epochs=4,learning_rate=3e-4,
        gamma=.99,gae_lambda=.95,ent_coef=.01,verbose=0,
        policy_kwargs={"lstm_hidden_size":64,
                       "net_arch":dict(pi=[64],vf=[64])})


def run(args):
    if (args.index not in range(4) or args.worlds!=TRAIN_WORLDS
            or args.epochs!=BC_EPOCHS or args.timesteps!=PPO_STEPS):
        raise ValueError("F9-O training budget locked")
    torch.set_num_threads(1)
    dependencies=ensure_dependencies()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Fresh output directory required")
    teacher=[demonstration(train_seed(args.index,j),size=13,reverse=reverse)
             for j in range(TRAIN_WORLDS) for reverse in (False,True)]
    datasets={}
    stats={}
    for arm in ARMS:
        dataset,audit=prepare_dataset(teacher,arm,args.index)
        if audit["incomplete"] or audit["scheduled"]!=len(teacher)*6 or audit["effective"]!=len(teacher)*6:
            raise AssertionError("Equal actual F4 blackout exposure not achieved: "+str((arm,audit)))
        datasets[arm]=dataset
        stats[arm]=audit
    if stats["fragmented6"]["teacher_sha256"]!=stats["continuous6"]["teacher_sha256"]:
        raise AssertionError("Teacher actions/frames not paired")
    for row_a,row_b in zip(datasets["fragmented6"],datasets["continuous6"]):
        if not np.array_equal(row_a[1],row_b[1]) or not np.array_equal(row_a[0][:,:148],row_b[0][:,:148]):
            raise AssertionError("Control/treated labels or public frames changed")
    report={
        "experiment":"F9-O matched-long vs fragmented F4 blackout training",
        "training_seed_base":TRAIN_BASE,"index":args.index,
        "heldout_opened":False,"confirmation_opened":False,
        "teacher_training_only":True,"ppo_blackout":False,
        "worlds":TRAIN_WORLDS,"epochs":BC_EPOCHS,
        "requested_rl_steps":PPO_STEPS,
        "training_dose_per_teacher_episode":6,
        "dependencies":dependencies,
        "arms":{},
    }
    for arm in ARMS:
        model=make_model(arm,args.index)
        try:
            details={
                "initial_policy_sha256":parameter_signature(model.policy),
                "parameters":sum(v.numel() for v in model.policy.parameters()),
                "input_width":int(model.observation_space.shape[0]),
                "bc_mask_stats":stats[arm],
            }
            details["bc_losses"]=clone(model,datasets[arm],BC_EPOCHS,args.index)
            bc_path=out/f"{arm}_after_bc.zip"
            model.save(str(bc_path))
            details["bc_sha256"]=file_sha(bc_path)
            # PPO gets normal, NONABLATE input in both arms, same budget.
            model.learn(total_timesteps=PPO_STEPS)
            details["actual_rl_steps"]=int(model.num_timesteps)
            rl_path=out/f"{arm}_after_rl.zip"
            model.save(str(rl_path))
            details["rl_sha256"]=file_sha(rl_path)
            report["arms"][arm]=details
            (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
            print("F9O_TRAINED",args.index,arm,json.dumps({
                k:v for k,v in details.items() if k!="bc_losses"}),flush=True)
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    a,b=(report["arms"][arm] for arm in ARMS)
    for field in ("parameters","input_width","initial_policy_sha256","actual_rl_steps"):
        if a[field]!=b[field]:
            raise AssertionError("Unequal arms "+field)
    if a["parameters"]!=380552 or a["input_width"]!=DIM or a["actual_rl_steps"]!=20096:
        raise AssertionError("Architecture or PPO step budget differs")
    print("F9O_TRAINING_COMPLETE",args.index,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--worlds",type=int,default=TRAIN_WORLDS)
    p.add_argument("--epochs",type=int,default=BC_EPOCHS)
    p.add_argument("--timesteps",type=int,default=PPO_STEPS)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
