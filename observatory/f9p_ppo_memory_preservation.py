"""F9-P: PPO long F4 readout interruptions versus matched sham, from SAME F9-O BC.

Strictly new S13 PPO training worlds: 550000+. No confirmation/holdout access.
The global agent-step mask clock is public/trajectory independent: 112/128
observation slots are scheduled as a contiguous block. Full F4 storage
operates in both arms; only the policy's 512 readout features are ablated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import ensure_dependencies, file_sha
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9i_selective_memory import SelectiveF4, DIM, FIELDS
from observatory.f9m_temporal_dropout_training import parameter_signature
from observatory.f9o_confirm_long_blackout import validate_training as validate_f9o

ARMS=("ppo_sham","ppo_blackout")
TRAIN_BASE=550_000
TRAIN_WORLDS=600
PPO_STEPS=20_000
ACTUAL_STEPS=20_096
CAPACITY=8
INFORMATIVE_END=148+CAPACITY*FIELDS
PERIOD=128
NORMAL_PREFIX=16
SOURCE_RUN=40957255987


def train_seed(index,j):
    if index not in range(4) or not 0<=j<TRAIN_WORLDS:
        raise ValueError("F9-P train seeds confined to four disjoint S13 sets")
    return TRAIN_BASE+index*10000+j


def scheduled(global_step):
    """No observation, label, position, behavior or memory controls this clock."""
    if not isinstance(global_step,int) or global_step<0:
        raise ValueError("global_step must be nonnegative int")
    return global_step%PERIOD>=NORMAL_PREFIX


class S13PPOTrainingWorlds(gym.Wrapper):
    """Native S13 train, paired inverted start cues, never confirmation."""
    def __init__(self,index):
        if index not in range(4):
            raise ValueError(index)
        super().__init__(build_env(13))
        self.index=index
        self.episodes=0

    def reset(self,*,seed=None,options=None):
        j=(self.episodes//2)%TRAIN_WORLDS
        reverse=bool(self.episodes%2)
        self.episodes+=1
        raw,_=self.env.reset(seed=train_seed(self.index,j))
        if reverse:
            raw,_=swap_cue(self.env,get_world_labels(self.env))
        return raw,{}


def actor_transform(encoded,arm,global_step):
    if arm not in ARMS:
        raise ValueError(arm)
    x=np.asarray(encoded,np.float32)
    if x.shape!=(DIM,):
        raise AssertionError("Actor input width not 660")
    out=x.copy()
    active=scheduled(global_step)
    if active:
        if arm=="ppo_blackout":
            out[148:]=0.
        else:
            if np.any(out[INFORMATIVE_END:]):
                raise AssertionError("Sham tail should already be zeros")
            out[INFORMATIVE_END:]=0.
    if not np.array_equal(x[:148],out[:148]):
        raise AssertionError("Actor public observation was modified")
    return out,active,bool(not np.array_equal(out,x))


class PPOClockedF4(gym.Wrapper):
    """Global PPO step schedule, unaffected by differing episode lengths."""
    def __init__(self,index,arm):
        super().__init__(S13PPOTrainingWorlds(index))
        if arm not in ARMS:
            raise ValueError(arm)
        self.index=index
        self.arm=arm
        self.writer=SelectiveF4("context",CAPACITY,seed=81000+index)
        self.observation_space=spaces.Box(0,1,shape=(DIM,),dtype=np.float32)
        self.global_step=0
        self.seen_clock=set()
        self.effective_clock=set()
        self.max_writer_capacity=0
        self.encoder_calls=0

    def _encode(self,raw):
        features=self.writer.encode(raw)
        self.max_writer_capacity=max(self.max_writer_capacity,len(self.writer.store.order))
        out,planned,changed=actor_transform(features,self.arm,self.global_step)
        self.encoder_calls+=1
        if planned:
            self.seen_clock.add(self.global_step)
        if changed:
            self.effective_clock.add(self.global_step)
        return out

    def reset(self,*,seed=None,options=None):
        raw,info=self.env.reset(seed=seed,options=options)
        self.writer.reset()
        return self._encode(raw),info

    def step(self,action):
        raw,reward,terminated,truncated,info=self.env.step(action)
        self.global_step+=1
        return self._encode(raw),reward,terminated,truncated,info

    def metrics(self):
        return {
            "native_steps":self.global_step,
            "scheduled_unique_steps":len(self.seen_clock),
            "effective_nonzero_unique_steps":len(self.effective_clock),
            "encoder_calls":self.encoder_calls,
            "writer_max_occupancy":self.max_writer_capacity,
            "worlds_reset":self.env.episodes,
        }


def run(args):
    if args.index not in range(4) or args.worlds!=TRAIN_WORLDS or args.steps!=PPO_STEPS:
        raise ValueError("F9-P fixed new-world PPO budget")
    torch.set_num_threads(1)
    deps=ensure_dependencies()
    source=Path(args.source)
    prior=json.loads((source/"report.json").read_text(encoding="utf8"))
    validate_f9o(prior,source,args.index)
    bc_path=source/"continuous6_after_bc.zip"
    bc_sha=prior["arms"]["continuous6"]["bc_sha256"]
    if file_sha(bc_path)!=bc_sha:
        raise AssertionError("F9-O frozen BC checkpoint SHA mismatch")
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("F9-P training output directory must be fresh")
    report={
        "experiment":"F9-P long exogenous F4 blackout during PPO versus sham",
        "seed_index":args.index,
        "source_f9o_run":SOURCE_RUN,
        "source_f9o_report_sha256":file_sha(source/"report.json"),
        "source_bc_sha256":bc_sha,
        "train_base":TRAIN_BASE,"train_worlds":TRAIN_WORLDS,
        "ppo_requested_steps":PPO_STEPS,
        "mask_period":PERIOD,"normal_prefix":NORMAL_PREFIX,
        "historical_holdout_opened":False,
        "previous_confirmations_reused":False,
        "new_confirmation_opened":False,
        "training_only":True,
        "initial_BC_retrained":False,
        "dependencies":deps,
        "arms":{},
    }
    for arm in ARMS:
        wrapper=PPOClockedF4(args.index,arm)
        model=RecurrentPPO.load(str(bc_path),env=wrapper,device="cpu")
        try:
            model.set_random_seed(train_seed(args.index,0))
            start_sha=parameter_signature(model.policy)
            nparams=int(sum(p.numel() for p in model.policy.parameters()))
            if model.observation_space.shape!=(DIM,) or nparams!=380552:
                raise AssertionError("Architecture differs from F9-O source")
            model.learn(total_timesteps=PPO_STEPS)
            info=wrapper.metrics()
            expected=sum(scheduled(t) for t in range(info["native_steps"]+1))
            if info["native_steps"]!=ACTUAL_STEPS or info["scheduled_unique_steps"]!=expected:
                raise AssertionError("Exogenous scheduled blackout exposure mismatch")
            if arm=="ppo_sham" and info["effective_nonzero_unique_steps"]!=0:
                raise AssertionError("Sham unexpectedly changed informative features")
            if arm=="ppo_blackout" and info["effective_nonzero_unique_steps"]<=0:
                raise AssertionError("Treatment never ablated nonzero F4")
            checkpoint=out/f"{arm}_after_rl.zip"
            model.save(str(checkpoint))
            report["arms"][arm]={
                "initial_policy_sha256":start_sha,
                "parameter_count":nparams,
                "input_width":int(model.observation_space.shape[0]),
                "actual_rl_steps":int(model.num_timesteps),
                "clock_stats":info,
                "checkpoint_sha256":file_sha(checkpoint),
            }
            (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
            print("F9P_PPO_TRAINED",args.index,arm,
                  json.dumps(report["arms"][arm]),flush=True)
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    a,b=[report["arms"][arm] for arm in ARMS]
    for name in ("initial_policy_sha256","parameter_count","input_width","actual_rl_steps"):
        if a[name]!=b[name]:
            raise AssertionError("Initial policy or equal budgets diverged: "+name)
    if a["clock_stats"]["scheduled_unique_steps"]!=b["clock_stats"]["scheduled_unique_steps"]:
        raise AssertionError("Not an equal-opportunity intervention schedule")
    print("F9P_PPO_COMPLETE",args.index,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--worlds",type=int,default=TRAIN_WORLDS)
    p.add_argument("--steps",type=int,default=PPO_STEPS)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
