"""F9-M: matched NEW RecurrentPPO training with public-only temporal F4 dropout.

A randomized augmentation clock starts two frames after the first PUBLIC
key/ball sighting. It never reads target labels, teacher success, fork
position, cue_visible, or future rewards. Same mode for behavior cloning and policy learning.
The active-treatment input masks all 512 F4 channels; the sham-treatment
masks only the already-zero 448-channel tail. F4 writer remains intact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import ensure_dependencies, file_sha
from observatory.f9d_autonomous_navigation import (
    TrainingWorlds, checked_seed, demonstration,
)
from observatory.f9i_selective_memory import SelectiveF4, DIM, FIELDS
from observatory.f9e_f4_autonomous import clone

CAPACITY = 8
ACTIVE_END = 148 + CAPACITY * FIELDS
ARMS = ("sham", "dropout")
TRAIN_WORLDS = 600
TRAIN_EPOCHS = 40
PPO_STEPS = 20000
DROPOUT_EPISODE_PROB = 0.5
DROPOUT_LENGTHS = (2, 4)


def parameter_signature(policy):
    """Initial, equal-architecture policies must start at identical weights."""
    digest = hashlib.sha256()
    for param in policy.parameters():
        digest.update(param.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


class PublicTemporalBlackout:
    """No oracle data; episode-independent RNG and finite public cue detector."""
    def __init__(self, arm, index, episode_id):
        if arm not in ARMS or index not in range(4) or episode_id < 0:
            raise ValueError("Unapproved dropout schedule")
        self.arm = arm
        self.index = index
        self.episode_id = episode_id
        rng = np.random.default_rng(500000 + index * 20000 + episode_id)
        self.enabled = bool(rng.random() < DROPOUT_EPISODE_PROB)
        self.length = int(DROPOUT_LENGTHS[int(rng.integers(2))])
        self.first_public_event_step = None
        self.step = 0
        self.started = False
        self.elapsed = 0
        self.scheduled_steps = 0
        self.effective_steps = 0

    def transform(self, raw, encoded, public_events):
        raw = np.asarray(raw, dtype=np.float32)
        if raw.shape != (148,) or encoded.shape != (DIM,):
            raise AssertionError("Public observation width changed")
        if not np.array_equal(raw, encoded[:148]):
            raise AssertionError("Only external F4 channels can be masked")
        if not np.isfinite(encoded).all():
            raise AssertionError("Policy received nonfinite features")
        if not 0 <= public_events <= 49:
            raise ValueError("Event count must derive from public image only")
        # Object visibility may persist at other locations, so "no public
        # key/ball anywhere" is NOT a valid approximation to cue disappearance.
        # Instead start two observation steps after the FIRST public sighting.
        if public_events and self.first_public_event_step is None:
            self.first_public_event_step = self.step
        if (self.first_public_event_step is not None and
                self.step >= self.first_public_event_step + 2):
            self.started = True
        active = self.enabled and self.started and self.elapsed < self.length
        out = encoded.copy()
        if active:
            self.scheduled_steps += 1
            if self.arm == "dropout":
                out[148:] = 0.
            else:
                if np.any(out[ACTIVE_END:]):
                    raise AssertionError("Sham region must already be zero")
                out[ACTIVE_END:] = 0.
            self.effective_steps += int(not np.array_equal(out, encoded))
            self.elapsed += 1
        self.step += 1
        return out


def encoded_demo(images, arm, index, episode_id):
    writer = SelectiveF4("context", CAPACITY, seed=81000 + index)
    blackout = PublicTemporalBlackout(arm, index, episode_id)
    observations = []
    for raw in images:
        encoded = writer.encode(raw)
        observations.append(blackout.transform(raw, encoded, len(writer.new_events)))
    output = np.stack(observations)
    if output.shape != (len(images), DIM):
        raise AssertionError("Imitation feature dimension changed")
    np.testing.assert_array_equal(output[:,:148], images)
    return output, blackout


class TemporalF4TrainingObservation(gym.ObservationWrapper):
    """The same observation transformation is applied during PPO rollouts."""
    def __init__(self, env, arm, index):
        super().__init__(env)
        if arm not in ARMS:
            raise ValueError(arm)
        self.arm, self.index = arm, index
        self.episode_id = -1
        self.writer = SelectiveF4("context",CAPACITY,seed=81000+index)
        self.blackout = None
        self.observation_space = spaces.Box(0,1,shape=(DIM,),dtype=np.float32)
        self.total_scheduled = 0
        self.total_effective = 0

    def reset(self, *, seed=None, options=None):
        self.episode_id += 1
        self.writer.reset()
        self.blackout = PublicTemporalBlackout(self.arm,self.index,self.episode_id)
        return super().reset(seed=seed, options=options)

    def observation(self, raw):
        encoded = self.writer.encode(raw)
        before = self.blackout.scheduled_steps
        changed_before = self.blackout.effective_steps
        out = self.blackout.transform(raw, encoded, len(self.writer.new_events))
        self.total_scheduled += self.blackout.scheduled_steps-before
        self.total_effective += self.blackout.effective_steps-changed_before
        return out


def make_model(arm, index, worlds):
    env = TemporalF4TrainingObservation(TrainingWorlds(index, worlds),arm,index)
    return RecurrentPPO(
        "MlpLstmPolicy", env=env,
        seed=checked_seed("train",index,0),
        device="cpu", n_steps=128, batch_size=128,
        n_epochs=4, learning_rate=3e-4,
        gamma=.99, gae_lambda=.95, ent_coef=.01, verbose=0,
        policy_kwargs={
            "lstm_hidden_size":64,
            "net_arch":dict(pi=[64],vf=[64]),
        },
    )


def training_dataset(teacher, arm, index):
    augmented=[]
    stats={"scheduled_steps":0,"effective_steps":0,
           "episodes_enabled":0,"episode_count":len(teacher)}
    for episode_id, (images, actions, seen) in enumerate(teacher):
        converted, clock = encoded_demo(images,arm,index,episode_id)
        augmented.append((converted,actions,seen))
        stats["scheduled_steps"] += clock.scheduled_steps
        stats["effective_steps"] += clock.effective_steps
        stats["episodes_enabled"] += int(clock.enabled)
    return augmented,stats


def run(args):
    if (args.index not in range(4) or args.worlds != TRAIN_WORLDS or
            args.epochs != TRAIN_EPOCHS or args.timesteps != PPO_STEPS):
        raise ValueError("F9-M locked training budget")
    torch.set_num_threads(1)
    dependencies = ensure_dependencies()
    out = Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("Use a fresh output directory")
    teacher = [
        demonstration(checked_seed("train",args.index,j),reverse=rev)
        for j in range(TRAIN_WORLDS)
        for rev in (False,True)
    ]
    teacher_signature=hashlib.sha256()
    for frames,actions,_ in teacher:
        teacher_signature.update(frames.tobytes())
        teacher_signature.update(actions.tobytes())
    report = {
        "experiment":"F9-M newly trained matched temporal-F4 blackout versus sham",
        "seed_index":args.index,
        "heldout_opened":False,"confirmation_opened":False,
        "training_only":True,"training_s7_only":True,
        "architecture":"context8 MlpLstmPolicy, H=64, obs=660",
        "worlds":TRAIN_WORLDS,"epochs":TRAIN_EPOCHS,
        "requested_rl_steps":PPO_STEPS,
        "teacher_sha256":teacher_signature.hexdigest(),
        "source_sha256":file_sha(Path(__file__)),
        "dependencies":dependencies,
        "arms":{},
    }
    for arm in ARMS:
        model=make_model(arm,args.index,TRAIN_WORLDS)
        try:
            dataset, mask_stats = training_dataset(teacher,arm,args.index)
            model_initial_sha=parameter_signature(model.policy)
            params=int(sum(p.numel() for p in model.policy.parameters()))
            details={
                "parameters":params,"input_width":int(model.observation_space.shape[0]),
                "initial_parameter_sha256":model_initial_sha,
                "bc_mask_stats":mask_stats,
            }
            details["bc_losses"]=clone(model,dataset,TRAIN_EPOCHS,args.index)
            # F9-I budget: supervised imitation before PPO, no early stopping.
            model.learn(total_timesteps=PPO_STEPS)
            details["actual_rl_steps"]=int(model.num_timesteps)
            wrapped=model.get_env().envs[0]
            # Monitor and VecEnv wrappers may sit between env and our wrapper.
            while not isinstance(wrapped,TemporalF4TrainingObservation):
                wrapped=wrapped.env
            details["ppo_mask_stats"]={
                "scheduled_steps":int(wrapped.total_scheduled),
                "effective_steps":int(wrapped.total_effective),
                "episodes_started":int(wrapped.episode_id+1),
            }
            path=out/f"{arm}_after_rl.zip"
            model.save(str(path))
            details["checkpoint_sha256"]=file_sha(path)
            report["arms"][arm]=details
            print("F9M_TRAINED",args.index,arm,json.dumps({
                k:v for k,v in details.items() if k!="bc_losses"}),flush=True)
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
        (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    a,b=[report["arms"][arm] for arm in ARMS]
    for field in ("parameters","input_width","initial_parameter_sha256","actual_rl_steps"):
        if a[field]!=b[field]:
            raise AssertionError("Training arms not matched: "+field)
    if a["input_width"]!=DIM or a["actual_rl_steps"] < PPO_STEPS:
        raise AssertionError("F9-M model contract violated")
    if a["bc_mask_stats"]["effective_steps"]!=0 or b["bc_mask_stats"]["effective_steps"]<=0:
        raise AssertionError("BC sham/no-blackout intervention not verified")
    if a["ppo_mask_stats"]["effective_steps"]!=0 or b["ppo_mask_stats"]["effective_steps"]<=0:
        raise AssertionError("PPO sham/no-blackout intervention not verified")
    print("F9M_TRAINING_COMPLETE",args.index,flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--worlds",type=int,default=TRAIN_WORLDS)
    p.add_argument("--epochs",type=int,default=TRAIN_EPOCHS)
    p.add_argument("--timesteps",type=int,default=PPO_STEPS)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
