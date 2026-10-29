"""F9-I: native seven-action LSTM with F9-G contextual F4 vs equal-width zero.

Strict exploratory comparison; original MiniGrid rewards, no teleport,
no shaped rewards, teacher ONLY generates separate TRAIN action labels.
Actor and F4 writer consume 148-d public observations, never hidden
cue/target/route labels. Selectors are FIFO, contextual diversity and random reservoir; none is LearnedRelevance. The F9-G representation gate is a prerequisite.

Ablations erase/swap use the *same frozen trained F4 checkpoint*.
Report both sides of paired cue flips: 50% without correct-side flips
is NOT evidence of memory. Heldout partition has no entry point.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import torch

from observatory.f9_external_minigrid import ensure_dependencies, file_sha
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9d_autonomous_navigation import (
    TrainingWorlds, checked_seed, cue_visible, demonstration, actor_action,
)
from observatory.f9e_f4_autonomous import clone
from observatory.f9i_selective_memory import (
    SelectiveF4, SelectiveObservation, DIM, CAPACITY, FIELDS, corrupt_one, unpack,
)
# Configured once before model construction; CLI accepts only preassigned budgets.
EVENT_CAPACITY = 64
SELECTOR_SEED = 0

MODES = ("zero", "fifo", "context", "random")
ABLATIONS = ("normal", "erase", "corrupt_initial", "mask_cue")


def augment_demonstrations(examples, mode):
    """All 1200 identical teacher trajectories, only writer input differs."""
    if mode not in MODES:
        raise ValueError(mode)
    result = []
    for images, actions, seen in examples:
        writer = SelectiveF4(mode, EVENT_CAPACITY, SELECTOR_SEED)
        converted = np.stack([writer.encode(frame) for frame in images])
        if converted.shape != (len(actions), DIM):
            raise AssertionError("Unmatched imitation input dimension")
        np.testing.assert_array_equal(converted[:, :148], images)
        if mode == "zero":
            assert not np.any(converted[:, 148:])
        result.append((converted, actions, seen))
    return result


def make_model(mode, seed_index, worlds):
    from sb3_contrib import RecurrentPPO
    if mode not in MODES:
        raise ValueError(mode)
    env = SelectiveObservation(TrainingWorlds(seed_index, worlds), mode=mode, capacity=EVENT_CAPACITY, seed=SELECTOR_SEED)
    return RecurrentPPO(
        "MlpLstmPolicy",
        env=env, seed=checked_seed("train", seed_index, 0),
        device="cpu", n_steps=128, batch_size=128,
        n_epochs=4, learning_rate=3e-4,
        gamma=.99, gae_lambda=.95, ent_coef=.01, verbose=0,
        policy_kwargs={
            "lstm_hidden_size": 64,
            "net_arch": dict(pi=[64], vf=[64]),
        },
    )


def paired_rollout(model, mode, size, seed, reverse=False, control="normal"):
    """Native navigation with separate privileged evaluation observer.

    For mask_cue, the counterfactual image is computed in a shadow
    environment at the same pose and used ONLY to redact pixels before
    the actor sees/writes them. Other controls modify only memory readout.
    """
    if mode not in MODES:
        raise ValueError(mode)
    if control not in ABLATIONS:
        raise ValueError(control)
    if mode == "zero" and control in ("erase", "corrupt_initial"):
        raise ValueError("F4-only readout intervention needs trained F4")

    env = build_env(size)
    shadow = build_env(size)
    memory = SelectiveF4(mode, EVENT_CAPACITY, SELECTOR_SEED)
    cue_ids = set()
    initial_id = None
    intervention_steps = 0
    fork_audit = None
    state, first = None, True
    trace = []
    seen_any = False
    total = 0.
    try:
        raw, _ = env.reset(seed=seed)
        if reverse:
            raw, _ = swap_cue(env, get_world_labels(env))
        labels = get_world_labels(env)

        if shadow is not None:
            shadow.reset(seed=seed)
            if not reverse:
                swap_cue(shadow, get_world_labels(shadow))

        def observed(raw_frame):
            nonlocal initial_id, intervention_steps
            current = raw_frame.copy()
            other = shadow.observation(shadow.unwrapped.gen_obs())
            difference = np.any(
                current[:147].reshape(7,7,3) != other[:147].reshape(7,7,3), axis=-1)
            if control == "mask_cue":
                current[:147].reshape(7,7,3)[difference] = 0
            encoded = memory.encode(current)
            # Provenance is evaluator-only; never sent into the selector/actor.
            for event in memory.new_events:
                _, x, y, _, _ = unpack(event.payload)
                if difference[x,y]:
                    cue_ids.add(event.key)
                    if initial_id is None:
                        initial_id = event.key
            if control == "erase":
                encoded[148:] = 0
            if control == "corrupt_initial" and not difference.any():
                changed = corrupt_one(encoded,memory,initial_id)
                intervention_steps += int(not np.array_equal(changed,encoded))
                encoded = changed
            return encoded

        actor_observation = observed(raw)
        while True:
            # This oracle probe is for audit only; never policy input.
            visible = bool(cue_visible(env))
            seen_any |= visible
            action, state = actor_action(model, actor_observation, state, first, True)
            if not 0 <= action < 7:
                raise AssertionError("Non-native action")
            position = [int(v) for v in env.unwrapped.agent_pos]
            audit = {
                "initial_written":initial_id is not None,
                "initial_retained":initial_id in memory.store.saved,
                "any_cue_retained":bool(cue_ids.intersection(memory.store.saved)),
                "events_since_initial":memory.serial-1-initial_id if initial_id is not None else None,
            }
            # First arrival at the branch corridor; audit only, no policy access.
            if fork_audit is None and position[0] == labels["success_position"][0]:
                fork_audit = dict(audit)
            # Hash the readout, never pass truth label to model.
            signature = hashlib.sha256(
                actor_observation[148:].tobytes()).hexdigest()[:16]
            trace.append({
                "step":len(trace), **audit,
                "action":action,
                "position":position,
                "cue_visible":visible,
                "f4_occupied":len(memory.store.order),
                "f4_events_seen":int(memory.serial),
                "f4_readout_hash16":signature,
            })
            raw, reward, terminated, truncated, _ = env.step(action)
            if shadow is not None:
                shadow.step(action)
            total += float(reward)
            first = False
            if terminated or truncated:
                break
            actor_observation = observed(raw)

        end = tuple(int(v) for v in env.unwrapped.agent_pos)
        if end == tuple(labels["success_position"]):
            chosen = labels["correct_side"]
        elif end == tuple(labels["failure_position"]):
            chosen = "upper" if labels["correct_side"] == "lower" else "lower"
        else:
            chosen = "none"
        return {
            "seed":seed, "reverse":reverse,
            "cue":labels["cue"], "correct_side":labels["correct_side"],
            "chosen_side":chosen, "success":int(total > 0),
            "native_return":total, "length":len(trace),
            "timeout":bool(truncated), "cue_seen":seen_any,
            "memory_ever_written":any(x["f4_occupied"] > 0 for x in trace),
            "final_event_count":int(memory.serial),
            "final_memory_occupied":len(memory.store.order),
            "trace":trace, "fork_audit":fork_audit,
            "intervention_steps":intervention_steps,
        }
    finally:
        env.close()
        if shadow is not None:
            shadow.close()


def evaluate(model, mode, index, size, pairs, control, trace_out):
    rows = []
    for episode in range(pairs):
        seed = checked_seed("development", index, episode)
        first = paired_rollout(model, mode, size, seed, False, control)
        second = paired_rollout(model, mode, size, seed, True, control)
        if first["correct_side"] == second["correct_side"]:
            raise AssertionError("Swapped cue did not reverse correct exit")
        rows.append((first, second))
        with trace_out.open("a", encoding="utf8") as f:
            f.write(json.dumps({
                "model":mode, "training_index":index,
                "environment":f"S{size}", "intervention":control,
                "paired_rollouts":[first,second],
            }) + "\n")

    flat=[v for pair in rows for v in pair]
    both_correct=sum(a["success"] and b["success"] for a,b in rows)
    flips=sum(
        a["chosen_side"] != b["chosen_side"] and
        a["chosen_side"] != "none" and b["chosen_side"] != "none"
        for a,b in rows
    )
    cue_obs=sum(v["cue_seen"] for v in flat)
    return {
        "paired_worlds":pairs,
        "original_success":statistics.mean(a["success"] for a,_ in rows),
        "swapped_success":statistics.mean(b["success"] for _,b in rows),
        "combined_success":statistics.mean(v["success"] for v in flat),
        "both_correct":both_correct,
        "valid_side_flips":flips,
        "cue_observed_rate":cue_obs/len(flat),
        "memory_written_rate":statistics.mean(v["memory_ever_written"] for v in flat),
        "mean_events_written":statistics.mean(v["final_event_count"] for v in flat),
        "mean_occupied_slots":statistics.mean(v["final_memory_occupied"] for v in flat),
        "timeout_rate":statistics.mean(v["timeout"] for v in flat),
        "mean_length":statistics.mean(v["length"] for v in flat),
        "chosen_side_counts":dict(collections.Counter(v["chosen_side"] for v in flat)),
        "fork_reached_rate":statistics.mean(v["fork_audit"] is not None for v in flat),
        "initial_retained_at_fork_count":sum(bool(v["fork_audit"] and v["fork_audit"]["initial_retained"]) for v in flat),
        "any_cue_retained_at_fork_count":sum(bool(v["fork_audit"] and v["fork_audit"]["any_cue_retained"]) for v in flat),
        "initial_written_at_fork_count":sum(bool(v["fork_audit"] and v["fork_audit"]["initial_written"]) for v in flat),
        "corruption_applied_episodes":sum(v["intervention_steps"] > 0 for v in flat),
    }


def run(args):
    global EVENT_CAPACITY, SELECTOR_SEED
    EVENT_CAPACITY = args.capacity
    SELECTOR_SEED = 81000 + args.index
    torch.set_num_threads(1)
    deps=ensure_dependencies()
    args.output.mkdir(parents=True,exist_ok=True)
    traces=args.output/"trajectories.jsonl"
    if traces.exists():
        raise FileExistsError("Use a fresh output folder")
    training = [
        demonstration(checked_seed("train",args.index,j),reverse=swap)
        for j in range(args.worlds) for swap in (False,True)
    ]
    result={
        "experiment":"F9-I contextual F4 versus equal-width zero native navigation",
        "origin":"F9-H frozen design; equal event budget selectors",
        "event_capacity":EVENT_CAPACITY, "selector_seed":SELECTOR_SEED,
        "git_sha":__import__("os").environ.get("GITHUB_SHA"),
        "heldout_opened":False,
        "not_confirmatory":True,
        "shaping":False,
        "teleport":False,
        "native_actions":7,
        "seed_index":args.index,
        "config":{"worlds":args.worlds,"epochs":args.epochs,
                  "requested_rl_steps":args.timesteps,"paired_eval_worlds":args.episodes},
        "dependencies":deps,
        "source_sha256":file_sha(Path(__file__)),
        "models":{},
    }

    for mode in MODES:
        model=make_model(mode,args.index,args.worlds)
        try:
            examples=augment_demonstrations(training,mode)
            metrics={
                "parameters":int(sum(p.numel() for p in model.policy.parameters())),
                "input_width":model.observation_space.shape[0],
                "bc_losses":clone(model,examples,args.epochs,args.index),
            }
            pre=args.output/f"{mode}_before_rl.zip"
            model.save(pre)
            metrics["before_rl_sha256"]=file_sha(pre)
            metrics["before_rl"]={
                f"S{size}/normal":evaluate(model,mode,args.index,size,args.episodes,"normal",traces)
                for size in (7,13)
            }
            model.learn(total_timesteps=args.timesteps)
            metrics["actual_rl_steps"]=int(model.num_timesteps)
            post=args.output/f"{mode}_after_rl.zip"
            model.save(post)
            metrics["after_rl_sha256"]=file_sha(post)
            metrics["after_rl"]={}
            for size in (7,13):
                for control in (ABLATIONS if mode!="zero" else ("normal","mask_cue")):
                    summary=evaluate(model,mode,args.index,size,args.episodes,control,traces)
                    metrics["after_rl"][f"S{size}/{control}"]=summary
                    print("F9I_RESULT",args.index,mode,
                          f"S{size}",control,json.dumps(summary),flush=True)
            result["models"][mode]=metrics
            (args.output/"report.json").write_text(json.dumps(result,indent=2),encoding="utf8")
        finally:
            model.get_env().close()

    for field in ("parameters","input_width","actual_rl_steps"):
        assert len({m[field] for m in result["models"].values()}) == 1
    print("F9I_FINISHED",args.index,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capacity",type=int,choices=(8,64),default=64)
    parser.add_argument("--index",type=int,choices=range(4),required=True)
    parser.add_argument("--worlds",type=int,default=600)
    parser.add_argument("--epochs",type=int,default=40)
    parser.add_argument("--timesteps",type=int,default=20000)
    parser.add_argument("--episodes",type=int,default=10)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if not (1 <= args.worlds <= 600 and 1 <= args.epochs <= 40 and
            128 <= args.timesteps <= 20000 and 1 <= args.episodes <= 100):
        parser.error("Budgets are bounded at previously developed F9-D/F9-E")
    run(args)


if __name__=="__main__":
    main()
