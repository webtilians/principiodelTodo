"""F9-J: distinguish retained recurrent state from F4 readout, no training.

Frozen F9-I (8-event) policy, S7/S13 development worlds ONLY.
Two independent diagnostics:
  (1) Paired, teacher-forced equal-geometry cue histories ending at the
      first cue-dependent teacher action. Factorially exchange the FROZEN
      LSTM hidden/cell state and current F4 readout. The teacher's correct
      action is evaluation-only; no teacher action/label goes to policy.
  (2) Native autonomous paired runs, perturb either the LSTM state or
      F4 readout on the first fork observation; or remove F4 *after the
      cue leaves sight*. Never change environment, action set, rewards.

The evaluator uses world labels solely to detect the fork and measure
success. Neither actor nor F4 encoder/selector receives privileged data.
The model is never retrained or updated. No heldout entry point.
"""
from __future__ import annotations

import argparse
import collections
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9d_autonomous_navigation import checked_seed, cue_visible, demonstration
from observatory.f9i_selective_memory import SelectiveF4, DIM

CAPACITY = 8
CONTROLS = (
    "normal",
    "reset_hidden_at_fork",
    "zero_f4_at_fork",
    "reset_hidden_and_zero_f4_at_fork",
    "zero_f4_after_cue",
)


def observer_fork(env, labels):
    """Fork-position probe, evaluator only, not returned to the model."""
    return int(env.unwrapped.agent_pos[0]) == int(labels["success_position"][0])


def frozen_action(model, observation, recurrent_state, first=False):
    """Only public vector and RNN state cross the policy boundary."""
    obs=np.asarray(observation,np.float32)
    if obs.shape != (DIM,):
        raise AssertionError("Actor must receive exactly 660 public/derived features")
    action,new_state=model.predict(
        obs.copy(),state=recurrent_state,
        episode_start=np.array([first]),deterministic=True)
    action=int(action)
    if not 0 <= action < 7:
        raise AssertionError("Non-native action")
    return action,new_state


def autonomous_episode(model, mode, training_index, size, seed, reverse, control):
    if control not in CONTROLS:
        raise ValueError(control)
    if mode not in ("context","zero"):
        raise ValueError(mode)
    if mode == "zero" and control in ("zero_f4_at_fork","reset_hidden_and_zero_f4_at_fork","zero_f4_after_cue"):
        raise ValueError("Empty memory has no extra features to ablate")
    env=build_env(size)
    writer=SelectiveF4(mode,CAPACITY,seed=81000+training_index)
    recurrent_state=None
    first=True
    seen_cue=False
    fork_hit=False
    memory_at_fork=None
    steps=0
    total=0.
    intervention_count=0
    try:
        raw,_=env.reset(seed=seed)
        if reverse:
            raw,_=swap_cue(env,get_world_labels(env))
        labels=get_world_labels(env)
        while True:
            encoded=writer.encode(raw)
            visible=bool(cue_visible(env))  # audit only
            seen_cue |= visible
            fork=(not fork_hit) and observer_fork(env,labels)
            intervention_start=fork
            if fork:
                fork_hit=True
                memory_at_fork=len(writer.store.order)
            current=encoded.copy()
            if (control=="zero_f4_after_cue" and seen_cue and not visible) or (
                fork and control in ("zero_f4_at_fork","reset_hidden_and_zero_f4_at_fork")):
                current[148:]=0
                intervention_count+=1
            first_for_actor=first or (
                fork and control in ("reset_hidden_at_fork","reset_hidden_and_zero_f4_at_fork"))
            action,recurrent_state=frozen_action(
                model,current,recurrent_state,first=first_for_actor)
            if fork and control in ("reset_hidden_at_fork","reset_hidden_and_zero_f4_at_fork"):
                intervention_count+=1
            raw,reward,terminated,truncated,_=env.step(action)
            total+=float(reward)
            steps+=1
            first=False
            if terminated or truncated:
                break
        pos=tuple(int(x) for x in env.unwrapped.agent_pos)
        if pos==tuple(labels["success_position"]):
            exit_side=labels["correct_side"]
        elif pos==tuple(labels["failure_position"]):
            exit_side="lower" if labels["correct_side"]=="upper" else "upper"
        else:
            exit_side="none"
        return {
            "success":int(total>0.),
            "chosen_side":exit_side,
            "correct_side":labels["correct_side"],
            "cue_seen":seen_cue,
            "fork_reached":fork_hit,
            "fork_memory_occupancy":memory_at_fork,
            "intervention_applied":intervention_count>0,
            "intervention_steps":intervention_count,
            "timeout":bool(truncated),
            "length":steps,
            "memory_events_seen":writer.serial,
        }
    finally:
        env.close()


def evaluate_autonomous(model,mode,index,size,pairs,control):
    tests=[]
    for j in range(pairs):
        seed=checked_seed("development",index,j)
        first=autonomous_episode(model,mode,index,size,seed,False,control)
        second=autonomous_episode(model,mode,index,size,seed,True,control)
        if first["correct_side"]==second["correct_side"]:
            raise AssertionError("Cue inversion did not reverse correct exit")
        tests.append([first,second])
    flat=[e for pair in tests for e in pair]
    return {
        "pairs":pairs,
        "success":statistics.mean(x["success"] for x in flat),
        "both_correct":sum(a["success"] and b["success"] for a,b in tests),
        "valid_flips":sum(a["chosen_side"] != b["chosen_side"] and
                          a["chosen_side"]!="none" and b["chosen_side"]!="none"
                          for a,b in tests),
        "cue_seen":statistics.mean(x["cue_seen"] for x in flat),
        "fork_reached":statistics.mean(x["fork_reached"] for x in flat),
        "intervention_applied":statistics.mean(x["intervention_applied"] for x in flat),
        "timeout":statistics.mean(x["timeout"] for x in flat),
        "mean_length":statistics.mean(x["length"] for x in flat),
        "episodes":tests,
    }


def teacher_fork_data(model,index,size,seed,reverse):
    """Force only the action sequence, not policy inputs or labels."""
    frames,actions,_=demonstration(seed,size=size,reverse=reverse)
    writer=SelectiveF4("context",CAPACITY,seed=81000+index)
    diverge=None
    # Caller supplies exact fork index as part of the evaluation-only fixture.
    return frames,actions,writer


def inspect_forced_pair(model,index,size,seed):
    original,actions_a,writer_a=teacher_fork_data(model,index,size,seed,False)
    swapped,actions_b,writer_b=teacher_fork_data(model,index,size,seed,True)
    fork=next((i for i,(x,y) in enumerate(zip(actions_a,actions_b)) if x!=y),None)
    if fork is None or fork>=min(len(original),len(swapped)):
        raise AssertionError("No native teacher decision fork found")
    if not np.array_equal(original[fork],swapped[fork]):
        raise AssertionError("Original public observation differs at decision")
    if not np.array_equal(actions_a[:fork],actions_b[:fork]):
        raise AssertionError("Teacher paths differ before intended fork")
    states=[]
    memory_inputs=[]
    state_distances=[]
    for images,writer in ((original,writer_a),(swapped,writer_b)):
        state=None
        for t in range(fork):
            _,state=frozen_action(model,writer.encode(images[t]),state,first=t==0)
        states.append(state)
        memory_inputs.append(writer.encode(images[fork]))
    raw_equal=np.array_equal(memory_inputs[0][:148],memory_inputs[1][:148])
    if not raw_equal:
        raise AssertionError("Public fork frames differ")
    memory_equal=np.array_equal(memory_inputs[0][148:],memory_inputs[1][148:])
    state_delta=sum(float(np.linalg.norm(a-b))
                    for a,b in zip(states[0],states[1]))
    decisions={}
    for h in (0,1):
        for m in (0,1):
            action,_=frozen_action(model,memory_inputs[m],states[h])
            decisions[f"hidden{h}/memory{m}"]=action
    decisions["teacher_original"]=int(actions_a[fork])
    decisions["teacher_swapped"]=int(actions_b[fork])
    return {
        "seed":seed, "size":size,"fork_step":fork,
        "raw_equal":raw_equal,"external_memory_equal":memory_equal,
        "hidden_state_l2_difference":state_delta,
        "external_memory_occupancy": [len(writer_a.store.order),len(writer_b.store.order)],
        "correct_teacher_actions_differ":bool(actions_a[fork]!=actions_b[fork]),
        "predicted_actions":decisions,
        "natural_correct_both":int(
            decisions["hidden0/memory0"]==actions_a[fork] and
            decisions["hidden1/memory1"]==actions_b[fork]),
        "hidden_swap_changes_action":int(
            decisions["hidden0/memory0"] != decisions["hidden1/memory0"] or
            decisions["hidden1/memory1"] != decisions["hidden0/memory1"]),
        "memory_swap_changes_action":int(
            decisions["hidden0/memory0"] != decisions["hidden0/memory1"] or
            decisions["hidden1/memory1"] != decisions["hidden1/memory0"]),
    }


def run(args):
    torch.set_num_threads(1)
    if not 1<=args.pairs<=10:
        raise ValueError("Development world budget restricted to 10 paired worlds per seed")
    source=Path(args.source)
    if not source.is_file():
        raise FileNotFoundError(source)
    report_path=source.parent/"report.json"
    original=json.loads(report_path.read_text())
    if original.get("event_capacity")!=8 or original["seed_index"]!=args.index:
        raise AssertionError("Must use original F9-I capacity8 training checkpoint")
    if original["heldout_opened"] is not False:
        raise AssertionError("Source touched heldout")
    if original["models"]["context"]["after_rl_sha256"] != file_sha(source):
        raise AssertionError("F9-I trained checkpoint SHA mismatch")

    model=RecurrentPPO.load(str(source),device="cpu")
    zero_path=source.parent/"zero_after_rl.zip"
    if original["models"]["zero"]["after_rl_sha256"] != file_sha(zero_path):
        raise AssertionError("Matched control checkpoint SHA mismatch")
    zero=RecurrentPPO.load(str(zero_path),device="cpu")

    results={
        "experiment":"F9-J frozen recurrent versus external F4 attribution",
        "source_run":40928547873,
        "training_performed":False,
        "heldout_opened":False,
        "source_sha256":file_sha(Path(__file__)),
        "seed_index":args.index,
        "capacity":CAPACITY,
        "model_sha256":file_sha(source),
        "zero_sha256":file_sha(zero_path),
        "paired_worlds_per_size":args.pairs,
        "autonomous":{},
        "controlled_teacher_forks":{},
    }

    for size in (7,13):
        for mode,policy,controls in (
            ("context",model,CONTROLS),
            ("zero",zero,("normal","reset_hidden_at_fork")),
        ):
            for control in controls:
                stats=evaluate_autonomous(policy,mode,args.index,size,args.pairs,control)
                results["autonomous"][f"S{size}/{mode}/{control}"]=stats
                print("F9J_AUTONOMOUS",args.index,size,mode,control,
                      json.dumps({k:v for k,v in stats.items() if k!="episodes"}),flush=True)
        controlled=[
            inspect_forced_pair(model,args.index,size,checked_seed("development",args.index,j))
            for j in range(args.pairs)
        ]
        n=len(controlled)
        summary={
            "pairs":n,
            "equal_current_observations":sum(r["raw_equal"] for r in controlled),
            "equal_external_memory_readouts":sum(r["external_memory_equal"] for r in controlled),
            "distinct_hidden_states":sum(r["hidden_state_l2_difference"]>1e-8 for r in controlled),
            "natural_correct_both_teacher_actions":sum(r["natural_correct_both"] for r in controlled),
            "changing_only_hidden_changes_action":sum(r["hidden_swap_changes_action"] for r in controlled),
            "changing_only_external_memory_changes_action":sum(r["memory_swap_changes_action"] for r in controlled),
            "cases":controlled,
        }
        results["controlled_teacher_forks"][f"S{size}"]=summary
        print("F9J_FORCED_FORK",args.index,size,json.dumps({
            k:v for k,v in summary.items() if k!="cases"}),flush=True)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(results,indent=2),encoding="utf8")
    print("F9J_COMPLETE",args.index,flush=True)
    return results


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--pairs",type=int,default=10)
    run(p.parse_args())


if __name__=="__main__":
    main()
