"""F9-Z-A: prospective, no-head-training causal memory-use positivity pilot.

New worlds only; identical frozen F9-P actor and SelectiveF4('context').
Factorial retention capacity N in {1,4,8} and READ horizon W in {1,4,8}.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from observatory.f9_external_minigrid import ensure_dependencies
from observatory.f9q_train_cue_probes import validate_sources
from observatory.f9q_causal_fork import matched_pair, replay_to_decision, clone_state, REGIMES
from observatory.f9i_selective_memory import SelectiveF4, DIM
from observatory.f9v_lexicographic_causal_f4 import one_seed_cases
from observatory.f9v_frozen_confirm import validate_original_implementation
from observatory.f9j_recurrent_memory_attribution import frozen_action
from observatory.f9n_stage_autonomy import StageObserver

BASE = 11_100_000
WORLDS = 24
CAPACITIES = (1, 4, 8)
WINDOWS = (1, 4, 8)
ORDER = ((8,1),(8,4),(8,8),(4,1),(4,4),(4,8),(1,1),(1,4),(1,8))
CELLS = (("key","history_f4_normal"),
         ("ball","history_f4_normal"),
         ("key","history_f4_postcue_off"),
         ("ball","history_f4_postcue_off"))
ZERO_TAIL = np.zeros(DIM-148, np.float32)


def seed_for(index, j):
    if index not in range(4) or j not in range(WORLDS):
        raise ValueError("Only 11100000+ new pilot worlds, not old holdouts")
    return BASE + 10_000*index + j


def fixed_snapshot(actor, index, pair, cue_index, regime, capacity):
    """Reproduce original F9-Q state exactly at N=8, otherwise same writer family."""
    if capacity not in CAPACITIES or regime not in REGIMES or cue_index not in (0,1):
        raise ValueError("Unregistered factorial treatment")
    frames=pair["frames"][cue_index]
    times=pair["times"]
    writer=SelectiveF4("context",capacity,seed=81000+index)
    recurrent=None
    for t in range(times["fork"]+1):
        encoded=writer.encode(frames[t])
        assert encoded.shape==(DIM,)
        assert np.array_equal(encoded[:148],frames[t])
        if t==times["fork"]:
            assert recurrent is not None
            return {
                "state":clone_state(recurrent),
                "raw":encoded[:148].copy(),
                "fork_tail":encoded[148:].copy(),
                "writer":writer,
                "fork_step":times["fork"],
            }
        masked=regime=="history_f4_postcue_off" and t>=times["post_cue"]
        if masked:encoded[148:]=0.
        _,recurrent=frozen_action(actor,encoded,recurrent,first=t==0)
    raise AssertionError("Missing fixed fork state")


def branch(actor,index,pair,receiver,snapshot,window,read):
    if window not in WINDOWS or read not in (0,1):
        raise ValueError("Not a planned fork intervention")
    env, raw, labels=replay_to_decision(
        pair["seed"],bool(receiver),
        pair["actions"][receiver][:pair["times"]["fork"]],
        snapshot["raw"])
    state=clone_state(snapshot["state"])
    writer=copy.deepcopy(snapshot["writer"])
    observer=StageObserver(labels,13)
    observer.observe(env.unwrapped.agent_pos,0)
    first_action=None
    step=0
    try:
        while True:
            if step==0:
                tail=snapshot["fork_tail"].copy()
            else:
                encoded=writer.encode(raw)
                assert np.array_equal(encoded[:148],raw)
                tail=encoded[148:].copy()
            if not read or step>=window:
                tail[:]=0.
            x=np.concatenate((raw,tail)).astype(np.float32)
            assert x.shape==(DIM,) and np.array_equal(x[:148],raw)
            action,state=frozen_action(actor,x,state,first=False)
            if first_action is None:first_action=int(action)
            raw,_,ended,truncated,_=env.step(action)
            step+=1
            observer.observe(env.unwrapped.agent_pos,step)
            if observer.first_commit_step is not None or ended or truncated:
                break
        summary=observer.summary()
        return {
            "first_action":first_action,
            "first_arm":summary["first_committed_side"],
            "first_arm_correct":int(summary["first_choice_correct"]),
            "committed":int(summary["first_committed"]),
            "steps_until_first_arm_or_exit":step,
            "read":int(read),
            "window":window,
        }
    finally:
        env.close()


def compute_world(actor,index,j):
    seed=seed_for(index,j)
    pair=matched_pair(actor,index,seed)
    result=[]
    for capacity in CAPACITIES:
        for regime in REGIMES:
            states=[fixed_snapshot(actor,index,pair,i,regime,capacity)
                    for i in range(2)]
            assert np.array_equal(states[0]["raw"],states[1]["raw"])
            for receiver in (0,1):
                state=states[receiver]
                for window in WINDOWS:
                    read=branch(actor,index,pair,receiver,state,window,1)
                    skip=branch(actor,index,pair,receiver,state,window,0)
                    r,s=read["first_arm_correct"],skip["first_arm_correct"]
                    result.append({
                        "seed":seed,"actor_index":index,"capacity":capacity,
                        "window":window,"regime":regime,
                        "cue":pair["labels"][receiver]["cue"],
                        "reverse":bool(receiver),
                        "fork_step":pair["times"]["fork"],
                        "f4_nonempty":int(np.any(state["fork_tail"])),
                        "effect":int(r-s),
                        "read_branch":read,"skip_branch":skip,
                        "public_equal_between_cues":True,
                        "same_start_hc":True,
                    })
    assert len(result)==36, len(result)
    # READ=0 is independent of the designated read window.
    for capacity in CAPACITIES:
        for regime in REGIMES:
            for cue in ("key","ball"):
                sub=[x for x in result if x["capacity"]==capacity and
                     x["regime"]==regime and x["cue"]==cue]
                assert len(sub)==3
                baseline={k:v for k,v in sub[0]["skip_branch"].items()
                          if k!="window"}
                assert all({k:v for k,v in row["skip_branch"].items()
                            if k!="window"}==baseline for row in sub)
    # Anchor reproduces the original F9-V one-time READ at cap=8 exactly.
    if j==0:
        original=one_seed_cases(actor,index,seed)
        for c in original:
            m=[x for x in result if x["capacity"]==8 and x["window"]==1 and
               x["cue"]==c["cue"] and x["regime"]==c["regime"]]
            assert len(m)==1
            for arm in ("read_branch","skip_branch"):
                for key in ("first_action","first_arm","first_arm_correct",
                            "committed","steps_until_first_arm_or_exit","read"):
                    assert m[0][arm][key]==c[arm][key], (
                        index,seed,c["cue"],c["regime"],arm,key,
                        m[0][arm][key],c[arm][key])
            assert m[0]["effect"]==c["effect"]
    return result


def summarize(cases):
    result={}
    for capacity,window in ORDER:
        by_cell={}
        for cue,regime in CELLS:
            part=[x for x in cases if x["capacity"]==capacity and
                  x["window"]==window and x["cue"]==cue and x["regime"]==regime]
            assert len(part)==WORLDS
            effects={str(d):sum(v["effect"]==d for v in part)
                     for d in (-1,0,1)}
            by_cell[f"{cue}|{regime}"]={
                "worlds":WORLDS,"effects":effects,
                "positive":effects["1"],
                "nonpositive":effects["0"]+effects["-1"],
                "nonempty_readout":sum(v["f4_nonempty"] for v in part),
                "read_correct":sum(v["read_branch"]["first_arm_correct"] for v in part),
                "skip_correct":sum(v["skip_branch"]["first_arm_correct"] for v in part),
            }
        result[f"N{capacity}_W{window}"]=by_cell
    return result


def execute(args):
    if args.index not in range(4):raise ValueError(args.index)
    torch.set_num_threads(1)
    ensure_dependencies()
    validate_original_implementation()
    paths,sha,provenance=validate_sources(args.bc_source,args.ppo_source,args.index)
    from sb3_contrib import RecurrentPPO
    actor=RecurrentPPO.load(str(paths["ppo_sham"]),device="cpu")
    actor.policy.set_training_mode(False)
    for param in actor.policy.parameters():
        param.requires_grad_(False)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    cases=[]
    for j in range(WORLDS):
        cases.extend(compute_world(actor,args.index,j))
        print("F9Z_NEW_PILOT_WORLD",args.index,j,seed_for(args.index,j),flush=True)
    assert len(cases)==36*WORLDS
    assert {r["seed"] for r in cases}=={seed_for(args.index,j) for j in range(WORLDS)}
    result={
        "experiment":"F9-Z-A causal support feasibility",
        "stage":"PILOT_NOT_TRAIN_DEV","index":args.index,
        "worlds":WORLDS,"cases_count":len(cases),
        "actor_sha256":sha["ppo_sham"],
        "source_provenance":provenance,
        "full_factorial_3x3":True,
        "anchor_original_f9v_w1_n8_passed":True,
        "skip_invariant_across_read_window":True,
        "actor_retrained":False,"head_trained":False,"F4_algorithm_modified":False,
        "prior_holdouts_reused":False,
        "future_F9Z_train_dev_confirm_opened":False,
        "world_seeds":[seed_for(args.index,j) for j in range(WORLDS)],
        "summary":summarize(cases),
        "cases":cases,
    }
    (out/"report.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("F9Z_PILOT_SUMMARY",args.index,json.dumps(result["summary"],ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    execute(p.parse_args())


if __name__=="__main__":
    main()
