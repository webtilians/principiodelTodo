"""F9-N: decompose autonomous S13 navigation into approach, commitment and exit.

Prospective 400000+ seeds only. Eight *frozen* F9-M model checkpoints.
No teacher forcing, oracle feature, new policy, probe, or historical holdout.
The geometry/labels below are OBSERVER-ONLY after the actor acts.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import RecurrentPPO

from observatory.f9_external_minigrid import file_sha
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9d_autonomous_navigation import cue_visible
from observatory.f9i_selective_memory import SelectiveF4, DIM
from observatory.f9j_recurrent_memory_attribution import CAPACITY, frozen_action
from observatory.f9m_confirm import validate_training
from observatory.f9m_temporal_dropout_training import ARMS

ORIGIN_RUN = 40950579984
SEED_BASE = 400_000
PAIRS_BY_SIZE = {7: 20, 13: 40}
CONDITIONS = ("normal", "cue_lost_continuous", "fixed_4")
CLOCK_TIMES = frozenset(range(3,7))


def confirmation_seed(index, size, j):
    """Locked new F9-N partition; never access 300000 or 900000."""
    if index not in range(4) or size not in PAIRS_BY_SIZE or not 0 <= j < PAIRS_BY_SIZE[size]:
        raise ValueError("F9-N preassigned worlds only")
    seed=SEED_BASE + 10_000*index + j
    if seed >= 900_000 or (1000 <= seed < 41000) or (100000 <= seed < 141000) or (
        300000 <= seed < 341000):
        raise AssertionError("Confirmation seed overlaps older partitions")
    return seed


def mask_actor(actor_input, condition, t, seen, visible):
    """Never change public 148 features or mutate the F4 writer."""
    if condition not in CONDITIONS:
        raise ValueError(condition)
    features=np.asarray(actor_input, dtype=np.float32)
    if features.shape != (DIM,):
        raise AssertionError("Agent input must be 660 dimensions")
    scheduled = (condition == "fixed_4" and t in CLOCK_TIMES) or (
        condition == "cue_lost_continuous" and seen and not visible)
    output=features.copy()
    if scheduled:
        output[148:]=0.
    if not np.array_equal(output[:148],features[:148]):
        raise AssertionError("Ablation altered public frame")
    return output, bool(scheduled), bool(scheduled and np.any(features[148:] != 0))


class StageObserver:
    """Track geometry after actions; never passes this information to policy."""
    def __init__(self, labels, size):
        self.labels=labels
        self.fork_x=int(labels["success_position"][0])
        self.mid=size//2
        self.junction_step=None
        self.first_commit_step=None
        self.first_commit_side="none"
        self.arm_without_center=False
        self.last_position=None
        self.steps_after_commit=0

    def observe(self, position, step):
        x,y=map(int,position)
        self.last_position=(x,y)
        if x==self.fork_x and y==self.mid and self.junction_step is None:
            self.junction_step=step
        if x==self.fork_x and y!=self.mid and self.first_commit_step is None:
            if self.junction_step is None:
                self.arm_without_center=True
            else:
                self.first_commit_step=step
                self.first_commit_side="upper" if y < self.mid else "lower"

    def summary(self):
        reached=self.junction_step is not None
        committed=self.first_commit_step is not None
        if committed and not reached:
            raise AssertionError("Commitment without a junction")
        correct=int(committed and self.first_commit_side==self.labels["correct_side"])
        return dict(junction_arrived=bool(reached),
                    first_committed=bool(committed),
                    first_committed_side=self.first_commit_side,
                    first_choice_correct=correct,
                    first_choice_wrong=int(committed and not correct),
                    arm_without_center=bool(self.arm_without_center),
                    junction_step=self.junction_step,
                    first_commit_step=self.first_commit_step)


def autonomous_episode(model,index,size,seed,reverse,condition):
    if condition not in CONDITIONS or size not in PAIRS_BY_SIZE or index not in range(4):
        raise ValueError("Unapproved F9-N comparison")
    env=build_env(size)
    writer=SelectiveF4("context",CAPACITY,seed=81000+index)
    state=None
    total=0.
    t=0
    seen=False
    scheduled=0
    effective=0
    final_reward=0.
    first_seen_step=None
    observer=None
    try:
        raw,_=env.reset(seed=seed)
        labels=get_world_labels(env)
        if reverse:
            raw,labels=swap_cue(env,labels)
        observer=StageObserver(labels,size)
        observer.observe(env.unwrapped.agent_pos,0)
        while True:
            encoded=writer.encode(raw)
            visible=bool(cue_visible(env))
            if visible and first_seen_step is None:
                first_seen_step=t
            seen=bool(seen or visible)
            input_vector,masked,changed=mask_actor(encoded,condition,t,seen,visible)
            scheduled+=int(masked)
            effective+=int(changed)
            action,state=frozen_action(model,input_vector,state,first=t==0)
            raw,reward,terminated,truncated,_=env.step(action)
            total+=float(reward)
            final_reward=float(reward)
            t+=1
            observer.observe(env.unwrapped.agent_pos,t)
            if terminated or truncated:
                break
        result=observer.summary()
        end=tuple(map(int,env.unwrapped.agent_pos))
        if end==tuple(labels["success_position"]):
            exit_side=labels["correct_side"]
        elif end==tuple(labels["failure_position"]):
            exit_side=("lower" if labels["correct_side"]=="upper" else "upper")
        else:
            exit_side="none"
        success=int(total>0)
        if (exit_side!="none" and result["junction_arrived"] is False):
            # If this happens, report it as protocol sensitivity rather than
            # silently excluding successful unusual trajectories.
            result["terminal_without_center"]=True
        else:
            result["terminal_without_center"]=False
        if success and exit_side!=labels["correct_side"]:
            raise AssertionError("Reward/goal side mismatch")
        result.update({
            "cue":labels["cue"],"correct_side":labels["correct_side"],
            "cue_seen":bool(seen),"first_seen_step":first_seen_step,
            "success":success,"exit_side":exit_side,"timeout":bool(truncated),
            "length":t,"scheduled_masks":scheduled,
            "effective_masks":effective,
            "clock_4_complete":bool(condition=="fixed_4" and scheduled==4),
            "intervention_applied":effective>0,
            "post_commit_steps":None if result["first_commit_step"] is None
                else t-result["first_commit_step"],
            "final_reward":final_reward,
        })
        if condition=="normal" and scheduled!=0:
            raise AssertionError("Sham normal was unexpectedly masked")
        if condition=="fixed_4" and scheduled!=len(CLOCK_TIMES.intersection(range(t))):
            raise AssertionError("Wall-clock ablation not exogenous")
        return result
    finally:
        env.close()


def pair_summary(pairs):
    """Report ALL world pairs; never drop no-fork and no-commit episodes."""
    n=len(pairs)
    if not n:
        raise ValueError("No paired worlds")
    for a,b in pairs:
        if a["correct_side"]==b["correct_side"] or a["cue"]==b["cue"]:
            raise AssertionError("Cue inversion not a true opposite choice")
    episodes=[r for pair in pairs for r in pair]
    total=len(episodes)
    counts={
        "cue_seen":sum(r["cue_seen"] for r in episodes),
        "junction_arrived":sum(r["junction_arrived"] for r in episodes),
        "first_committed":sum(r["first_committed"] for r in episodes),
        "first_choice_correct":sum(r["first_choice_correct"] for r in episodes),
        "first_choice_wrong":sum(r["first_choice_wrong"] for r in episodes),
        "success":sum(r["success"] for r in episodes),
        "timeout":sum(r["timeout"] for r in episodes),
        "arm_without_center":sum(r["arm_without_center"] for r in episodes),
        "terminal_without_center":sum(r["terminal_without_center"] for r in episodes),
        "scheduled_masks":sum(r["scheduled_masks"] for r in episodes),
        "effective_masks":sum(r["effective_masks"] for r in episodes),
        "episodes_intervened":sum(r["intervention_applied"] for r in episodes),
        "clock_4_complete":sum(r["clock_4_complete"] for r in episodes),
    }
    if not 0 <= counts["first_choice_correct"] <= counts["first_committed"] <= counts["junction_arrived"] <= total:
        raise AssertionError("Stage counts inconsistent")
    if counts["first_choice_wrong"]+counts["first_choice_correct"]!=counts["first_committed"]:
        raise AssertionError("Each committed side must be correct or incorrect")
    pair_counts={
        "both_seen":sum(a["cue_seen"] and b["cue_seen"] for a,b in pairs),
        "both_junction":sum(a["junction_arrived"] and b["junction_arrived"] for a,b in pairs),
        "both_committed":sum(a["first_committed"] and b["first_committed"] for a,b in pairs),
        "opposite_first_commits":sum(
            a["first_committed"] and b["first_committed"] and
            a["first_committed_side"] != b["first_committed_side"]
            for a,b in pairs),
        "both_first_correct":sum(
            a["first_choice_correct"] and b["first_choice_correct"] for a,b in pairs),
        "both_final_success":sum(a["success"] and b["success"] for a,b in pairs),
    }
    out={
        "pairs":n,"episodes":total,"counts":counts,
        "rates":{key:counts[key]/total for key in (
            "cue_seen","junction_arrived","first_committed","first_choice_correct",
            "first_choice_wrong","success","timeout","arm_without_center",
            "terminal_without_center","episodes_intervened","clock_4_complete")},
        "pair_counts":pair_counts,
        "conditional": {
            "commit_given_junction":None if not counts["junction_arrived"]
                else counts["first_committed"]/counts["junction_arrived"],
            "correct_given_commit":None if not counts["first_committed"]
                else counts["first_choice_correct"]/counts["first_committed"],
            "denominator_arrived":counts["junction_arrived"],
            "denominator_committed":counts["first_committed"],
        },
        "mean_length":statistics.mean(r["length"] for r in episodes),
        "mean_masked_steps":statistics.mean(r["scheduled_masks"] for r in episodes),
        "mean_effective_masked_steps":statistics.mean(r["effective_masks"] for r in episodes),
        "by_cue":{
            cue:{
                "episodes":len(group),
                "junction_arrived":sum(r["junction_arrived"] for r in group),
                "first_choice_correct":sum(r["first_choice_correct"] for r in group),
                "success":sum(r["success"] for r in group),
            } for cue in ("key","ball")
            for group in ([r for r in episodes if r["cue"]==cue],)
        },
    }
    return out


def run(args):
    if args.index not in range(4) or args.s13_pairs!=40 or args.s7_pairs!=20:
        raise ValueError("F9-N confirmation budget immutable")
    torch.set_num_threads(1)
    source=Path(args.source)
    report=json.loads((source/"report.json").read_text(encoding="utf8"))
    validate_training(report,source,args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("New confirmation directory must start empty")
    result={
        "experiment":"F9-N frozen F9-M stage-resolved independent autonomous confirmation",
        "origin_run":ORIGIN_RUN,
        "seed_index":args.index,
        "new_confirmation_base":SEED_BASE,
        "pairs_by_size":{str(k):v for k,v in PAIRS_BY_SIZE.items()},
        "controls":list(CONDITIONS),
        "training_performed":False,"probes_fitted":False,
        "historical_holdout_opened":False,
        "F9M_confirmation_reused":False,
        "source_training_report_sha256":file_sha(source/"report.json"),
        "checkpoint_sha256":{arm:report["arms"][arm]["checkpoint_sha256"] for arm in ARMS},
        "arms":{},
    }
    for arm in ARMS:
        model=RecurrentPPO.load(str(source/f"{arm}_after_rl.zip"),device="cpu")
        try:
            results={}
            for size, count in PAIRS_BY_SIZE.items():
                for condition in CONDITIONS:
                    pairs=[]
                    for j in range(count):
                        seed=confirmation_seed(args.index,size,j)
                        a=autonomous_episode(model,args.index,size,seed,False,condition)
                        b=autonomous_episode(model,args.index,size,seed,True,condition)
                        pairs.append((a,b))
                    summary=pair_summary(pairs)
                    key=f"S{size}/{condition}"
                    results[key]={"summary":summary,"pairs_data":[list(pair) for pair in pairs]}
                    print("F9N_STAGE",args.index,arm,key,
                          json.dumps(summary),flush=True)
            result["arms"][arm]=results
        finally:
            env=model.get_env()
            if env is not None:
                env.close()
    (out/"report.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print("F9N_CONFIRM_COMPLETE",args.index,flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index",type=int,required=True,choices=range(4))
    p.add_argument("--s13-pairs",type=int,default=40)
    p.add_argument("--s7-pairs",type=int,default=20)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
