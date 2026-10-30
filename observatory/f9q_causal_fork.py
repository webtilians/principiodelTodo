"""F9-Q causal fork memory: teacher-matched history, absent F4, H/C swap.

No policy training, no external readout at the fork or during subsequent
autonomous commitment. Observer-only teacher actions select a *common* fork.
This module never selects the train/confirmation seed split by itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9d_autonomous_navigation import demonstration
from observatory.f9i_selective_memory import SelectiveF4, DIM
from observatory.f9j_recurrent_memory_attribution import frozen_action
from observatory.f9k_temporal_memory_probes import stage_indices
from observatory.f9n_stage_autonomy import StageObserver

CAPACITY=8
REGIMES=("history_f4_normal","history_f4_postcue_off")
INTERVENTIONS=("intact","swap_HC","swap_H","swap_C","zero_HC","zero_H","zero_C")
ZERO_TAIL=np.zeros(DIM-148,dtype=np.float32)


def snapshot(model,index,frames,times,regime):
    if index not in range(4) or regime not in REGIMES:
        raise ValueError("Invalid F9-Q model index/regime")
    writer=SelectiveF4("context",CAPACITY,seed=81000+index)
    state=None
    effective=scheduled=0
    for t in range(times["fork"]+1):
        natural=writer.encode(frames[t])
        if natural.shape!=(DIM,) or not np.allclose(natural[:148],frames[t],rtol=0,atol=0):
            raise AssertionError("Public partial observations were changed")
        if t==times["fork"]:
            if state is None:
                raise AssertionError("No recurrent history before fork")
            h,c=clone_state(state)
            return {
                "state":(h,c),"raw":natural[:148].copy(),
                "natural_f4":natural[148:].copy(),
                "scheduled_prefix_masks":scheduled,
                "effective_prefix_masks":effective,
            }
        actor=natural.copy()
        mask=regime=="history_f4_postcue_off" and t>=times["post_cue"]
        if mask:
            scheduled+=1
            actor[148:]=0.
            effective+=int(not np.array_equal(actor,natural))
        _,state=frozen_action(model,actor,state,first=t==0)
    raise AssertionError("Missing fork snapshot")


def clone_state(state):
    if state is None or len(state)!=2:
        raise AssertionError("Expected LSTM (hidden,cell) state")
    h,c=[np.asarray(x,dtype=np.float32).copy() for x in state]
    if h.shape!=(1,1,64) or c.shape!=(1,1,64) or not (
        np.isfinite(h).all() and np.isfinite(c).all()):
        raise AssertionError("Unexpected LSTM state shape/values")
    return h,c


def intervene(own,opposite,kind):
    """Do not touch model weights or recipient observation/environment."""
    if kind not in INTERVENTIONS:
        raise ValueError(kind)
    h,c=clone_state(own)
    hh,cc=clone_state(opposite)
    if kind=="swap_HC":
        h,c=hh,cc
    elif kind=="swap_H":
        h=hh
    elif kind=="swap_C":
        c=cc
    elif kind=="zero_HC":
        h[:]=0.;c[:]=0.
    elif kind=="zero_H":
        h[:]=0.
    elif kind=="zero_C":
        c[:]=0.
    return h,c


def matched_pair(model,index,seed):
    """Original and inverted cue: same teacher pose/path until first fork."""
    if index not in range(4):
        raise ValueError(index)
    frames=[]
    actions=[]
    vis=[]
    labels=[]
    for reverse in (False,True):
        x,a,v=demonstration(seed,size=13,reverse=reverse)
        frames.append(x)
        actions.append(a)
        vis.append(v)
        env=build_env(13)
        try:
            env.reset(seed=seed)
            if reverse:
                _,lbl=swap_cue(env,get_world_labels(env))
            else:
                lbl=get_world_labels(env)
            labels.append(lbl)
        finally:
            env.close()
    times=stage_indices(*actions,*vis)
    fork=times["fork"]
    if not np.array_equal(actions[0][:fork],actions[1][:fork]):
        raise AssertionError("Teacher prefix action paths differ")
    if fork>=min(len(x) for x in frames):
        raise AssertionError("Fork frame missing")
    if not np.array_equal(frames[0][fork],frames[1][fork]):
        raise AssertionError("Public decision frame differs between key and ball")
    if labels[0]["cue"]==labels[1]["cue"] or labels[0]["correct_side"]==labels[1]["correct_side"]:
        raise AssertionError("Cue/goal intervention invalid")
    for name in ("start_position","start_direction","upper_object","lower_object"):
        if labels[0][name]!=labels[1][name]:
            raise AssertionError("Geometry/start changed under cue inversion: "+name)
    return {
        "frames":frames,"actions":actions,"times":times,"labels":labels,
        "seed":seed,
    }


def history_pair(model,index,pair,regime):
    if regime not in REGIMES:
        raise ValueError(regime)
    a=snapshot(model,index,pair["frames"][0],pair["times"],regime)
    b=snapshot(model,index,pair["frames"][1],pair["times"],regime)
    if not np.array_equal(a["raw"],b["raw"]):
        raise AssertionError("Fork observation not exactly matched")
    return (a,b)


def replay_to_decision(seed,reverse,prefix,expected_public):
    """Reconstruct recipient state at exactly the teacher's first divergence."""
    env=build_env(13)
    raw,_=env.reset(seed=seed)
    if reverse:
        raw,_=swap_cue(env,get_world_labels(env))
    labels=get_world_labels(env)
    for action in prefix:
        raw,_,terminated,truncated,_=env.step(int(action))
        if terminated or truncated:
            env.close()
            raise AssertionError("Teacher prefix terminated before fork")
    if not np.array_equal(raw,expected_public):
        env.close()
        raise AssertionError("Replayed recipient public fork view changed")
    return env,raw,labels


def continue_from_fork(model,index,pair,receiver,donor_states,kind):
    """Actor completely autonomous AFTER matched teacher prefix, with F4=0.

    The teacher is NEVER used once the fork starts; only raw public input
    plus the intervention-defined H/C state reaches model.predict.
    """
    own=donor_states[receiver]["state"]
    other=donor_states[1-receiver]["state"]
    state=intervene(own,other,kind)
    prefix=pair["actions"][receiver][:pair["times"]["fork"]]
    env,raw,labels=replay_to_decision(pair["seed"],bool(receiver),prefix,
                                     donor_states[receiver]["raw"])
    observer=StageObserver(labels,13)
    observer.observe(env.unwrapped.agent_pos,0)
    first_action=None
    steps=0
    timed_out=False
    terminated=False
    try:
        while True:
            public=np.asarray(raw,dtype=np.float32)
            observation=np.concatenate((public,ZERO_TAIL)).astype(np.float32)
            if observation.shape!=(DIM,) or np.any(observation[148:]):
                raise AssertionError("F4 not suppressed at decision/continuation")
            action,state=frozen_action(model,observation,state,first=False)
            if first_action is None:
                first_action=int(action)
            raw,_,end,truncated,_=env.step(action)
            steps+=1
            observer.observe(env.unwrapped.agent_pos,steps)
            if observer.first_commit_step is not None:
                break
            if end or truncated:
                timed_out=bool(truncated)
                terminated=bool(end)
                break
        summary=observer.summary()
        target=labels["correct_side"]
        opposite_target=pair["labels"][1-receiver]["correct_side"]
        return {
            "receiver":receiver,"kind":kind,
            "first_action":first_action,
            "teacher_first_action":int(pair["actions"][receiver][pair["times"]["fork"]]),
            "first_action_teacher_correct":int(
                first_action==int(pair["actions"][receiver][pair["times"]["fork"]])),
            "first_committed_side":summary["first_committed_side"],
            "committed":summary["first_committed"],
            "recipient_correct_side":target,
            "opposite_donor_correct_side":opposite_target,
            "recipient_first_correct":int(summary["first_committed"] and
                                           summary["first_committed_side"]==target),
            "opposite_donor_first_correct":int(summary["first_committed"] and
                                              summary["first_committed_side"]==opposite_target),
            "steps_until_commit_or_end":steps,
            "timeout":timed_out,
            "terminated_without_commit":terminated,
            "F4_zero_throughout_decision":True,
        }
    finally:
        env.close()


def analyze_pair(model,index,seed,regime,probes=None,include_all=True):
    """Fixed full-factorial H/C interventions, no cherry-picked situations."""
    pair=matched_pair(model,index,seed)
    states=history_pair(model,index,pair,regime)
    h_l2=float(np.linalg.norm(states[0]["state"][0]-states[1]["state"][0]))
    c_l2=float(np.linalg.norm(states[0]["state"][1]-states[1]["state"][1]))
    if pair["labels"][0]["cue"]==pair["labels"][1]["cue"]:
        raise AssertionError("Identical cue labels")
    if not np.array_equal(states[0]["raw"],states[1]["raw"]):
        raise AssertionError("Natural fork public mismatch")
    result={
        "seed":seed,
        "cue":[x["cue"] for x in pair["labels"]],
        "correct_sides":[x["correct_side"] for x in pair["labels"]],
        "fork_step":pair["times"]["fork"],
        "postcue":pair["times"]["post_cue"],
        "history_mask_steps":[x["scheduled_prefix_masks"] for x in states],
        "history_effective_mask_steps":[x["effective_prefix_masks"] for x in states],
        "fork_public_equal":True,"fork_F4_actor_zero":True,
        "H_distance":h_l2,"C_distance":c_l2,
        "runs":{},
    }
    if probes is not None:
        result["probe"]={}
        for feature,component in (("H",0),("C",1)):
            probe=probes[feature]
            center=np.asarray(probe["center"],np.float32)
            scale=np.asarray(probe["scale"],np.float32)
            weight=np.asarray(probe["weight"],np.float32)
            x=np.stack([state["state"][component].reshape(-1) for state in states])
            predicted=np.where(((x-center)/scale)@weight>=0.,1,-1)
            true=np.array([1 if name=="ball" else -1 for name in result["cue"]])
            result["probe"][feature]={
                "both_correct":int(np.array_equal(predicted,true)),
                "individually_correct":[int(a==b) for a,b in zip(predicted,true)],
                "paired_flip":int(predicted[0]!=predicted[1]),
            }
    if include_all:
        for kind in INTERVENTIONS:
            result["runs"][kind]=[
                continue_from_fork(model,index,pair,recipient,states,kind)
                for recipient in (0,1)]
    return result


def summarize(cases):
    n=len(cases)
    if not n:
        raise ValueError("No cases")
    for c in cases:
        if c["cue"][0]==c["cue"][1] or c["correct_sides"][0]==c["correct_sides"][1]:
            raise AssertionError("Counterfactual pairing invalid")
        if not c["fork_public_equal"] or not c["fork_F4_actor_zero"]:
            raise AssertionError("Fork not controlled")
    out={
        "pairs":n,
        "mean_H_distance":float(np.mean([c["H_distance"] for c in cases])),
        "mean_C_distance":float(np.mean([c["C_distance"] for c in cases])),
        "avg_prefix_effective_F4_masks":float(np.mean([
            x for c in cases for x in c["history_effective_mask_steps"]])),
        "interventions":{},
    }
    if all("probe" in c for c in cases):
        out["probes"]={
            feature:{
                "both_decoded":sum(c["probe"][feature]["both_correct"] for c in cases),
                "paired_prediction_flips":sum(
                    c["probe"][feature]["paired_flip"] for c in cases),
                "individual_correct":sum(sum(
                    c["probe"][feature]["individually_correct"]) for c in cases),
            } for feature in ("H","C")}
    if all("runs" in c and c["runs"] for c in cases):
        for kind in INTERVENTIONS:
            runs=[c["runs"][kind] for c in cases]
            flat=[r for pair in runs for r in pair]
            out["interventions"][kind]={
                "both_teacher_first_actions_correct":sum(all(
                    row["first_action_teacher_correct"] for row in pair) for pair in runs),
                "both_first_arms_recipient_correct":sum(all(
                    row["recipient_first_correct"] for row in pair) for pair in runs),
                "both_first_arms_donor_correct":sum(all(
                    row["opposite_donor_first_correct"] for row in pair) for pair in runs),
                "first_action_recipient_correct":sum(
                    r["first_action_teacher_correct"] for r in flat),
                "first_arm_recipient_correct":sum(
                    r["recipient_first_correct"] for r in flat),
                "first_arm_donor_correct":sum(
                    r["opposite_donor_first_correct"] for r in flat),
                "committed":sum(r["committed"] for r in flat),
                "timeout":sum(r["timeout"] for r in flat),
                "both_committed":sum(all(r["committed"] for r in pair) for pair in runs),
            }
        out["transplant"]={
            "both_actions_changed":sum(all(
                a["first_action"]!=b["first_action"]
                for a,b in zip(c["runs"]["intact"],c["runs"]["swap_HC"]))
                for c in cases),
            "both_first_arms_flipped":sum(all(
                a["committed"] and b["committed"] and
                a["first_committed_side"]!=b["first_committed_side"]
                for a,b in zip(c["runs"]["intact"],c["runs"]["swap_HC"]))
                for c in cases),
            "causal_donor_specific_full_pair":sum(all(
                a["recipient_first_correct"] and b["opposite_donor_first_correct"] and
                a["first_committed_side"]!=b["first_committed_side"]
                for a,b in zip(c["runs"]["intact"],c["runs"]["swap_HC"]))
                for c in cases),
        }
    return out
