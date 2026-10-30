"""F9-R: same teacher trajectory and readout F4=0 at fork, controlled H/C updates.

This is a *causal ablation of recurrent updates* at S13 matched fork, not policy
training. All actor weights are frozen. Ten prespecified temporal interventions
use identical public history, simulator, reward, teacher action prefix, and
external-F4 masked input from last-cue+1 until action at the fork.
"""
from __future__ import annotations

import numpy as np

from observatory.f9i_selective_memory import SelectiveF4, DIM
from observatory.f9j_recurrent_memory_attribution import frozen_action
from observatory.f9q_causal_fork import matched_pair, clone_state, continue_from_fork

CAPACITY=8
CONDITIONS=(
    "normal_reference",
    "blackout_update",
    "blackout_freeze_HC",
    "blackout_freeze_H",
    "blackout_freeze_C",
    "blackout_patch_normal_mid_HC",
    "blackout_patch_normal_last_HC",
    "blackout_patch_normal_fork_HC",
    "blackout_patch_normal_fork_H",
    "blackout_patch_normal_fork_C",
)
SWAPS=("intact","swap_HC")


def trace_history(model,index,frames,times,condition,natural_states=None):
    """Return state just BEFORE first divergent teacher action; never act as teacher.

    For frozen states, model.predict is executed, but its updated H/C state is
    partially/totally discarded during blackout, exactly as preregistered.
    The F4 encoder sees all raw partial observations in every condition.
    """
    if index not in range(4) or condition not in CONDITIONS:
        raise ValueError("Unexpected F9-R intervention/index")
    p=int(times["post_cue"])
    f=int(times["fork"])
    m=p+(f-p)//2
    last=f-1
    if not 0<p<m<last<f or f>=len(frames):
        raise AssertionError("Cannot preregister midpoint/last patch times")
    writer=SelectiveF4("context",CAPACITY,seed=81000+index)
    state=None
    snapshots={}
    planned=effective=0
    for t in range(f+1):
        natural=np.asarray(writer.encode(frames[t]),np.float32)
        if natural.shape!=(DIM,) or not np.array_equal(natural[:148],frames[t]):
            raise AssertionError("Public observation changed by writer")
        if state is not None:
            snapshots[t]=clone_state(state)
        if t==f:
            if state is None:
                raise AssertionError("Missing fork recurrent state")
            if condition.startswith("blackout_patch_normal_fork_"):
                if natural_states is None or f not in natural_states:
                    raise AssertionError("Frozen natural-state control missing")
                natural_h,natural_c=clone_state(natural_states[f])
                old_h,old_c=clone_state(state)
                if condition.endswith("_HC"):
                    state=(natural_h,natural_c)
                elif condition.endswith("_H"):
                    state=(natural_h,old_c)
                else:
                    state=(old_h,natural_c)
            return {
                "state":clone_state(state),
                "raw":natural[:148].copy(),
                "natural_f4":natural[148:].copy(),
                "scheduled_prefix_masks":planned,
                "effective_prefix_masks":effective,
                "natural_snapshots":snapshots if condition=="normal_reference" else None,
                "fork":f,"postcue":p,"mid":m,"last":last,
            }
        actor=natural.copy()
        masked=condition!="normal_reference" and t>=p
        if masked:
            planned+=1
            actor[148:]=0.
            effective+=int(not np.array_equal(actor,natural))
        if not np.array_equal(actor[:148],natural[:148]):
            raise AssertionError("Mask affected raw public feature")
        if condition in ("blackout_patch_normal_mid_HC","blackout_patch_normal_last_HC"):
            patch_t=m if condition=="blackout_patch_normal_mid_HC" else last
            if t==patch_t:
                if natural_states is None or t not in natural_states:
                    raise AssertionError("Missing same-time unmasked state")
                state=clone_state(natural_states[t])
        _,computed=frozen_action(model,actor,state,first=(t==0))
        if masked and condition.startswith("blackout_freeze_"):
            if state is None:
                raise AssertionError("Freeze starts after valid cue")
            old_h,old_c=clone_state(state)
            new_h,new_c=clone_state(computed)
            if condition=="blackout_freeze_HC":
                state=(old_h,old_c)
            elif condition=="blackout_freeze_H":
                state=(old_h,new_c)
            elif condition=="blackout_freeze_C":
                state=(new_h,old_c)
            else:
                raise AssertionError("Undeclared freeze condition")
        else:
            state=clone_state(computed)
    raise AssertionError("No fork snapshot")


def prepare_matched_conditions(model,index,world):
    """Full factorial state pairs, with natural checkpoint used as fixed donor."""
    p,f=world["times"]["post_cue"],world["times"]["fork"]
    records={}
    naturally=[
        trace_history(model,index,frames,world["times"],"normal_reference")
        for frames in world["frames"]]
    if not np.array_equal(naturally[0]["raw"],naturally[1]["raw"]):
        raise AssertionError("Original/inverted public observations differ at fork")
    for condition in CONDITIONS:
        states=(naturally if condition=="normal_reference" else [
            trace_history(model,index,world["frames"][j],world["times"],
                          condition,natural_states=naturally[j]["natural_snapshots"])
            for j in range(2)])
        for snapshot in states:
            if condition=="normal_reference":
                if snapshot["scheduled_prefix_masks"] or snapshot["effective_prefix_masks"]:
                    raise AssertionError("Normal history unexpectedly masked")
            elif (snapshot["scheduled_prefix_masks"]!=f-p or
                  snapshot["effective_prefix_masks"]!=f-p):
                raise AssertionError("Blackout not exactly equal across conditions")
            if snapshot["fork"]!=f or snapshot["postcue"]!=p:
                raise AssertionError("Missing matched-history audit")
        if not np.array_equal(states[0]["raw"],states[1]["raw"]):
            raise AssertionError("Cue changes raw view at fork")
        records[condition]=states
    for j in (0,1):
        ref=records["normal_reference"][j]["state"]
        intervention=records["blackout_patch_normal_fork_HC"][j]["state"]
        if not all(np.array_equal(a,b) for a,b in zip(ref,intervention)):
            raise AssertionError("Positive control did not exactly restore H/C")
    return records


def first_choice_metrics(model,index,world,conditions):
    """Self vs transplanted H/C, continued autonomously only FROM the fork."""
    per_condition={}
    for condition in CONDITIONS:
        states=conditions[condition]
        runs={}
        for mode in SWAPS:
            runs[mode]=[
                continue_from_fork(model,index,world,receiver,states,mode)
                for receiver in (0,1)]
        per_condition[condition]={
            "runs":runs,
            "H_pair_l2":float(np.linalg.norm(
                states[0]["state"][0]-states[1]["state"][0])),
            "C_pair_l2":float(np.linalg.norm(
                states[0]["state"][1]-states[1]["state"][1])),
            "scheduled_masks":states[0]["scheduled_prefix_masks"],
            "effective_masks":states[0]["effective_prefix_masks"],
        }
    for mode in SWAPS:
        a=per_condition["normal_reference"]["runs"][mode]
        b=per_condition["blackout_patch_normal_fork_HC"]["runs"][mode]
        for x,y in zip(a,b):
            if x["first_action"]!=y["first_action"] or x["first_committed_side"]!=y["first_committed_side"]:
                raise AssertionError("Normal-fork H/C replay positive control differed")
    return per_condition


def analyze_world(model,index,seed):
    world=matched_pair(model,index,seed)
    states=prepare_matched_conditions(model,index,world)
    cases=first_choice_metrics(model,index,world,states)
    return {
        "seed":seed,
        "cue":[v["cue"] for v in world["labels"]],
        "correct_sides":[v["correct_side"] for v in world["labels"]],
        "fork":world["times"]["fork"],
        "postcue":world["times"]["post_cue"],
        "fork_public_identical":True,
        "F4_at_fork_and_continuation_all_zero":True,
        "conditions":cases,
    }


def summarize_worlds(records):
    if not records:
        raise ValueError("No F9-R records to report")
    output={"pairs":len(records),"conditions":{}}
    for condition in CONDITIONS:
        summaries=[r["conditions"][condition] for r in records]
        intact=[row["runs"]["intact"] for row in summaries]
        swapped=[row["runs"]["swap_HC"] for row in summaries]
        all_intact=[v for pair in intact for v in pair]
        all_swapped=[v for pair in swapped for v in pair]
        donor_transfer=sum(all(
            original["recipient_first_correct"] and transplant["opposite_donor_first_correct"]
            and original["first_committed_side"]!=transplant["first_committed_side"]
            for original,transplant in zip(a,b))
            for a,b in zip(intact,swapped))
        output["conditions"][condition]={
            "pairs":len(records),
            "both_intact_first_arms_correct":sum(all(
                x["recipient_first_correct"] for x in pair) for pair in intact),
            "both_swap_HC_donor_arms_correct":sum(all(
                x["opposite_donor_first_correct"] for x in pair) for pair in swapped),
            "both_intact_teacher_first_actions_correct":sum(all(
                x["first_action_teacher_correct"] for x in pair) for pair in intact),
            "both_swap_changes_first_arms":sum(all(
                a["first_committed_side"]!=b["first_committed_side"]
                and a["committed"] and b["committed"]
                for a,b in zip(first,second))
                for first,second in zip(intact,swapped)),
            "donor_specific_causal_full_pair":donor_transfer,
            "first_intact_choice_correct_episodes":sum(
                x["recipient_first_correct"] for x in all_intact),
            "first_commit_episodes":sum(x["committed"] for x in all_intact),
            "timeouts":sum(x["timeout"] for x in all_intact),
            "mean_H_pair_l2":float(np.mean([s["H_pair_l2"] for s in summaries])),
            "mean_C_pair_l2":float(np.mean([s["C_pair_l2"] for s in summaries])),
            "mean_effective_prefix_masks":float(np.mean([s["effective_masks"] for s in summaries])),
            "mean_scheduled_prefix_masks":float(np.mean([s["scheduled_masks"] for s in summaries])),
        }
    return output
