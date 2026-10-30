"""F9-S: frozen one-dimensional cue-contrast H/C patches at last S13 step.

Train cue directions on entirely separate 890000 worlds. At the fork, F4 is
always zero. With blackouts, it is also zero from last public cue +1 onward.
No gradients/updates to any policy and no tuning using confirm worlds.
"""
from __future__ import annotations

import numpy as np

from observatory.f9i_selective_memory import SelectiveF4,DIM
from observatory.f9j_recurrent_memory_attribution import frozen_action
from observatory.f9q_causal_fork import matched_pair, clone_state, continue_from_fork
from observatory.f9r_recurrent_state_drift import trace_history

CONDITIONS=(
    "normal_reference",
    "blackout_update",
    "full_last_same",
    "blackout_cue_axis_same",
    "blackout_cue_axis_opposite",
    "blackout_orthogonal_same",
    "full_last_axis_opposite",
    "full_last_orthogonal_opposite",
    "full_last_axis_clamp_mean",
    "full_last_axis_swap_H",
    "full_last_axis_swap_C",
)
MODES=("intact","swap_HC")
CAPACITY=8


def normalize(v):
    a=np.asarray(v,np.float64)
    norm=float(np.linalg.norm(a))
    if a.shape!=(64,) or not np.isfinite(norm) or norm<1.e-10:
        raise AssertionError("Degenerate or nonfinite 64D cue direction")
    return (a/norm).astype(np.float32)


def orthogonal_control(w,seed):
    w=normalize(w)
    rng=np.random.default_rng(int(seed))
    r=rng.standard_normal(64)
    r-=np.dot(r,w)*w
    u=normalize(r)
    if abs(float(np.dot(w,u)))>=1.e-5:
        raise AssertionError("Control direction not orthogonal")
    return u


def validate_axes(axes):
    if set(axes)!={"H","C"}:
        raise AssertionError("Both cue subspaces H,C required")
    for name in ("H","C"):
        axis=axes[name]
        w=np.asarray(axis["cue"],np.float32)
        u=np.asarray(axis["orthogonal"],np.float32)
        if w.shape!=(64,) or u.shape!=(64,) or not np.isfinite(w).all() or not np.isfinite(u).all():
            raise AssertionError("Invalid axis shape/content")
        if abs(np.linalg.norm(w)-1)>1.e-5 or abs(np.linalg.norm(u)-1)>1.e-5:
            raise AssertionError("Axes must be unit")
        if abs(float(np.dot(w,u)))>=1.e-5:
            raise AssertionError("Orthogonal control must not overlap cue axis")
        if axis["paired_train_worlds"]!=80 or axis["fit_method"]!="mean_ball_minus_key_at_last":
            raise AssertionError("Only preregistered axis estimator permitted")
    return True


def patch_axis(target,source,axis,orthogonal=False):
    """Replace one 1D cue coefficient, or add equal-L2 orthogonal placebo."""
    target=np.asarray(target,np.float32).copy()
    source=np.asarray(source,np.float32)
    w=np.asarray(axis["cue"],np.float32)
    u=np.asarray(axis["orthogonal"],np.float32)
    if target.shape!=(1,1,64) or source.shape!=(1,1,64):
        raise AssertionError("Unexpected recurrent state")
    change=float(np.dot((source-target).reshape(-1),w))
    direction=u if orthogonal else w
    result=target+change*direction.reshape(1,1,64)
    if orthogonal and not np.isclose(np.linalg.norm(result-target),abs(change),atol=1e-5):
        raise AssertionError("Magnitude-matched sham violated")
    return np.asarray(result,np.float32)


def state_at_last_blackout(model,index,frames,times):
    """Blackout update until last step, BEFORE it is fed to the recurrent cell."""
    p,f=int(times["post_cue"]),int(times["fork"])
    last=f-1
    if not 0<p<last<f:
        raise AssertionError("No last-step intervention possible")
    writer=SelectiveF4("context",CAPACITY,seed=81000+index)
    state=None
    scheduled=effective=0
    for t in range(last):
        encoded=np.asarray(writer.encode(frames[t]),np.float32)
        if not np.array_equal(encoded[:148],frames[t]):
            raise AssertionError("Encoder touched public input")
        actor=encoded.copy()
        if t>=p:
            scheduled+=1
            actor[148:]=0.
            effective+=int(not np.array_equal(actor,encoded))
        _,state=frozen_action(model,actor,state,first=t==0)
    at_last=np.asarray(writer.encode(frames[last]),np.float32)
    if not np.array_equal(at_last[:148],frames[last]):
        raise AssertionError("Last public frame changed")
    if scheduled!=last-p or effective!=last-p:
        raise AssertionError("Blackout not equal and effective before last")
    return clone_state(state),at_last


def construct_states(model,index,world,axes):
    """Exact same world and frames for 11 causal patch conditions."""
    validate_axes(axes)
    p,f=int(world["times"]["post_cue"]),int(world["times"]["fork"])
    last=f-1
    if not 0<p<last<f:
        raise AssertionError("No valid last decision step")
    natural=[
        trace_history(model,index,frames,world["times"],"normal_reference")
        for frames in world["frames"]]
    if not np.array_equal(natural[0]["raw"],natural[1]["raw"]):
        raise AssertionError("Public fork frame differs across cues")
    at_last=[clone_state(x["natural_snapshots"][last]) for x in natural]
    blackout=[state_at_last_blackout(model,index,frames,world["times"])
              for frames in world["frames"]]
    if not np.array_equal(blackout[0][1][:148],blackout[1][1][:148]):
        raise AssertionError("Last public frame differs across cues")
    result={"normal_reference":[x["state"] for x in natural]}
    audits={}
    for condition in CONDITIONS:
        if condition=="normal_reference":
            audits[condition]={"scheduled":0,"effective":0}
            continue
        states=[]
        for j in range(2):
            orig=at_last[j]
            opposite=at_last[1-j]
            b,last_obs=blackout[j]
            hh,cc=clone_state(b)
            oh,oc=clone_state(orig)
            dh,dc=clone_state(opposite)
            if condition=="blackout_update":
                pass
            elif condition=="full_last_same":
                hh,cc=oh,oc
            elif condition=="blackout_cue_axis_same":
                hh=patch_axis(hh,oh,axes["H"])
                cc=patch_axis(cc,oc,axes["C"])
            elif condition=="blackout_cue_axis_opposite":
                hh=patch_axis(hh,dh,axes["H"])
                cc=patch_axis(cc,dc,axes["C"])
            elif condition=="blackout_orthogonal_same":
                hh=patch_axis(hh,oh,axes["H"],orthogonal=True)
                cc=patch_axis(cc,oc,axes["C"],orthogonal=True)
            elif condition=="full_last_axis_opposite":
                hh=patch_axis(oh,dh,axes["H"])
                cc=patch_axis(oc,dc,axes["C"])
            elif condition=="full_last_orthogonal_opposite":
                hh=patch_axis(oh,dh,axes["H"],orthogonal=True)
                cc=patch_axis(oc,dc,axes["C"],orthogonal=True)
            elif condition=="full_last_axis_clamp_mean":
                hh=patch_axis(oh,(oh+dh)/2,axes["H"])
                cc=patch_axis(oc,(oc+dc)/2,axes["C"])
            elif condition=="full_last_axis_swap_H":
                hh=patch_axis(oh,dh,axes["H"])
                cc=oc
            elif condition=="full_last_axis_swap_C":
                hh=oh
                cc=patch_axis(oc,dc,axes["C"])
            else:
                raise AssertionError("Unpreregistered patch type")
            actor=last_obs.copy()
            actor[148:]=0.
            if not np.array_equal(actor[:148],last_obs[:148]):
                raise AssertionError("Public input touched in patch")
            _,new_state=frozen_action(model,actor,(hh,cc),first=False)
            states.append(clone_state(new_state))
        result[condition]=states
        audits[condition]={"scheduled":f-p,"effective":f-p}
    for j in range(2):
        # The full-last condition reproduces the published F9-R intervention.
        f9r=trace_history(
            model,index,world["frames"][j],world["times"],
            "blackout_patch_normal_last_HC",
            natural_states=natural[j]["natural_snapshots"])
        if not all(np.array_equal(x,y) for x,y in zip(f9r["state"],result["full_last_same"][j])):
            raise AssertionError("F9-S full-last state differs from F9-R instrument")
    return result,audits


def analyze_world(model,index,seed,axes):
    world=matched_pair(model,index,seed)
    states,audit=construct_states(model,index,world,axes)
    conditions={}
    for name in CONDITIONS:
        snapshots=[{"state":s,"raw":np.asarray(world["frames"][j][world["times"]["fork"]][:148])}
                   for j,s in enumerate(states[name])]
        runs={kind:[
            continue_from_fork(model,index,world,receiver,snapshots,kind)
            for receiver in (0,1)]
            for kind in MODES}
        conditions[name]={"runs":runs,"mask_audit":audit[name]}
    base=conditions["full_last_same"]["runs"]["intact"]
    def donor_flip(target):
        return int(all(
            a["committed"] and b["committed"]
            and a["recipient_first_correct"] and b["opposite_donor_first_correct"]
            and a["first_committed_side"]!=b["first_committed_side"]
            for a,b in zip(base,conditions[target]["runs"]["intact"])))
    return {
        "seed":seed,
        "cues":[x["cue"] for x in world["labels"]],
        "correct_sides":[x["correct_side"] for x in world["labels"]],
        "fork":world["times"]["fork"],
        "postcue":world["times"]["post_cue"],
        "fork_public_identical":True,
        "F4_fork_and_continuation_zero":True,
        "conditions":conditions,
        "donor_flip_from_full_last":{
            "cue_axis_opposite":donor_flip("full_last_axis_opposite"),
            "orthogonal_opposite":donor_flip("full_last_orthogonal_opposite"),
            "cue_H_only":donor_flip("full_last_axis_swap_H"),
            "cue_C_only":donor_flip("full_last_axis_swap_C"),
        },
    }


def summarize(cases):
    if not cases:
        raise ValueError("Missing F9-S cases")
    output={"pairs":len(cases),"conditions":{}}
    for cond in CONDITIONS:
        rows=[r["conditions"][cond] for r in cases]
        intact=[r["runs"]["intact"] for r in rows]
        swaps=[r["runs"]["swap_HC"] for r in rows]
        def both_correct(p):
            return int(all(x["recipient_first_correct"] for x in p))
        def donor_causal(a,b):
            return int(all(
                aa["recipient_first_correct"] and bb["opposite_donor_first_correct"]
                and aa["committed"] and bb["committed"]
                and aa["first_committed_side"]!=bb["first_committed_side"]
                for aa,bb in zip(a,b)))
        output["conditions"][cond]={
            "both_first_correct":sum(map(both_correct,intact)),
            "both_swap_donor_correct":sum(int(all(x["opposite_donor_first_correct"] for x in p)) for p in swaps),
            "donor_specific_full_pair":sum(donor_causal(a,b) for a,b in zip(intact,swaps)),
            "first_correct_individual":sum(x["recipient_first_correct"] for p in intact for x in p),
            "first_committed_individual":sum(x["committed"] for p in intact for x in p),
            "both_first_teacher_actions_correct":sum(all(x["first_action_teacher_correct"] for x in p) for p in intact),
            "mean_effective_mask":float(np.mean([x["mask_audit"]["effective"] for x in rows])),
        }
    output["cue_axis_causal_flip_pairs"]=sum(x["donor_flip_from_full_last"]["cue_axis_opposite"] for x in cases)
    output["orthogonal_causal_flip_pairs"]=sum(x["donor_flip_from_full_last"]["orthogonal_opposite"] for x in cases)
    output["H_only_causal_flip_pairs"]=sum(x["donor_flip_from_full_last"]["cue_H_only"] for x in cases)
    output["C_only_causal_flip_pairs"]=sum(x["donor_flip_from_full_last"]["cue_C_only"] for x in cases)
    return output
