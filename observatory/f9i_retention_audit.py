"""F9-I: audit forgotten cue events and counterfactual one-record corruptions.

NO retraining. Replays original F9-H trajectories against the same seven-action
MiniGrid environment and validates F4 readout hashes before measuring retention.
Runs controlled *frozen-checkpoint* policy tests separately from replay.

Training-policy input: only 148 public egocentric observation features plus
Event/FIFO/Store-derived memory. The cue oracle (paired shadow world) is used
ONLY by evaluation to mark/intervene on one exact stored cue event. It NEVER
selects memory events for the normal/anchored agent.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, OBJECT_TO_IDX

from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9d_autonomous_navigation import checked_seed
from observatory.f9g_context_memory import CAPACITY, DIM, FIELDS, ContextF4
from observatory.selective_memory import Event, FIFO


class PreserveFirstFIFO(FIFO):
    """Same capacity/Store/Event layout as FIFO, preserve first 16 events.

    Deterministic and observation-only. The policy never knows which objects
    are cues. Evicts oldest unprotected event on overflow; no hidden labels.
    """
    def __init__(self, capacity=CAPACITY, protected=16):
        super().__init__(capacity, seed=0)
        if not 0 < protected < capacity:
            raise ValueError("Require 0 < protected < memory capacity")
        self.protected=protected

    def consider(self, store, event):
        if event.key in store.saved:
            store.put(event)
        elif len(store.saved)<store.capacity:
            store.put(event)
        else:
            eligible=[k for k in store.order if k>=self.protected]
            if not eligible:
                raise AssertionError("No unprotected event available")
            store.put(event, eligible[0])


def writer_for(selection):
    if selection not in ("fifo","first16"):
        raise ValueError(selection)
    writer=ContextF4()
    if selection=="first16":
        writer.selector=PreserveFirstFIFO(CAPACITY,16)
        writer.reset()
    return writer


def visible_cue_event_ids(raw, shadow_raw, before_serial):
    """Observer marks writes corresponding to a changed key/ball pixel.

    The agent never receives this function's output: only the replay/auditor
    or an explicitly labeled frozen-checkpoint intervention uses the keys.
    """
    raw=np.asarray(raw,dtype=np.float32)
    other=np.asarray(shadow_raw,dtype=np.float32)
    changed=np.any(raw[:147].reshape(7,7,3)!=
                   other[:147].reshape(7,7,3),axis=-1)
    frame=raw[:147].reshape(7,7,3)
    locations=[]
    for x in range(7):
        for y in range(7):
            obj=int(np.rint(frame[x,y,0]*max(OBJECT_TO_IDX.values())))
            color=int(np.rint(frame[x,y,1]*max(COLOR_TO_IDX.values())))
            if obj in (OBJECT_TO_IDX["key"],OBJECT_TO_IDX["ball"]) and 0<=color<6:
                locations.append((x,y))
    return {before_serial+i for i,(x,y) in enumerate(locations) if changed[x,y]},len(locations)


def coupled_worlds(size, seed, reverse):
    normal,shadow=build_env(size),build_env(size)
    try:
        raw,_=normal.reset(seed=seed)
        shadow.reset(seed=seed)
        if reverse:
            raw,_=swap_cue(normal,get_world_labels(normal))
        else:
            swap_cue(shadow,get_world_labels(shadow))
        yield normal,shadow,raw
    finally:
        normal.close()
        shadow.close()


def audit_single_trace(episode, size):
    """Exact checkpoint-era replay; asserts original input readout hash."""
    import contextlib
    from observatory.f9h_context_memory_navigation import paired_rollout
    seed,reverse=int(episode["seed"]),bool(episode["reverse"])
    if episode["length"]!=len(episode["trace"]):
        raise AssertionError("Source report length mismatch")
    with contextlib.contextmanager(coupled_worlds)(size,seed,reverse) as (env,shadow,raw):
        fifo=writer_for("fifo")
        anchor=writer_for("first16")
        all_cue_ids=set()
        first_cue_id=None
        first_cue_step=None
        first_evicted_at=None
        ever_fifo_full=False
        last_result=None
        for step,logged in enumerate(episode["trace"]):
            pos=list(map(int,env.unwrapped.agent_pos))
            if pos!=logged["position"]:
                raise AssertionError("Replay action/position diverged from frozen report")
            other=shadow.observation(shadow.unwrapped.gen_obs())
            cue_ids,new_count=visible_cue_event_ids(raw,other,fifo.serial)
            before=fifo.serial
            encoded=fifo.encode(raw)
            anchor_encoded=anchor.encode(raw)
            if fifo.serial-before!=new_count:
                raise AssertionError("F9-G event ordering changed")
            if anchor.serial!=fifo.serial:
                raise AssertionError("Admission strategy changed event-generation")
            if hashlib.sha256(encoded[148:].tobytes()).hexdigest()[:16]!=logged["f4_readout_hash16"]:
                raise AssertionError("F9-H exact readout SHA mismatch; source not replayable")
            if logged["f4_events_seen"]!=fifo.serial:
                raise AssertionError("F9-H event-count replay mismatch")
            if logged["f4_occupied"]!=len(fifo.store.order):
                raise AssertionError("F9-H FIFO occupancy replay mismatch")
            if cue_ids and first_cue_id is None:
                first_cue_id=min(cue_ids)
                first_cue_step=step
            all_cue_ids.update(cue_ids)
            if first_cue_id is not None and first_cue_id not in fifo.store.saved and first_evicted_at is None:
                first_evicted_at=step
            ever_fifo_full|=len(fifo.store.saved)==CAPACITY
            last_result={
                "last_action_step":step,
                "first_cue_visible_step":first_cue_step,
                "first_cue_serial":first_cue_id,
                "first_cue_evicted_at":first_evicted_at,
                "all_cue_event_ids":len(all_cue_ids),
                "final_fifo_cue_slots":len(all_cue_ids.intersection(fifo.store.saved)),
                "final_first16_cue_slots":len(all_cue_ids.intersection(anchor.store.saved)),
                "fifo_retains_first_cue":bool(first_cue_id is not None and first_cue_id in fifo.store.saved),
                "first16_retains_first_cue":bool(first_cue_id is not None and first_cue_id in anchor.store.saved),
                "fifo_retains_any_cue":bool(all_cue_ids.intersection(fifo.store.saved)),
                "first16_retains_any_cue":bool(all_cue_ids.intersection(anchor.store.saved)),
                "fifo_memory_full":ever_fifo_full,
                "final_event_serial":fifo.serial,
                "final_fifo_occupancy":len(fifo.store.saved),
                "final_first16_occupancy":len(anchor.store.saved),
            }
            action=int(logged["action"])
            if not 0<=action<7:
                raise AssertionError("Non-native action in source trajectory")
            raw,reward,done,truncated,_=env.step(action)
            shadow.step(action)
            if done or truncated:
                if step!=len(episode["trace"])-1:
                    raise AssertionError("Frozen trace continues after terminal action")
                break
        if last_result is None:
            raise AssertionError("Empty trace")
        return dict(
            **last_result,
            seed=seed,reverse=reverse,size=size,
            cue=episode["cue"],
            chosen_side=episode["chosen_side"],
            correct_side=episode["correct_side"],
            success=bool(episode["success"]),
            timeout=bool(episode["timeout"]),
        )


def load_frozen_seed(path):
    path=Path(path)
    report=json.loads((path/"report.json").read_text())
    if report.get("experiment")!="F9-H contextual F4 versus equal-width zero native navigation":
        raise AssertionError("Wrong original experiment")
    if report["heldout_opened"] is not False:
        raise AssertionError("Original holdout not sealed")
    models=report["models"]
    if set(models)!={"zero","f4"}:
        raise AssertionError("Original comparison missing")
    if models["f4"]["parameters"]!=models["zero"]["parameters"]:
        raise AssertionError("Unmatched source agent parameters")
    for mode in ("f4","zero"):
        for stage in ("before_rl","after_rl"):
            checkpoint=path/f"{mode}_{stage}.zip"
            if not checkpoint.exists():
                raise FileNotFoundError(checkpoint)
            sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if sha!=models[mode][f"{stage}_sha256"]:
                raise AssertionError("Frozen source policy SHA mismatch")
    records=[json.loads(line) for line in
             (path/"trajectories.jsonl").read_text().splitlines()]
    return report,records


def replay_original_seed(path, mode="f4"):
    report,records=load_frozen_seed(path)
    index=report["seed_index"]
    results={}
    count=report["config"]["paired_eval_worlds"]
    for size in (7,13):
        candidates=[r for r in records
                    if r["model"]==mode and r["environment"]==f"S{size}"
                    and r["intervention"]=="normal" and r["training_index"]==index]
        if len(candidates)!=2*count:
            raise AssertionError("F9-H before- and after-RL normal trace stage sizes changed")
        candidates=candidates[count:]   # frozen post-RL trajectories
        rows=[]
        for pair in candidates:
            first,second=pair["paired_rollouts"]
            if first["correct_side"]==second["correct_side"]:
                raise AssertionError("Invalid cue-inversion pair")
            rows.extend([audit_single_trace(first,size),audit_single_trace(second,size)])
        results[f"S{size}"]={
            "episodes":len(rows),
            "independent_training_seed":index,
            "cue_observed":sum(x["first_cue_serial"] is not None for x in rows),
            "first_cue_retained_fifo_at_last_action":sum(x["fifo_retains_first_cue"] for x in rows),
            "first_cue_retained_first16_at_last_action":sum(x["first16_retains_first_cue"] for x in rows),
            "any_cue_retained_fifo_at_last_action":sum(x["fifo_retains_any_cue"] for x in rows),
            "any_cue_retained_first16_at_last_action":sum(x["first16_retains_any_cue"] for x in rows),
            "full_fifo_reached":sum(x["fifo_memory_full"] for x in rows),
            "success_original_model":sum(x["success"] for x in rows),
            "timeouts":sum(x["timeout"] for x in rows),
            "episodes":rows,
        }
        print("F9I_REPLAY",index,size,json.dumps({
            k:v for k,v in results[f"S{size}"].items() if k!="episodes"
        }),flush=True)
    return {
        "seed_index":index,
        "original_training_steps":{k:models["actual_rl_steps"] for k,models in report["models"].items()},
        "source_checkpoint_verified":True,
        "by_size":results,
    }


def alter_one_slot(encoded,writer,event_id):
    """Swap ONLY key/ball channels of one event in readout, not raw frame."""
    if event_id is None or event_id not in writer.store.saved:
        return encoded,False
    position=writer.store.order.index(event_id)
    start=148+position*FIELDS
    altered=encoded.copy()
    altered[start],altered[start+1]=altered[start+1],altered[start]
    np.testing.assert_array_equal(altered[:148],encoded[:148])
    if sum(encoded[148:]!=altered[148:])!=2:
        raise AssertionError("Intervention modified more than one event")
    return altered,True


def frozen_checkpoint_rollout(model,size,seed,reverse,selection,intervention):
    """Frozen policy, optional one-event observationally localized perturbation.

    Important: an observer ONLY at evaluation identifies the initial cue
    location by paired cue swap. The normal selector (FIFO/first16)
    cannot see which event is marked, nor does training use the marker.
    """
    import contextlib
    if intervention not in ("normal","cue_one","noncue_one"):
        raise ValueError(intervention)
    with contextlib.contextmanager(coupled_worlds)(size,seed,reverse) as (env,shadow,raw):
        writer=writer_for(selection)
        seen_cue_ids=set()
        first_cue_id=None
        first_noncue_id=None
        interventions_applied=0
        state=None
        first=True
        total=0.
        count=0
        while True:
            other=shadow.observation(shadow.unwrapped.gen_obs())
            cue_ids,n_new=visible_cue_event_ids(raw,other,writer.serial)
            start=writer.serial
            encoded=writer.encode(raw)
            all_new=set(range(start,writer.serial))
            if writer.serial-start != n_new:
                raise AssertionError("Writer and audit event count out of sync")
            if cue_ids and first_cue_id is None:
                first_cue_id=min(cue_ids)
            if first_noncue_id is None and all_new-cue_ids:
                first_noncue_id=min(all_new-cue_ids)
            seen_cue_ids.update(cue_ids)
            target=(first_cue_id if intervention=="cue_one" else
                    first_noncue_id if intervention=="noncue_one" else None)
            if intervention!="normal":
                encoded,changed=alter_one_slot(encoded,writer,target)
                interventions_applied+=int(changed)
            action,state=model.predict(
                encoded.copy(),state=state,
                episode_start=np.array([first]),deterministic=True)
            action=int(action)
            assert 0<=action<7
            raw,reward,done,truncated,_=env.step(action)
            shadow.step(action)
            total+=float(reward)
            first=False
            count+=1
            if done or truncated:
                break
        labels=get_world_labels(env)
        end=tuple(int(x) for x in env.unwrapped.agent_pos)
        if end==tuple(labels["success_position"]):
            chosen=labels["correct_side"]
        elif end==tuple(labels["failure_position"]):
            chosen="lower" if labels["correct_side"]=="upper" else "upper"
        else:
            chosen="none"
        return {
            "seed":seed,"size":size,"reverse":reverse,
            "selection":selection,"intervention":intervention,
            "success":int(total>0),"chosen_side":chosen,
            "cue_seen":bool(seen_cue_ids),
            "first_cue_in_memory_at_end":bool(first_cue_id is not None and first_cue_id in writer.store.saved),
            "any_cue_in_memory_at_end":bool(seen_cue_ids.intersection(writer.store.saved)),
            "intervention_steps":interventions_applied,
            "timeout":bool(truncated),
            "length":count,
        }


def run_frozen_inference(path, pairs=10):
    from sb3_contrib import RecurrentPPO
    report,_=load_frozen_seed(path)
    index=report["seed_index"]
    model=RecurrentPPO.load(str(Path(path)/"f4_after_rl.zip"),device="cpu")
    results={}
    for size in (7,13):
        for selection,intervention in (
            ("fifo","normal"),
            ("fifo","cue_one"),
            ("fifo","noncue_one"),
            ("first16","normal"),
            ("first16","cue_one"),
        ):
            rows=[]
            for j in range(pairs):
                seed=checked_seed("development",index,j)
                a=frozen_checkpoint_rollout(model,size,seed,False,selection,intervention)
                b=frozen_checkpoint_rollout(model,size,seed,True,selection,intervention)
                rows.append([a,b])
            flat=[x for pair in rows for x in pair]
            key=f"S{size}/{selection}/{intervention}"
            results[key]={
                "paired_worlds":pairs,
                "combined_success":statistics.mean(x["success"] for x in flat),
                "both_correct":sum(a["success"] and b["success"] for a,b in rows),
                "valid_flips":sum(a["chosen_side"]!=b["chosen_side"]
                                  and a["chosen_side"]!="none" and b["chosen_side"]!="none"
                                  for a,b in rows),
                "cue_seen":statistics.mean(x["cue_seen"] for x in flat),
                "memory_intervention_applied_episode_rate":statistics.mean(
                    x["intervention_steps"]>0 for x in flat),
                "first_cue_retained_at_final_action":statistics.mean(
                    x["first_cue_in_memory_at_end"] for x in flat),
                "timeout_rate":statistics.mean(x["timeout"] for x in flat),
                "mean_length":statistics.mean(x["length"] for x in flat),
                "episodes":rows,
            }
            print("F9I_FROZEN",index,key,json.dumps({
                k:v for k,v in results[key].items() if k!="episodes"
            }),flush=True)
    return {
        "seed_index":index,
        "training_performed":False,
        "heldout_opened":False,
        "caveat":"Frozen FIFO-trained policy applied to alternate first16 selection is distribution shift; never a fair trained first16 competitor",
        "by_condition":results,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--frozen-inference",action="store_true")
    p.add_argument("--pairs",type=int,default=10)
    args=p.parse_args()
    if not 1<=args.pairs<=10:
        p.error("Development-only up to 10 paired worlds per training seed")
    replay=replay_original_seed(args.source)
    output={
        "experiment":"F9-I forensic cue retention and one-event causal corruption",
        "source_run":40925227783,
        "source":"four frozen F9-H agent checkpoints / development traces",
        "training_performed":False,
        "heldout_opened":False,
        "capacity":CAPACITY,
        "retention_comparison":"real FIFO vs 16-event first-occurrence anchor; same capacity",
        "f4_scope":"Contextual Event/Store/FIFO; not LearnedRelevance/F5",
        "replay":replay,
    }
    if args.frozen_inference:
        output["frozen_inference"]=run_frozen_inference(args.source,args.pairs)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"report.json").write_text(json.dumps(output,indent=2))
    print("F9I_FINISHED",replay["seed_index"],flush=True)


if __name__=="__main__":
    main()
