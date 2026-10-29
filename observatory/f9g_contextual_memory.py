"""F9-G: temporally contextual visible-event memory; GATE BEFORE training.

Extends F4 Event/FIFO/Store with egocentric spatial object snapshots and
observation-relative age. All inputs are PUBLIC 148-dimensional MiniGrid
partial observations. Never consults simulator positions, targets, cue labels
or other privileged metadata in the encoder.

This is an exploratory deterministic contextual-retention policy. It is NOT
F4 LearnedRelevance, F5, or proof that a learned policy uses the memory.

Three retained slots have the same fixed input width in contextual/zero
controls: first distinctive view, second distinctive view and cumulative
spatial key/ball appearance counts across the episode. First two are pinned.
The third captures differences that become visible only later.
This explicitly changes the FIFO *admission* policy, but retains real F4
Event/Store/FIFO as the data structure. No oracle decides which is the cue.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX

from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue
from observatory.f9d_autonomous_navigation import checked_seed, demonstration
from observatory.selective_memory import Event, FIFO

SLOTS = 3
PIXELS = 49
TYPES = 2
SLOT_WIDTH = PIXELS * TYPES + 2  # 98 visible spatial bits, age, occupied
READOUT_WIDTH = SLOTS * SLOT_WIDTH
OBS_WIDTH = 148
TOTAL_WIDTH = OBS_WIDTH + READOUT_WIDTH
MODES = ("context", "zero", "erase", "swap_identity")


def visible_spatial_bits(observation):
    """Pack actual egocentric object pixels into a 98-bit payload.

    MiniGrid symbolic observation: 7x7 object code, color, state + compass.
    This generic transform does not know which visible object is a cue.
    """
    raw = np.asarray(observation, dtype=np.float32)
    if raw.shape != (OBS_WIDTH,):
        raise ValueError("F9-G accepts ONLY native 148-dimensional observation")
    obj = np.rint(
        raw[:147].reshape(PIXELS, 3)[:, 0] *
        max(OBJECT_TO_IDX.values())
    ).astype(np.int64)
    mask = np.empty((PIXELS, TYPES), dtype=np.uint8)
    mask[:, 0] = (obj == OBJECT_TO_IDX["key"])
    mask[:, 1] = (obj == OBJECT_TO_IDX["ball"])
    packed = 0
    for index, bit in enumerate(mask.reshape(-1)):
        if int(bit):
            packed |= 1 << index
    return packed


def unpack_spatial_bits(payload):
    if not isinstance(payload, int) or payload < 0 or payload >= 1 << (PIXELS*TYPES):
        raise ValueError("Invalid visible symbolic spatial payload")
    return np.fromiter(
        ((payload >> i) & 1 for i in range(PIXELS*TYPES)),
        dtype=np.float32,
        count=PIXELS*TYPES,
    )


COUNTER_BITS = 12
COUNTER_MASK = (1 << COUNTER_BITS) - 1


def pack_counts(counts):
    """One integer payload, 98 exact nonnegative observation counters."""
    if len(counts) != PIXELS*TYPES:
        raise ValueError("Wrong visible histogram width")
    packed=0
    for j,value in enumerate(counts):
        count=int(value)
        if not 0<=count<=COUNTER_MASK:
            raise ValueError("Observation frequency overflow")
        packed |= count << (j*COUNTER_BITS)
    return packed


def unpack_counts(payload):
    if not isinstance(payload,int) or payload<0:
        raise ValueError("Invalid cumulative memory payload")
    counts=np.fromiter(
        ((payload >> (i*COUNTER_BITS)) & COUNTER_MASK
         for i in range(PIXELS*TYPES)),
        dtype=np.float32,
        count=PIXELS*TYPES,
    )
    return counts/(counts+8.0)


class ContextualF4:
    """Contextual Event/Store/FIFO with TWO early anchors and ONE episode summary.

    Snapshot slots 0/1 hold first two *distinct* object-containing
    egocentric views. Slot 2 keeps 98 cumulative frequency counters by
    object-type AND egocentric pixel; it updates from EVERY visible frame.
    This is a deterministic non-evicting fixed-key admission policy, not
    F4 LearnedRelevance. Counts and views are public observation-only.
    """
    def __init__(self,mode="context"):
        if mode not in MODES:
            raise ValueError(mode)
        self.mode=mode
        self.selector=FIFO(SLOTS,seed=0)
        self.reset()

    def reset(self):
        self.store=self.selector.begin()
        self._tick=0
        self.timestamps={}
        self.counts=[0]*(PIXELS*TYPES)

    def encode(self,original):
        original=np.asarray(original,dtype=np.float32)
        if original.shape!=(OBS_WIDTH,):
            raise ValueError("Expected original MiniGrid 148-dim vector")
        payload=visible_spatial_bits(original)
        if self.mode!="zero" and payload:
            if 0 not in self.store.saved:
                self.selector.consider(self.store,Event(key=0,payload=payload))
                self.timestamps[0]=self._tick
            elif 1 not in self.store.saved and payload!=self.store.saved[0]:
                self.selector.consider(self.store,Event(key=1,payload=payload))
                self.timestamps[1]=self._tick

            for i in range(PIXELS*TYPES):
                if (payload >> i)&1:
                    self.counts[i]+=1
                    if self.counts[i]>COUNTER_MASK:
                        raise ValueError("Episode exceeds exact histogram counter range")

            self.selector.consider(
                self.store,Event(key=2,payload=pack_counts(self.counts)))
            if 2 not in self.timestamps:
                self.timestamps[2]=self._tick

        result=np.concatenate([original,self.readout()]).astype(np.float32)
        self._tick+=1
        return result

    def readout(self):
        out=np.zeros((SLOTS,SLOT_WIDTH),dtype=np.float32)
        if self.mode in ("zero","erase"):
            return out.flatten()
        for slot in range(SLOTS):
            if slot not in self.store.saved:
                continue
            payload=self.store.saved[slot]
            values=(
                unpack_counts(payload) if slot==2
                else unpack_spatial_bits(payload)
            )
            if self.mode=="swap_identity":
                values=values.reshape(PIXELS,TYPES)[:,::-1].reshape(-1)
            out[slot,:PIXELS*TYPES]=values
            age=self._tick-self.timestamps[slot]
            out[slot,PIXELS*TYPES]=min(1.0,age/100.0)
            out[slot,PIXELS*TYPES+1]=1.
        return out.flatten()

    def provenance(self):
        return {
            "event_ids":list(self.store.order),
            "timestamps":[self.timestamps[k] for k in self.store.order],
            "number_of_objects_seen_in_anchor":
                int(unpack_spatial_bits(self.store.saved[0]).sum())
                if 0 in self.store.saved else 0,
            "cumulative_visible_object_pixels":int(sum(self.counts)),
        }



def demonstration_pair_probe(seed, size):
    """Observer-only gate along genuine native teacher navigation.

    Teacher paths/labels are used here exclusively to select a *matching
    fork moment* for measuring memory. They are NOT fed to ContextualF4.
    The common action prefix is replayed via original raw observations.
    No teleport, no shaping, no trained agent and no heldout.
    """
    first_obs, first_actions, _ = demonstration(seed, size=size, reverse=False)
    second_obs, second_actions, _ = demonstration(seed, size=size, reverse=True)
    diverge = next(
        (i for i,(a,b) in enumerate(zip(first_actions,second_actions)) if a!=b),
        None,
    )
    if diverge is None:
        raise AssertionError("Expected teacher action divergence for changed cue")
    if not np.array_equal(first_obs[diverge], second_obs[diverge]):
        raise AssertionError("Decision frame must be the SAME public observation")
    memories = []
    sequences = []
    for source in (first_obs,second_obs):
        writer = ContextualF4()
        first_readout = None
        at_divergence = None
        for step in range(diverge+1):
            frame = writer.encode(source[step])
            if step == 0:
                first_readout = frame[OBS_WIDTH:].copy()
            if step == diverge:
                at_divergence = frame[OBS_WIDTH:].copy()
        memories.append((first_readout,at_divergence,writer.provenance()))
        sequences.append([int(i) for i in source[diverge][:OBS_WIDTH]])
    return {
        "seed": int(seed), "size":size, "decision_step":int(diverge),
        "initial_readout_distinct": bool(not np.array_equal(memories[0][0],memories[1][0])),
        "decision_readout_distinct": bool(not np.array_equal(memories[0][1],memories[1][1])),
        "identical_public_frame_at_decision": True,
        "first_memory_metadata":memories[0][2],
        "second_memory_metadata":memories[1][2],
        "public_frame_width":OBS_WIDTH,
        "memory_readout_width":READOUT_WIDTH,
    }


def run_gate(worlds=50):
    if not 2 <= worlds <= 100:
        raise ValueError("Development gate limited to 2..100 worlds per S7/S13")
    per_size = {}
    for size in (7,13):
        checks=[
            demonstration_pair_probe(checked_seed("development",0,j),size)
            for j in range(worlds)
        ]
        per_size[f"S{size}"] = {
            "pairs":worlds,
            "initial_readout_distinct":sum(x["initial_readout_distinct"] for x in checks),
            "decision_readout_distinct":sum(x["decision_readout_distinct"] for x in checks),
            "same_public_decision_frame":sum(x["identical_public_frame_at_decision"] for x in checks),
            "min_decision_step":min(x["decision_step"] for x in checks),
            "max_decision_step":max(x["decision_step"] for x in checks),
            "cases":checks,
        }
        s=per_size[f"S{size}"]
        print("F9G_GATE",f"S{size}","initial",s["initial_readout_distinct"],"/",worlds,
              "decision",s["decision_readout_distinct"],"/",worlds,
              "current_view_equal",s["same_public_decision_frame"],"/",worlds,flush=True)
    passed=all(
        p["decision_readout_distinct"]==worlds and
        p["same_public_decision_frame"]==worlds
        for p in per_size.values()
    )
    return {
        "experiment":"F9-G contextual F4 information sufficiency gate",
        "phase":"development_only_no_training",
        "passed":passed,
        "training_performed":False,
        "heldout_opened":False,
        "context_policy":"first two distinct public key/ball spatial views pinned; third slot accumulates pixel/type appearance counts",
        "f4_integration":"real Event/FIFO/Store with deterministic contextual event admission",
        "not_learned_relevance":True,
        "input_width":TOTAL_WIDTH,
        "memory_width":READOUT_WIDTH,
        "worlds_by_size":per_size,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worlds",type=int,default=50)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    r=run_gate(a.worlds)
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"gate.json").write_text(json.dumps(r,indent=2),encoding="utf8")
    if not r["passed"]:
        raise SystemExit("F9-G BLOCKED: memory does not distinguish both cues at decision")


if __name__=="__main__":
    main()
