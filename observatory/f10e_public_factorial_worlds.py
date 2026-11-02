"""F10-E frozen 2x2 public credential x temporal precedence factorial.

Generates all four counterfactual HISTORIES from a single world seed. No actor,
oracle label, D effect or future query is supplied to the event writer.
"""
from __future__ import annotations

import hashlib
import numpy as np
from observatory.f10b_temporal_evidence_task import (
    KEYS, EVENT_FEATURES, PUBLIC_FEATURES, CAPACITY, WORLD_EVENTS,
    event, PublicEventF4, _correct_from_public,
)

BASE=16_200_000
N_WORLDS=108
COUNTS=(3,5,8)
TIMINGS=("early","middle","late")
SLOTS=((34,35),(39,49),(48,59))
VARIANTS=("U_before","U_after","S_before","S_after")

def dev_seed(replica,j):
    if replica not in range(4) or j not in range(N_WORLDS):
        raise ValueError("F10-E fixed 4x108 development worlds only")
    return BASE+10000*replica+j

def _build_source(seed,replica,j):
    assert seed==dev_seed(replica,j)
    rng=np.random.default_rng(seed)
    # j%9 traverses all 3x3 nuisance conditions exactly 12x/replica.
    k=COUNTS[(j%9)//3]
    time_idx=j%3
    timing=TIMINGS[time_idx]
    slots=SLOTS[time_idx]
    q=int(rng.integers(0,KEYS))
    b=int(rng.integers(0,2))
    records=[None]*WORLD_EVENTS
    keys=[int(z) for z in rng.permutation(KEYS)]
    # 24 immutable baseline reports: for every key an authentic observation.
    for t,key in enumerate(keys):
        value=b if key==q else int(rng.integers(0,2))
        records[t]=(key,value,1,int(rng.integers(0,2)))
    # Extra reports for q before the pair. No signed conflicting value.
    for extra in range(k-3):
        choose_signed=int(rng.integers(0,2))
        records[24+extra]=(q,b if choose_signed else 1-b,
                           choose_signed,int(rng.integers(0,2)))
    other=[key for key in keys if key!=q]
    for t in range(24,WORLD_EVENTS):
        if records[t] is None and t not in slots:
            records[t]=(int(rng.choice(other)),int(rng.integers(0,2)),
                        int(rng.integers(0,2)),int(rng.integers(0,2)))
    confirm=(q,b,1,int(rng.integers(0,2)))
    challenge=(q,1-b,None,int(rng.integers(0,2)))
    assert len(records)==60 and sum(r is None for r in records)==2
    return records,confirm,challenge,q,b,k,timing,slots

def world(seed,replica,j):
    records,confirm,challenge,q,b,k,timing,slots=_build_source(
        seed,replica,j)
    result=[]
    for name in VARIANTS:
        signed=name.startswith("S_")
        before=name.endswith("_before")
        x=(challenge[0],challenge[1],int(signed),challenge[3])
        # before means contradicting X precedes confirming C in time.
        left,right=(x,confirm) if before else (confirm,x)
        current=records.copy()
        current[slots[0]]=left
        current[slots[1]]=right
        store=PublicEventF4()
        seen=[]
        for t,(key,val,cred,source) in enumerate(current):
            row=event(key,val,cred,source,t)
            store.observe(row)
            seen.append(row)
        history=np.stack(seen)
        f4=store.readout()
        assert history.shape==(WORLD_EVENTS,PUBLIC_FEATURES)
        assert f4.shape==(CAPACITY,EVENT_FEATURES)
        assert np.array_equal(f4[-WORLD_EVENTS:],history[:,:-1])
        query=np.zeros(PUBLIC_FEATURES,np.float32)
        query[q]=1.
        query[-1]=1.
        truth=_correct_from_public(history,q)
        assert truth==(1-b if signed and not before else b),(
            seed,name,truth,b)
        assert sum(row[q]>0 for row in history)==k
        assert all(row[-1]==0 for row in history)
        result.append({
            "seed":seed,"replica":replica,"world_j":j,
            "variant":name,"truth":truth,"base_value":b,"key":q,
            "record_count":k,"timing":timing,"slots":list(slots),
            "history_sha256":hashlib.sha256(history.tobytes()).hexdigest(),
            "public":np.concatenate((history,query[None]),axis=0),
            "memory":f4,
            "challenge_source":challenge[3],"confirm_source":confirm[3],
            "challenge_signed":bool(signed),"challenge_before":bool(before),
        })
    assert len(result)==4 and {x["variant"] for x in result}==set(VARIANTS)
    assert len({x["key"] for x in result})==1
    return result

def public_last_any(case):
    q=case["key"]
    return int([int(r[KEYS]) for r in case["public"][:-1] if r[q]>0][-1])

def public_last_signed(case):
    return _correct_from_public(case["public"][:-1],case["key"])
