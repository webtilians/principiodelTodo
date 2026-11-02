"""F10-C: novel 60-event histories with FIVE reports per queried key.

Only public provenance, timestamps, values and randomized source identities.
The same history supports all four queries; no query-target marker until end.
"""
from __future__ import annotations
import hashlib
import numpy as np
from observatory.f10b_temporal_evidence_task import (
    KEYS,EVENT_FEATURES,PUBLIC_FEATURES,CAPACITY,WORLD_EVENTS,
    CELLS,event,PublicEventF4,_correct_from_public,stale_memory,forged_credential,
)

TRAIN_BASE=14_100_000
DEV_BASE=14_200_000
TRAIN_WORLDS=512
DEV_WORLDS=64
GROUP_NAMES=tuple(a+"|"+b for a,b in CELLS)

def seed_for(split,index,j):
    if split not in ("train","dev") or index not in range(4):
        raise ValueError("F10-C has only prospective TRAIN and DEV worlds")
    upper=TRAIN_WORLDS if split=="train" else DEV_WORLDS
    if j not in range(upper):
        raise ValueError("World index exceeds preregistered limit")
    return (TRAIN_BASE if split=="train" else DEV_BASE)+10_000*index+j

def _compose(seed):
    rng=np.random.default_rng(seed)
    perm=[int(x) for x in rng.permutation(KEYS)]
    keys={name:perm[2*i:2*i+2] for i,name in enumerate(GROUP_NAMES)}
    target_ids={k for values in keys.values() for k in values}
    initial_bits=[int(x) for x in rng.integers(0,2,size=KEYS)]
    base=[(int(k),initial_bits[k],1,int(rng.integers(0,2)))
          for k in rng.permutation(KEYS)]
    pending={}
    for cell,ids in keys.items():
        for k in ids:
            b=initial_bits[k]
            if cell=="signed|stable":
                pattern=[(1-b,0),(1-b,0),(b,1),(b,1)]
            elif cell=="unsigned|stable":
                pattern=[(1-b,0),(b,1),(1-b,0),(1-b,0)]
            elif cell=="signed|revised":
                pattern=[(1-b,1),(b,0),(1-b,1),(1-b,1)]
            elif cell=="unsigned|revised":
                pattern=[(1-b,1),(b,0),(b,0),(b,0)]
            else:raise AssertionError(cell)
            pending[k]=[(k,value,signed,int(rng.integers(0,2)))
                        for value,signed in pattern]
    # The last 4 reports belong to NONQUERY keys; prevent target leakage.
    distractors=[k for k in perm if k not in target_ids]
    for k in rng.choice(distractors,4,replace=False):
        k=int(k)
        b=initial_bits[k]
        pending[k]=[(k,1-b,0,int(rng.integers(0,2)))]
    tail=[]
    while pending:
        k=int(rng.choice(list(pending)))
        tail.append(pending[k].pop(0))
        if not pending[k]:del pending[k]
    reports=base+tail
    assert len(reports)==WORLD_EVENTS
    return reports,keys,rng

def world(seed):
    reports,keys,rng=_compose(seed)
    store=PublicEventF4()
    observed=[]
    for t,(key,value,signed,src) in enumerate(reports):
        frame=event(key,value,signed,src,t)
        store.observe(frame)
        observed.append(frame)
    history=np.stack(observed)
    mem=store.readout()
    assert history.shape==(WORLD_EVENTS,PUBLIC_FEATURES)
    assert mem.shape==(CAPACITY,EVENT_FEATURES)
    assert np.array_equal(mem[-WORLD_EVENTS:],history[:,:-1])
    digest=hashlib.sha256(history.tobytes()).hexdigest()
    out=[]
    for cell in GROUP_NAMES:
        key=int(rng.choice(keys[cell]))
        q=np.zeros(PUBLIC_FEATURES,np.float32)
        q[key]=1.
        q[-1]=1.
        x=np.concatenate((history,q[None]),axis=0)
        occurrences=[row for row in history if row[key]>0]
        assert len(occurrences)==5
        signature,kind=cell.split("|")
        signed=[int(row[KEYS]) for row in occurrences if row[KEYS+1]==1]
        assert bool(occurrences[-1][KEYS+1])==(signature=="signed")
        assert (signed[0]!=signed[-1])==(kind=="revised")
        assert len({int(z[KEYS]) for z in occurrences})==2
        true=_correct_from_public(history,key)
        assert true==signed[-1]
        out.append({
            "seed":seed,"key":key,"cell":cell,"truth":true,
            "public":x.copy(),"memory":mem.copy(),"history_sha256":digest,
            "number_of_matching_reports":len(occurrences),
            "number_of_contradictions":sum(int(row[KEYS])!=true for row in occurrences),
            "last_unsigned":signature=="unsigned",
            "latest_unsigned":signature=="unsigned",
            "revised":kind=="revised",
        })
    assert len(out)==4 and len({c["key"] for c in out})==4
    assert len({c["history_sha256"] for c in out})==1
    assert all(np.array_equal(out[0]["public"][:-1],c["public"][:-1]) for c in out)
    return out
