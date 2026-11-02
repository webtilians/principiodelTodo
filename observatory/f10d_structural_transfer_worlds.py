"""F10-D: frozen transfer worlds with variable record counts and provenance patterns.

No actor weights, no old seeds, no training; four final queries share one
bit-identical 60-event history. Correct label derives from signed PUBLIC rows.
"""
from __future__ import annotations

import hashlib
import numpy as np
from observatory.f10b_temporal_evidence_task import (
    KEYS,EVENT_FEATURES,PUBLIC_FEATURES,CAPACITY,WORLD_EVENTS,
    event,PublicEventF4,_correct_from_public,
)

DEV_BASE=15_200_000
DEV_WORLDS=64
CARDINALITIES=(3,4,6,8)
CELLS=("signed|stable","unsigned|stable","signed|revised","unsigned|revised")
SOURCE_MODES=("iid","signed_A","signed_B","alternating")

def dev_seed(index,j):
    if index not in range(4) or j not in range(DEV_WORLDS):
        raise ValueError("F10-D frozen DEV only: no TRAIN/confirm executor")
    return DEV_BASE+10000*index+j

def _valid_source(rng,signed,t,mode):
    if mode=="iid":return int(rng.integers(0,2))
    if mode=="signed_A":return int(not signed)
    if mode=="signed_B":return int(signed)
    if mode=="alternating":return t%2
    raise ValueError("Unknown preregistered source mode")

def _target_history(rng,key,bit,k,cell):
    """Four hard-coded, target-agnostic types with randomized nuisance events."""
    if cell not in CELLS or k not in CARDINALITIES:
        raise ValueError("Unknown F10-D treatment")
    tail=[]
    if cell=="signed|stable":
        middle=[(1-bit,0)]
        middle += [(bit,1) if rng.integers(0,2) else (1-bit,0)
                   for _ in range(k-3)]
        finish=(bit,1)
    elif cell=="unsigned|stable":
        middle=[(bit,1)]
        middle += [(bit,1) if rng.integers(0,2) else (1-bit,0)
                   for _ in range(k-3)]
        finish=(1-bit,0)
    elif cell=="signed|revised":
        middle=[(1-bit,1)]
        middle += [(1-bit,1) if rng.integers(0,2) else (bit,0)
                   for _ in range(k-3)]
        finish=(1-bit,1)
    elif cell=="unsigned|revised":
        middle=[(1-bit,1)]
        middle += [(1-bit,1) if rng.integers(0,2) else (bit,0)
                   for _ in range(k-3)]
        finish=(bit,0)
    tail=[(key,val,signed) for val,signed in middle+[finish]]
    assert len(tail)==k-1
    return tail

def world(seed,index,j):
    assert seed==dev_seed(index,j)
    rng=np.random.default_rng(seed)
    perm=[int(v) for v in rng.permutation(KEYS)]
    targets={cell:perm[i] for i,cell in enumerate(CELLS)}
    bits=[int(x) for x in rng.integers(0,2,size=KEYS)]
    initial=[(k,bits[k],1) for k in rng.permutation(KEYS)]
    kmap={cell:CARDINALITIES[((j//4)+ci)%4] for ci,cell in enumerate(CELLS)}
    assert set(kmap.values())==set(CARDINALITIES)
    pending={}
    for cell,key in targets.items():
        pending[key]=_target_history(rng,key,bits[key],kmap[cell],cell)
    n_extra=60-24-sum(k-1 for k in kmap.values())
    assert n_extra==19
    other_keys=[k for k in perm if k not in set(targets.values())]
    for _ in range(n_extra):
        key=int(rng.choice(other_keys))
        pending.setdefault(key,[]).append((
            key,int(rng.integers(0,2)),int(rng.integers(0,2))))
    after=[]
    while pending:
        key=int(rng.choice(list(pending)))
        after.append(pending[key].pop(0))
        if not pending[key]:del pending[key]
    records=initial+after
    assert len(records)==60
    mode=SOURCE_MODES[j%4]
    writer=PublicEventF4()
    frames=[]
    for t,(key,val,signed) in enumerate(records):
        raw=event(int(key),int(val),int(signed),
                  _valid_source(rng,int(signed),t,mode),t)
        writer.observe(raw)
        frames.append(raw)
    history=np.stack(frames)
    memory=writer.readout()
    assert history.shape==(WORLD_EVENTS,PUBLIC_FEATURES)
    assert np.array_equal(memory[-WORLD_EVENTS:],history[:,:-1])
    digest=hashlib.sha256(history.tobytes()).hexdigest()
    cases=[]
    for cell,key in targets.items():
        query=np.zeros(PUBLIC_FEATURES,dtype=np.float32)
        query[key]=1.
        query[-1]=1.
        seq=np.concatenate((history,query[None]),axis=0)
        observed=[x for x in history if x[key]>0]
        assert len(observed)==kmap[cell]
        signs=[int(x[KEYS]) for x in observed if x[KEYS+1]==1]
        assert (signs[-1]!=signs[0])==(cell.endswith("revised"))
        assert bool(observed[-1][KEYS+1])==cell.startswith("signed")
        truth=_correct_from_public(history,key)
        assert truth==signs[-1]
        assert len({int(o[KEYS]) for o in observed})==2
        cases.append({
            "seed":seed,"cell":cell,"key":key,"truth":truth,
            "record_count":kmap[cell],"source_mode":mode,
            "history_sha256":digest,"public":seq.copy(),
            "memory":memory.copy(),"latest_unsigned":cell.startswith("unsigned"),
            "revised":cell.endswith("revised"),
        })
    assert len(cases)==4 and len({r["history_sha256"] for r in cases})==1
    assert all(np.array_equal(cases[0]["public"][:-1],c["public"][:-1])
               for c in cases)
    return cases

def majority_observed(case):
    key=case["key"]
    values=[int(r[KEYS]) for r in case["public"][:-1] if r[key]>0]
    return int(sum(values)*2>=len(values))

def last_any_observed(case):
    key=case["key"]
    return int([r for r in case["public"][:-1] if r[key]>0][-1][KEYS])

def last_signed_public(case):
    return _correct_from_public(case["public"][:-1],case["key"])
