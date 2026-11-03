"""INFINITO F10-J prospective worlds: metadata observable, F4 VALUES PRIVATE.

No F10-I seeds, no product benchmarks, and no confirmation namespace.
Informative queries require READ; public queries explicitly provide the answer
and should SKIP. Multiple signed versions and an unsigned newer distractor.
"""
from __future__ import annotations
import hashlib
import numpy as np

KEYS=24
EVENTS=48
FEATURES=KEYS+5
VARIANTS=("U_late","U_early","S_late","S_early")
SOURCE_MODES=("iid","signed_source_0","signed_source_1","alternating_storage_source")
BASE={"train":21_100_000,"dev":21_200_000,"ood":21_400_000}
N={"train":512,"dev":128,"ood":128}
COUNTS={"train":(6,9),"dev":(6,9),"ood":(12,16)}
SIGNED={"train":(3,5),"dev":(3,5),"ood":(6,9)}

def seed_for(split,replica,j):
    if split not in BASE or replica not in range(4) or j not in range(N.get(split,0)):
        raise ValueError("F10-J only preregistered TRAIN/DEV/OOD; CONFIRM 21300000+ SEALED")
    return BASE[split]+10_000*replica+j

def world(split,replica,j):
    seed=seed_for(split,replica,j)
    rng=np.random.default_rng(seed)
    k=COUNTS[split][j%2]
    s=SIGNED[split][j%2]
    assert s<=k-3 and s>=3
    info=(j%2==0)
    query=int(rng.integers(KEYS))
    latest_bit=int(rng.integers(2))
    public_bit=int(rng.integers(2))
    xor=int(rng.integers(2))
    keys=np.concatenate((
        np.full(k,query,dtype=np.int64),
        rng.choice([q for q in range(KEYS) if q!=query],EVENTS-k)
    ))
    values=rng.integers(2,size=EVENTS).astype(np.int64)
    times=rng.permutation(EVENTS).astype(np.int64)
    order=np.argsort(times[:k])
    signed_ranks=np.sort(rng.choice(np.arange(1,k-2),size=s,replace=False))
    signed_slots=order[signed_ranks]
    signs=rng.integers(2,size=EVENTS).astype(np.int64)
    signs[:k]=0
    signs[signed_slots]=1
    # Last accredited report is random 0/1; earlier signed updates alternate.
    for n,slot in enumerate(signed_slots):
        values[slot]=latest_bit^((s-1-n)%2)
    assert values[signed_slots[-1]]==latest_bit
    last_rank=int(signed_ranks[-1])
    candidates=[int(order[p]) for p in range(last_rank+1,k-1)]
    assert candidates
    x_slot=int(rng.choice(candidates))
    early_slot=int(order[0])
    assert signs[x_slot]==0 and signs[early_slot]==0
    values[x_slot]=1-latest_bit
    storage=rng.permutation(EVENTS)
    mode=SOURCE_MODES[(j//2)%4] if split=="ood" else "iid"
    if mode=="iid":
        sources=rng.integers(2,size=EVENTS).astype(np.int64)
    elif mode=="signed_source_0":
        sources=(1-signs).copy()
    elif mode=="signed_source_1":
        sources=signs.copy()
    else:
        sources=np.empty(EVENTS,dtype=np.int64)
        sources[storage]=np.arange(EVENTS,dtype=np.int64)%2

    rows=[]
    for variant in VARIANTS:
        sig=signs.copy()
        ts=times.copy()
        if variant.startswith("S_"):
            sig[x_slot]=1
        if variant.endswith("_early"):
            ts[x_slot],ts[early_slot]=ts[early_slot],ts[x_slot]
        mem=np.zeros((EVENTS,FEATURES),np.float32)
        mem[np.arange(EVENTS),keys]=1.
        mem[:,KEYS]=(values*2-1).astype(np.float32)
        mem[:,KEYS+1]=sig.astype(np.float32)
        mem[:,KEYS+2]=ts.astype(np.float32)/EVENTS
        mem[np.arange(EVENTS),KEYS+3+sources]=1.
        mem=mem[storage]
        eligible=(mem[:,query]>.5)
        signed=eligible&(mem[:,KEYS+1]>.5)
        winner=int(np.argmax(np.where(signed,mem[:,KEYS+2],-1.)))
        true_memory_bit=int(mem[winner,KEYS]>0)
        assert true_memory_bit==((1-latest_bit) if variant=="S_late" else latest_bit)
        truth=(true_memory_bit if info else public_bit)^xor
        hist=np.zeros((EVENTS+1,FEATURES+1),np.float32)
        hist[:EVENTS,:FEATURES]=mem
        hist[:EVENTS,KEYS]=0.   # CRITICAL: no F4 values in recurrent public history.
        hist[-1,query]=1.
        hist[-1,KEYS+1]=float(info)
        hist[-1,KEYS+2]=float(public_bit*2-1)
        hist[-1,-1]=float(xor)
        assert np.count_nonzero(hist[:EVENTS,KEYS])==0
        assert hist[-1,KEYS]==0
        signed_updates=sorted(mem[signed],key=lambda v:float(v[KEYS+2]))
        changes=sum((a[KEYS]>0)!=(b[KEYS]>0) for a,b in zip(signed_updates,signed_updates[1:]))
        assert changes>=2
        assert any(row[query]>.5 and row[KEYS+1]==0 and
                   row[KEYS+2]>mem[winner,KEYS+2] for row in mem)
        digest=hashlib.sha256(hist.tobytes()+mem.tobytes()).hexdigest()
        rows.append({
            "seed":seed,"split":split,"replica":replica,
            "variant":variant,"informative":bool(info),
            "k":int(k),"signed_count":int(signed.sum()),
            "signed_changes":int(changes),"source_mode":mode,
            "history":hist,"memory":mem,
            "q":query,"xor":xor,"public_bit":public_bit,
            "winner":winner,"true_memory_bit":true_memory_bit,
            "truth":int(truth),
            "delay":2+j%3,"hash":digest,
        })
    assert len(rows)==4 and len({row["hash"] for row in rows})==4
    by={row["variant"]:row for row in rows}
    if info:
        assert by["U_late"]["truth"]!=by["S_late"]["truth"]
        assert by["U_late"]["truth"]==by["U_early"]["truth"]==by["S_early"]["truth"]
    else:
        assert len({row["truth"] for row in rows})==1
    return rows

def dataset(split,replica):
    return [row for j in range(N[split]) for row in world(split,replica,j)]
