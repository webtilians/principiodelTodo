"""F10-H new multi-revision worlds, independent of all F10-G data/code generators.

Four matched public-factorial variants for each fixed 48-event world. Only
reports public to actors define labels. No confirmation generator.
"""
from __future__ import annotations

import hashlib
import numpy as np

KEYS=24
EVENTS=48
FEATURES=KEYS+5
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCE_MODES=("iid","signed_source_0","signed_source_1","alternating_storage_source")
BASE={"train":19_100_000,"dev":19_200_000,"ood":19_400_000}
COUNTS={"train":(7,9),"dev":(7,9),"ood":(12,16)}
PRIOR_SIGNED={"train":(3,4),"dev":(3,4),"ood":(5,7)}

def seed_for(split,replica,j):
    if split not in BASE or replica not in range(4):
        raise ValueError("Only preregistered TRAIN/DEV/OOD; confirmations sealed")
    n=512 if split=="train" else 128
    if j not in range(n):raise ValueError("World outside preregistered range")
    return BASE[split]+10000*replica+j

def mode_for(split,j):
    return SOURCE_MODES[(j//2)%4] if split=="ood" else "iid"

def world(split,replica,j):
    seed=seed_for(split,replica,j)
    rng=np.random.default_rng(seed)
    k=COUNTS[split][j%2]
    pre=PRIOR_SIGNED[split][j%2]
    q=int(rng.integers(KEYS))
    b=int(rng.integers(2))
    inversion=int(rng.integers(2))
    keys=np.array([q]*k+[int(x) for x in rng.choice(
        [t for t in range(KEYS) if t!=q],EVENTS-k)],dtype=np.int64)
    values=rng.integers(2,size=EVENTS).astype(np.int64)
    signs=rng.integers(2,size=EVENTS).astype(np.int64)
    signs[:k]=0
    signs[:pre]=1
    c_idx=pre
    x_idx=pre+1
    signs[c_idx]=1
    # At least 3 accredited changes BEFORE optional accredited X:
    # pre signed values alternate, ending in 1-b; C then sets b.
    for i in range(pre):
        values[i]=b^((pre-i)%2)
    values[c_idx]=b
    values[x_idx]=1-b
    assert pre>=3 and c_idx+1<k
    assert sum(signs[:k])==pre+1
    times=np.empty(EVENTS,dtype=np.int64)
    earlier=np.sort(rng.choice(np.arange(36),size=pre,replace=False))
    contender=np.sort(rng.choice(np.arange(36,EVENTS),size=2,replace=False))
    times[:pre]=earlier
    times[c_idx],times[x_idx]=contender
    used=set(int(t) for t in earlier)|set(int(t) for t in contender)
    remaining=np.array([t for t in range(EVENTS) if t not in used])
    rng.shuffle(remaining)
    indices=[i for i in range(EVENTS) if i not in tuple(range(pre))+(c_idx,x_idx)]
    assert len(indices)==len(remaining)
    times[indices]=remaining
    assert len(set(times.tolist()))==EVENTS
    assert all(times[t]<times[c_idx] for t in range(pre))
    assert all(times[t]<times[x_idx] for t in range(pre))
    # A/B nuisance metadata is defined ONCE and remains fixed across the
    # signature interventions (X becomes an adversarial exception).
    # The storage permutation is also fixed across the four variants.
    storage_perm=rng.permutation(EVENTS)
    mode=mode_for(split,j)
    if mode=="iid":
        sources=rng.integers(2,size=EVENTS).astype(np.int64)
    elif mode=="signed_source_0":
        sources=(1-signs).copy()
    elif mode=="signed_source_1":
        sources=signs.copy()
    else:
        sources=np.empty(EVENTS,dtype=np.int64)
        sources[storage_perm]=np.arange(EVENTS,dtype=np.int64)%2
    rows=[]
    for variant in VARIANTS:
        v_signs=signs.copy()
        v_times=times.copy()
        v_signs[x_idx]=int(variant.startswith("S"))
        if variant.endswith("_before"):
            v_times[x_idx],v_times[c_idx]=contender[0],contender[1]
        else:
            v_times[x_idx],v_times[c_idx]=contender[1],contender[0]
        memory=np.zeros((EVENTS,FEATURES),np.float32)
        memory[np.arange(EVENTS),keys]=1.
        memory[:,KEYS]=(values*2-1).astype(np.float32)
        memory[:,KEYS+1]=v_signs
        memory[:,KEYS+2]=v_times/EVENTS
        memory[np.arange(EVENTS),KEYS+3+sources]=1.
        memory=memory[storage_perm]
        valid=memory[:,q]>.5
        signed=valid&(memory[:,KEYS+1]>.5)
        assert int(signed.sum())>=4
        winner=int(np.argmax(np.where(signed,memory[:,KEYS+2],-1.)))
        truth=int(memory[winner,KEYS]>0.)^inversion
        expected=((1-b) if variant=="S_after" else b)^inversion
        assert truth==expected
        history=np.zeros((EVENTS+1,FEATURES+1),np.float32)
        history[:EVENTS,:FEATURES]=memory
        history[-1,q]=1.
        history[-1,-1]=inversion
        signed_sequence=[int(z[KEYS]>0) for z in sorted(
            memory[signed],key=lambda z:float(z[KEYS+2]))]
        changes=sum(a!=b2 for a,b2 in zip(signed_sequence,signed_sequence[1:]))
        assert changes>=3
        rows.append({
            "seed":seed,"split":split,"replica":replica,"j":j,
            "variant":variant,"source_mode":mode,
            "k":k,"signed_count":int(signed.sum()),"signed_value_changes":changes,
            "history":history,"memory":memory,
            "winner":winner,"truth":truth,
            "base_value":b,"inversion":inversion,"query":q,
            "hash":hashlib.sha256(history.tobytes()).hexdigest(),
        })
    assert len(rows)==4 and len({x["hash"] for x in rows})==4
    assert all(x["query"]==q and x["inversion"]==inversion for x in rows)
    return rows

def dataset(split,replica):
    n=512 if split=="train" else 128
    return [x for j in range(n) for x in world(split,replica,j)]
