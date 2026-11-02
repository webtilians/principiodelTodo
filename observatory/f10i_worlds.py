"""F10-I prospective experience worlds: no previous generator, no confirm namespace.

Counterfactual evidence is public; actor receives no winner/truth during rollouts.
Verifier observation is only revealed AFTER the policy has acted.
"""
from __future__ import annotations
import hashlib
import numpy as np

KEYS=24
EVENTS=48
FEATURES=KEYS+5
VARIANTS=("U_late","U_early","S_late","S_early")
SOURCE_MODES=("iid","signed_source_0","signed_source_1","alternating_storage_source")
BASE={"train":20_100_000,"dev":20_200_000,"ood":20_400_000}
COUNTS={"train":(5,8),"dev":(5,8),"ood":(12,16)}
SIGNED={"train":(3,5),"dev":(3,5),"ood":(6,9)}
N={"train":512,"dev":128,"ood":128}

def seed_for(split,replica,j):
    if split not in BASE or replica not in range(4) or j not in range(N.get(split,0)):
        raise ValueError("F10-I only registered TRAIN/DEV/OOD; confirm sealed")
    return BASE[split]+10_000*replica+j

def world(split,replica,j):
    seed=seed_for(split,replica,j)
    rng=np.random.default_rng(seed)
    k=COUNTS[split][j%2]
    s=SIGNED[split][j%2]
    assert s<=k-2 and s>=3
    q=int(rng.integers(KEYS))
    b=int(rng.integers(2))
    inv=int(rng.integers(2))
    keys=np.concatenate((np.full(k,q,dtype=np.int64),
                         rng.choice([key for key in range(KEYS) if key!=q],EVENTS-k)))
    values=rng.integers(2,size=EVENTS).astype(np.int64)
    times=rng.permutation(EVENTS).astype(np.int64)
    target_sorted=np.argsort(times[:k])
    # Always one unsigned report before and after the latest signed;
    # winner and challenge occupy arbitrary storage positions after permutation.
    signed_ranks=np.sort(rng.choice(np.arange(1,k-1),size=s,replace=False))
    signed_indices=target_sorted[signed_ranks]
    signs=rng.integers(2,size=EVENTS).astype(np.int64)
    signs[:k]=0
    signs[signed_indices]=1
    for n,slot in enumerate(signed_indices):
        values[slot]=b^((s-1-n)%2)
    assert values[signed_indices[-1]]==b
    last_rank=int(signed_ranks[-1])
    eligible_after=[int(target_sorted[t]) for t in range(last_rank+1,k)]
    x_idx=int(rng.choice(eligible_after))
    before_idx=int(target_sorted[0])
    assert signs[x_idx]==signs[before_idx]==0 and x_idx!=before_idx
    values[x_idx]=1-b
    # Source metadata assigned before variants; remains constant even if X signs.
    storage_perm=rng.permutation(EVENTS)
    mode=(SOURCE_MODES[(j//2)%4] if split=="ood" else "iid")
    if mode=="iid":
        sources=rng.integers(2,size=EVENTS).astype(np.int64)
    elif mode=="signed_source_0":
        sources=(1-signs).copy()
    elif mode=="signed_source_1":
        sources=signs.copy()
    else:
        sources=np.empty(EVENTS,dtype=np.int64)
        sources[storage_perm]=np.arange(EVENTS,dtype=np.int64)%2
    # All four variants share key, query, initial values, source, physical storage.
    # Shift swaps X timestamp with earlier unsigned; signing X changes the last
    # signed truth ONLY when X retains its late timestamp.
    rows=[]
    for variant in VARIANTS:
        v_sign=signs.copy()
        v_time=times.copy()
        if variant.startswith("S_"):
            v_sign[x_idx]=1
        if variant.endswith("_early"):
            v_time[x_idx],v_time[before_idx]=v_time[before_idx],v_time[x_idx]
        memory=np.zeros((EVENTS,FEATURES),dtype=np.float32)
        memory[np.arange(EVENTS),keys]=1.
        memory[:,KEYS]=(values*2-1).astype(np.float32)
        memory[:,KEYS+1]=v_sign.astype(np.float32)
        memory[:,KEYS+2]=v_time.astype(np.float32)/EVENTS
        memory[np.arange(EVENTS),KEYS+3+sources]=1.
        memory=memory[storage_perm]
        valid=memory[:,q]>.5
        signed=valid & (memory[:,KEYS+1]>.5)
        assert int(signed.sum())>=3
        winner=int(np.argmax(np.where(signed,memory[:,KEYS+2],-1)))
        true_status=int(memory[winner,KEYS]>0)
        assert true_status==((1-b) if variant=="S_late" else b)
        action_truth=true_status^inv
        ordered=sorted(memory[signed],key=lambda x:float(x[KEYS+2]))
        flips=sum((a[KEYS]>0)!=(bb[KEYS]>0) for a,bb in zip(ordered,ordered[1:]))
        assert flips>=2
        latest_signed=float(memory[winner,KEYS+2])
        assert any(row[q]>.5 and row[KEYS+1]==0 and
                   row[KEYS+2]>latest_signed for row in memory)
        history=np.zeros((EVENTS+1,FEATURES+1),np.float32)
        history[:EVENTS,:FEATURES]=memory
        history[-1,q]=1.
        history[-1,-1]=inv
        # Independent noisy verifier appears ONLY after final action.
        sensor=true_status^int(rng.random()<.20)
        delay=2+(j%3)
        rows.append({
            "seed":seed,"split":split,"replica":replica,
            "variant":variant,"k":k,"signed_count":int(signed.sum()),
            "signed_value_changes":flips,"source_mode":mode,
            "history":history,"memory":memory,"winner":winner,
            "truth":action_truth,"true_status":true_status,
            "sensor":sensor,"delay":delay,
            "q":q,"inv":inv,
            "hash":hashlib.sha256(history.tobytes()).hexdigest(),
        })
    assert len(rows)==4 and len({r["hash"] for r in rows})==4
    assert len({r["q"] for r in rows})==len({r["inv"] for r in rows})==1
    assert sum(r["truth"]!=rows[0]["truth"] for r in rows)==1
    return rows

def dataset(split,replica):
    return [row for j in range(N[split]) for row in world(split,replica,j)]
