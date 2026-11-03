"""F10-L-C confirmation worlds using FROZEN F10-L OOD algorithm, new seeds.

This module MUST NOT be used in tests on 23300000+.
Preflight exercises only build_world with *existing OOD* seeds, compares
bit-for-bit to the original select_world("ood",rep,j). Before the final
original-checkpoint SHA gate, no code imports/executes confirm_cases.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from observatory.f10l_worlds import MAX_SLOTS,FEATURES,VARIANTS,SOURCE_MODES

CONFIRM_BASE=23_300_000
N_WORLDS=128

def build_world(seed,replica,j,split):
    """Exact copy of select_world's OOD path apart from explicit seed/split."""
    if replica not in range(4) or j not in range(N_WORLDS):
        raise ValueError("F10-L-C only 4 frozen replicas x128 worlds")
    if split not in ("ood","confirm"):
        raise ValueError("No training nor DEV data in confirmation generator")
    rng=np.random.default_rng(seed)
    n=(9 if j%2==0 else 11)
    n_signed=(4 if j%2==0 else 5)
    assert n_signed<=n-3
    timestamps=rng.permutation(n)
    order=np.argsort(timestamps)
    sranks=np.sort(rng.choice(np.arange(1,n-2),size=n_signed,replace=False))
    signed=order[sranks]
    signs=np.zeros(n,dtype=np.int64)
    signs[signed]=1
    x_candidates=order[int(sranks[-1])+1:n-1]
    assert len(x_candidates)>0
    x=int(rng.choice(x_candidates))
    early=int(order[0])
    assert signs[x]==0 and signs[early]==0
    current=int(rng.integers(2))
    values=rng.integers(2,size=n)
    values[int(signed[-1])]=current
    values[x]=1-current
    mode=SOURCE_MODES[(j//2)%4]
    if mode=="iid":
        sources=rng.integers(2,size=n)
    elif mode=="signed_source0":
        sources=1-signs
    elif mode=="signed_source1":
        sources=signs.copy()
    else:
        sources=np.arange(n)%2
    storage=rng.permutation(n)
    rows=[]
    for variant in VARIANTS:
        vs=signs.copy()
        vt=timestamps.copy()
        if variant.startswith("S_"):
            vs[x]=1
        if variant.endswith("_before"):
            vt[x],vt[early]=vt[early],vt[x]
        feats=np.zeros((MAX_SLOTS,FEATURES),np.float32)
        contents=np.zeros(MAX_SLOTS,np.float32)
        mask=np.zeros(MAX_SLOTS,np.float32)
        feats[:n,0]=vs[storage].astype(np.float32)
        feats[:n,1]=vt[storage].astype(np.float32)/MAX_SLOTS
        feats[:n,2]=feats[:n,0]*feats[:n,1]
        feats[:n,3]=sources[storage].astype(np.float32)
        contents[:n]=2*values[storage]-1
        mask[:n]=1.
        signed_positions=np.flatnonzero((feats[:,0]>.5)&(mask>.5))
        winner=int(signed_positions[np.argmax(feats[signed_positions,1])])
        status=int(contents[winner]>0)
        expected=(1-current) if variant=="S_after" else current
        assert status==expected,(split,replica,j,variant,status,expected)
        assert any(mask[z]>.5 and feats[z,0]==0 and feats[z,1]>feats[winner,1]
                   for z in range(MAX_SLOTS))
        query={
            "task":"select","split":split,"replica":replica,
            "seed":seed,"base_world":j,"variant":variant,"source_mode":mode,
            "n":n,"signed_count":int(sum(vs)),
            "winner":winner,"status":status,
            "features":feats,"contents":contents,"mask":mask,
        }
        query["hash"]=hashlib.sha256(
            json.dumps({k:v for k,v in query.items() if not isinstance(v,np.ndarray)},
                       sort_keys=True).encode()
            +feats.tobytes()+contents.tobytes()+mask.tobytes()
        ).hexdigest()
        rows.append(query)
    by={r["variant"]:r for r in rows}
    assert by["U_before"]["status"]==by["U_after"]["status"]==by["S_before"]["status"]
    assert by["S_after"]["status"]!=by["U_after"]["status"]
    assert len({r["hash"] for r in rows})==4
    return rows

def confirm_cases(replica):
    """Only call after original SHA check and OOD-inference regression PASS."""
    if replica not in range(4):
        raise ValueError("not a preregistered replica")
    return [case for j in range(N_WORLDS) for case in
            build_world(CONFIRM_BASE+10_000*replica+j,replica,j,"confirm")]

def regression_existing_ood(replica):
    """PRE-confirm: uses previously opened OOD only, new generator must match."""
    if replica not in range(4):
        raise ValueError("not a preregistered replica")
    return [case for j in range(N_WORLDS) for case in
            build_world(23_400_000+10_000*replica+j,replica,j,"ood")]
