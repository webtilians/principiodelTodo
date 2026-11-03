"""F10-L new data, disjoint from F10-I/J/K. Confirm cannot be generated.

XOR: four balanced b/flip combinations per group. SELECT: four versions
of one public evidence history, selector never sees F4 content.
OOD only available through the separate stage-two entrypoint.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np

MAX_SLOTS=12
FEATURES=4
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCE_MODES=("iid","signed_source0","signed_source1","alternating")
COUNTS={"train":4096,"dev":512,"ood":128}

def xor_seed(split,replica,j):
    if split not in ("train","dev") or replica not in range(4):
        raise ValueError("F10-L XOR only TRAIN/DEV. CONFIRM 23300000+ sealed")
    if j not in range(COUNTS[split]):
        raise ValueError("index not preregistered")
    base=23_100_000 if split=="train" else 23_200_000
    return base+10_000*replica+j

def xor_data(split,replica):
    cases=[]
    for j in range(COUNTS[split]):
        seed=xor_seed(split,replica,j)
        b=(j//2)%2
        f=j%2
        info={
            "task":"xor","split":split,"replica":replica,
            "seed":seed,"base_world":j//4,"bit":b,"flip":f,
            "truth":b^f,
            "input":np.array((2*b-1,2*f-1),dtype=np.float32),
        }
        info["hash"]=hashlib.sha256(
            f"{split}/{replica}/{seed}/{b}/{f}".encode()+info["input"].tobytes()
        ).hexdigest()
        cases.append(info)
    assert len(cases)==COUNTS[split]
    return cases

def select_seed(split,replica,world_index):
    if split not in ("train","dev","ood") or replica not in range(4):
        raise ValueError("F10-L no CONFIRM generator 23300000+")
    n={"train":1024,"dev":128,"ood":128}[split]
    if world_index not in range(n):
        raise ValueError("world outside preregistered range")
    base={"train":23_105_000,"dev":23_205_000,"ood":23_400_000}[split]
    return base+10_000*replica+world_index

def select_world(split,replica,j):
    seed=select_seed(split,replica,j)
    rng=np.random.default_rng(seed)
    n=(9 if j%2==0 else 11) if split=="ood" else (5 if j%2==0 else 7)
    n_signed=(4 if j%2==0 else 5) if split=="ood" else (2 if j%2==0 else 3)
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
    # Signed history can contain previous changes; one final signed value and
    # challenger X are definitely contradictory.
    values[int(signed[-1])]=current
    values[x]=1-current
    mode=SOURCE_MODES[(j//2)%4] if split=="ood" else "iid"
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
        assert status==expected, (split,replica,j,variant,status,expected)
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

def select_data(split,replica):
    worlds={"train":1024,"dev":128,"ood":128}[split]
    return [case for j in range(worlds) for case in select_world(split,replica,j)]
