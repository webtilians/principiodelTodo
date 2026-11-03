"""INFINITO F10-N: prospectively new worlds for operation-sequence discovery.

TRAIN/DEV include PUBLIC+0, PUBLIC+1, MEMORY+0.
OOD is ONLY the never-trained MEMORY+1 combination.
Confirmation 26300000+ is intentionally not addressable.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import numpy as np

OPS=("P","S","R","X")
PROGRAMS=((),)+tuple(
    seq for n in (1,2,3) for seq in itertools.product(OPS,repeat=n)
)
assert len(PROGRAMS)==85
HELDOUT_PROGRAM=("S","R","X")
TRAIN_PROGRAMS=tuple(seq for seq in PROGRAMS if seq!=HELDOUT_PROGRAM)
assert len(TRAIN_PROGRAMS)==84 and HELDOUT_PROGRAM not in TRAIN_PROGRAMS
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCES=("iid","signed_source0","signed_source1","alternating")
TASKS=(("PUBLIC",0),("PUBLIC",1),("MEMORY",0))
MAX_SLOTS=12
N_FEATURES=4
TRAIN_BASE=26_100_000
DEV_BASE=26_200_000
OOD_BASE=26_400_000

def seed_for(split,rep,j,task=None):
    if split not in ("train","dev","ood") or rep not in range(4):
        raise ValueError("F10-N CONFIRM 26300000+ is SEALED")
    n={"train":256,"dev":64,"ood":128}[split]
    if j not in range(n):
        raise ValueError("outside preregistered split/world")
    if split=="ood":
        if task not in (None,("MEMORY",1)):
            raise ValueError("F10-N OOD is exclusively unseen MEMORY+FLIP1")
        return OOD_BASE+10000*rep+j
    if task not in TASKS:
        raise ValueError("F10-N TRAIN/DEV only three listed contexts")
    t=TASKS.index(task)
    base=TRAIN_BASE if split=="train" else DEV_BASE
    return base+10000*rep+1000*t+j

def program_features():
    """Only syntactic descriptors: no optimal-program indicators or labels."""
    feats=np.zeros((len(PROGRAMS),33),np.float32)
    for i,seq in enumerate(PROGRAMS):
        feats[i,-1]=len(seq)/3.
        for pos,op in enumerate(seq):
            k=OPS.index(op)
            feats[i,k]+=1/3.
            feats[i,4+pos*4+k]=1.
        for a,b in zip(seq,seq[1:]):
            feats[i,16+4*OPS.index(a)+OPS.index(b)]+=0.5
    assert feats.shape==(85,33)
    return feats

def _world(split,rep,j,task):
    seed=seed_for(split,rep,j,task)
    rng=np.random.default_rng(seed)
    n=(9 if j%2==0 else 11) if split=="ood" else (5 if j%2==0 else 7)
    signed_count=(4 if j%2==0 else 5) if split=="ood" else (2 if j%2==0 else 3)
    assert signed_count<=n-3
    times=rng.permutation(n)
    order=np.argsort(times)
    ranks=np.sort(rng.choice(np.arange(1,n-2),size=signed_count,replace=False))
    signed_slots=order[ranks]
    sign=np.zeros(n,dtype=np.int64)
    sign[signed_slots]=1
    x_candidates=order[int(ranks[-1])+1:n-1]
    assert len(x_candidates)
    challenger=int(rng.choice(x_candidates))
    first=int(order[0])
    assert sign[challenger]==0 and sign[first]==0
    current=int(rng.integers(2))
    vals=rng.integers(2,size=n)
    vals[int(signed_slots[-1])]=current
    vals[challenger]=1-current
    public_bit=int(rng.integers(2))
    mode=SOURCES[(j//2)%4] if split=="ood" else "iid"
    if mode=="iid":
        sources=rng.integers(2,size=n)
    elif mode=="signed_source0":
        sources=1-sign
    elif mode=="signed_source1":
        sources=sign.copy()
    else:
        sources=np.arange(n)%2
    storage=rng.permutation(n)
    result=[]
    for variant in VARIANTS:
        signs=sign.copy()
        ts=times.copy()
        if variant.startswith("S_"):
            signs[challenger]=1
        if variant.endswith("_before"):
            ts[challenger],ts[first]=ts[first],ts[challenger]
        f=np.zeros((MAX_SLOTS,N_FEATURES),np.float32)
        contents=np.zeros(MAX_SLOTS,np.float32)
        valid=np.zeros(MAX_SLOTS,np.float32)
        f[:n,0]=signs[storage]
        f[:n,1]=ts[storage]/MAX_SLOTS
        f[:n,2]=f[:n,0]*f[:n,1]
        f[:n,3]=sources[storage]
        contents[:n]=2*vals[storage]-1
        valid[:n]=1.
        signed=np.flatnonzero((f[:,0]>.5)&(valid>.5))
        winner=int(signed[np.argmax(f[signed,1])])
        memory_bit=int(contents[winner]>0)
        assert memory_bit==((1-current) if variant=="S_after" else current)
        assert any(valid[z]>.5 and f[z,0]==0 and f[z,1]>f[winner,1]
                   for z in range(MAX_SLOTS))
        source,flip=task
        truth= (public_bit if source=="PUBLIC" else memory_bit)^flip
        row={
            "split":split,"replica":rep,"seed":seed,"base_world":j,
            "task":source,"flip":flip,"variant":variant,
            "n":n,"signed_count":int(signs.sum()),"source_mode":mode,
            "public_bit":public_bit,"truth":int(truth),"winner":winner,
            "features":f,"contents":contents,"mask":valid,
        }
        row["case_sha"]=hashlib.sha256(
            json.dumps({k:v for k,v in row.items() if not isinstance(v,np.ndarray)},
                       sort_keys=True).encode()
            +f.tobytes()+contents.tobytes()+valid.tobytes()
        ).hexdigest()
        result.append(row)
    by={x["variant"]:x for x in result}
    assert len({x["case_sha"] for x in result})==4
    if source=="MEMORY":
        assert len({x["truth"] for x in result})==2
        assert by["S_after"]["truth"]!=by["U_after"]["truth"]
    else:
        assert len({x["truth"] for x in result})==1
    return result

def dataset(split,rep):
    if split not in ("train","dev","ood") or rep not in range(4):
        raise ValueError("CONFIRM 26300000+ SEALED")
    tasks=TASKS if split!="ood" else (("MEMORY",1),)
    n={"train":256,"dev":64,"ood":128}[split]
    return [row for task in tasks for j in range(n)
            for row in _world(split,rep,j,task)]

def context(row):
    """Public metadata ONLY; never F4 contents, winner, truth or program ID."""
    return np.array([
        float(row["task"]=="MEMORY"),
        float(row["flip"]),
        float(row["public_bit"]),
    ],dtype=np.float32)

def evaluate_program(row,seq,selected_slot,xor_lookup,mode="normal"):
    """Real execution of ONE chosen program. Truth only at terminal reward."""
    working=None
    selected=None
    invalid=0
    acts=0
    seen=[]
    for op in seq:
        acts+=1
        seen.append(op)
        if op=="P":
            bit=row["public_bit"]
            if mode=="public_randomized" and row["task"]=="MEMORY":
                bit=1-bit
            working=int(bit)
        elif op=="S":
            selected=int(selected_slot)
        elif op=="R":
            if selected is None:
                invalid+=1
            else:
                values=row["contents"]
                if mode=="shuffle":
                    slots=np.flatnonzero(row["mask"]>.5)
                    values=values.copy()
                    values[slots]=values[slots[::-1]]
                working=int(values[selected]>0)
        elif op=="X":
            if working is None:
                invalid+=1
            elif mode!="x_disabled":
                working=int(xor_lookup[int(working)])
        else:
            raise AssertionError("operation not in frozen catalog")
    answer=0 if working is None else int(working)
    success=int(answer==row["truth"])
    observed=(2*success-1)-0.01*acts-0.10*invalid
    return {
        "answer":answer,"success":success,"return":float(observed),
        "invalid":invalid,"operations":acts,
        "read":int("R" in seen and seen.index("R")>seen.index("S")
                   if "S" in seen else 0),
        "executed":tuple(seen),
    }
