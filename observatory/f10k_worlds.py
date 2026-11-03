"""F10-K: disjoint new worlds for reward-only primitive-to-composition transfer.

Task P: public bit XOR flip, no F4; I: single F4 bit, flip=0;
S: select latest signed among 5/7 reports, flip=0;
M: *never trained* version selection among 9/11 reports + flip=1.

The actor's selector sees ONLY signs, timestamps and source, never bits;
the shared processor receives ONLY the chosen bit and flip, never task type.
"""
from __future__ import annotations
import hashlib
import json
import numpy as np

TYPES=("P","I","S","M")
VARIANTS=("U_before","U_after","S_before","S_after")
MAX_SLOTS=12
FEAT=4
SPLITS={"train":22_100_000,"dev":22_200_000,"ood":22_400_000}
COUNTS={"train":512,"dev":64,"ood":128}
TYPE_OFFSETS={"P":0,"I":2000,"S":4000,"M":0}
SOURCE_MODES=("iid","signed_source0","signed_source1","alternating")

def seed_for(split,replica,task,j):
    if split not in SPLITS or replica not in range(4):
        raise ValueError("F10-K CONFIRM 22300000+ is sealed")
    if (split=="ood")!=(task=="M") or task not in TYPES:
        raise ValueError("F10-K composition only in OOD; primitives only in TRAIN/DEV")
    if j not in range(COUNTS[split]):
        raise ValueError("index outside preregistered split")
    return SPLITS[split]+10000*replica+TYPE_OFFSETS[task]+j

def _case(seed,split,replica,task,variant,features,values,valid,public_bit,flip,
          winner,truth,source_mode):
    payload={
        "seed":seed,"split":split,"replica":replica,"task":task,
        "variant":variant,"source_mode":source_mode,
        "n":int(np.sum(valid)),"winner":int(winner),
        "public_bit":int(public_bit),"flip":int(flip),"truth":int(truth),
    }
    payload["features"]=features.astype(np.float32)
    payload["values"]=values.astype(np.float32)
    payload["valid"]=valid.astype(np.float32)
    # Case identifier: hashes include seed/variant AND observations. Identical
    # small-bit primitive observations may repeat; not independent unique states.
    h=hashlib.sha256(
        json.dumps({k:v for k,v in payload.items() if not isinstance(v,np.ndarray)},
                   sort_keys=True).encode()
        +payload["features"].tobytes()+payload["values"].tobytes()
        +payload["valid"].tobytes()
    ).hexdigest()
    payload["case_sha"]=h
    return payload

def world(split,replica,task,j):
    seed=seed_for(split,replica,task,j)
    rng=np.random.default_rng(seed)
    rows=[]
    if task in ("P","I"):
        for vi,variant in enumerate(VARIANTS):
            f=np.zeros((MAX_SLOTS,FEAT),dtype=np.float32)
            v=np.zeros(MAX_SLOTS,dtype=np.float32)
            valid=np.zeros(MAX_SLOTS,dtype=np.float32)
            if task=="P":
                public_bit=(vi//2)%2
                flip=vi%2
                winner=0
                truth=public_bit^flip
            else:
                public_bit=int(rng.integers(2))
                flip=0
                slot=(2*vi+int(rng.integers(2)))%MAX_SLOTS
                winner=slot
                valid[slot]=1.
                f[slot]=[1.,.25+(vi%2)*.5, .25+(vi%2)*.5,float((vi//2)%2)]
                v[slot]=float((vi//2)*2-1)
                truth=vi//2
            rows.append(_case(seed,split,replica,task,variant,f,v,valid,
                              public_bit,flip,winner,truth,"iid"))
        assert len(rows)==4 and len({r["case_sha"] for r in rows})==4
        return rows

    n=(9 if j%2==0 else 11) if task=="M" else (5 if j%2==0 else 7)
    n_signed=(4 if j%2==0 else 5) if task=="M" else (2 if j%2==0 else 3)
    assert n_signed<=n-3
    bit=int(rng.integers(2))
    public_bit=int(rng.integers(2))  # independent decoy, not used for memory.
    timestamp=rng.permutation(n)
    ordering=np.argsort(timestamp)
    # At least one unsigned before X and one newer unsigned after X.
    signed_ranks=np.sort(rng.choice(np.arange(1,n-2),size=n_signed,replace=False))
    signed=ordering[signed_ranks]
    sign=np.zeros(n,dtype=np.int64)
    sign[signed]=1
    last_rank=int(signed_ranks[-1])
    x_slot=int(rng.choice(ordering[last_rank+1:n-1]))
    before_slot=int(ordering[0])
    assert not sign[x_slot] and not sign[before_slot]
    values=np.full(n,1-bit,dtype=np.int64)
    values[signed[-1]]=bit
    values[x_slot]=1-bit
    # Every report has opposite content from the current last-signed winner;
    # this creates a clear reward signal for selecting an eligible slot.
    assert sum(values==bit)==1
    if task=="M":
        source_mode=SOURCE_MODES[(j//2)%4]
    else:
        source_mode="iid"
    if source_mode=="iid":
        sources=rng.integers(2,size=n)
    elif source_mode=="signed_source0":
        sources=1-sign
    elif source_mode=="signed_source1":
        sources=sign.copy()
    else:
        sources=np.arange(n)%2
    storage=rng.permutation(n)
    for variant in VARIANTS:
        sig=sign.copy()
        ts=timestamp.copy()
        if variant.startswith("S_"):
            sig[x_slot]=1
        if variant.endswith("_before"):
            ts[x_slot],ts[before_slot]=ts[before_slot],ts[x_slot]
        # The "after" candidate is physically not last; order of storage differs.
        f=np.zeros((MAX_SLOTS,FEAT),np.float32)
        v=np.zeros(MAX_SLOTS,np.float32)
        valid=np.zeros(MAX_SLOTS,np.float32)
        f[:n,0]=sig[storage]
        f[:n,1]=ts[storage]/MAX_SLOTS
        f[:n,2]=f[:n,0]*f[:n,1]
        f[:n,3]=sources[storage]
        v[:n]=(values[storage]*2-1).astype(np.float32)
        valid[:n]=1.
        signed_valid=(f[:,0]>.5)&(valid>.5)
        winner=int(np.argmax(np.where(signed_valid,f[:,1],-1)))
        status=int(v[winner]>0)
        flip=1 if task=="M" else 0
        truth=status^flip
        assert status==((1-bit) if variant=="S_after" else bit), (task,j,variant)
        # Even if X becomes signed, an unsigned report remains newer.
        assert any(valid[i]>.5 and f[i,0]==0 and f[i,1]>f[winner,1]
                   for i in range(MAX_SLOTS))
        rows.append(_case(seed,split,replica,task,variant,f,v,valid,
                          public_bit,flip,winner,truth,source_mode))
    assert len(rows)==4 and len({r["case_sha"] for r in rows})==4
    assert rows[0]["truth"]==rows[1]["truth"]==rows[2]["truth"]
    assert rows[3]["truth"]!=rows[1]["truth"]
    return rows

def dataset(split,replica):
    tasks=("M",) if split=="ood" else ("P","I","S")
    return [r for task in tasks for j in range(COUNTS[split])
            for r in world(split,replica,task,j)]
