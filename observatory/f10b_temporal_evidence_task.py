"""F10-B: synthetic PUBLIC provenance/event-version task, no evaluator label input.

A whole world is exactly 60 observed records, then four distinct queries. All
four queries share bitwise-identical histories. F4 is an observation-only FIFO
log; the correct answer is reconstructed from PUBLIC signed reports only.
"""
from __future__ import annotations

import hashlib
import numpy as np

KEYS = 24
EVENT_FEATURES = KEYS + 1 + 1 + 1 + 2
PUBLIC_FEATURES = EVENT_FEATURES + 1
CAPACITY = 64
WORLD_EVENTS = 60
CELLS = (("signed","stable"),("unsigned","stable"),
         ("signed","revised"),("unsigned","revised"))
TRAIN_BASE = 13_100_000
DEV_BASE = 13_200_000
CONFIRM_BASE = 13_300_000
TRAIN_WORLDS = 256
DEV_WORLDS = 64

def seed_for(split,index,j):
    if index not in range(4) or split not in ("train","dev"):
        raise ValueError("Only new F10-B TRAIN/DEV (no confirm executor)")
    count = TRAIN_WORLDS if split=="train" else DEV_WORLDS
    if j not in range(count):
        raise ValueError("F10-B world index outside preregistered range")
    return (TRAIN_BASE if split=="train" else DEV_BASE) + 10_000*index+j

def event(key,value,signed,source,t):
    if not 0 <= key < KEYS or value not in (0,1) or signed not in (0,1) or source not in (0,1):
        raise ValueError("Invalid publicly observed evidence")
    row = np.zeros(PUBLIC_FEATURES,dtype=np.float32)
    row[key] = 1.
    row[KEYS] = value
    row[KEYS+1] = signed
    row[KEYS+2] = t / float(WORLD_EVENTS-1)
    row[KEYS+3+source] = 1.
    return row

class PublicEventF4:
    """Finite FIFO of records from already observed PUBLIC event frames only."""
    def __init__(self):
        self.records=[]
    def observe(self,frame):
        raw=np.asarray(frame,dtype=np.float32)
        if raw.shape!=(PUBLIC_FEATURES,):
            raise AssertionError("Observation not publicly typed")
        if raw[-1]!=0:
            raise AssertionError("Future query must never be written as an event")
        if not np.isclose(raw[:KEYS].sum(),1.) or not np.isclose(raw[KEYS+3:KEYS+5].sum(),1.):
            raise AssertionError("Bad public key/source encoding")
        self.records.append(raw[:-1].copy())
        if len(self.records)>CAPACITY:
            self.records.pop(0)
    def readout(self):
        mem=np.zeros((CAPACITY,EVENT_FEATURES),dtype=np.float32)
        if self.records:
            mem[-len(self.records):]=np.stack(self.records)
        return mem

def _reports(seed):
    rng=np.random.default_rng(seed)
    perm=[int(x) for x in rng.permutation(KEYS)]
    group_keys={cell:perm[n*6:(n+1)*6] for n,cell in enumerate(CELLS)}
    bits=[int(x) for x in rng.integers(0,2,size=KEYS)]
    initial=[(key,bits[key],1,int(rng.integers(0,2))) for key in rng.permutation(KEYS)]
    todo={}
    for (signature,change),keys in group_keys.items():
        for key in keys:
            steps=[]
            current=bits[key]
            if change=="revised":
                current=1-current
                steps.append((key,current,1,int(rng.integers(0,2))))
            if signature=="signed":
                steps.append((key,current,1,int(rng.integers(0,2))))
            else:
                steps.append((key,1-current,0,int(rng.integers(0,2))))
            todo[key]=steps
    following=[]
    while todo:
        key=int(rng.choice(list(todo.keys())))
        following.append(todo[key].pop(0))
        if not todo[key]: del todo[key]
    reports=initial+following
    assert len(reports)==WORLD_EVENTS,(len(reports),seed)
    return reports,group_keys,rng

def _correct_from_public(events,key):
    """The observer's ground truth comes ONLY from signed PUBLIC observations."""
    best=None
    for x in events:
        if x[key]>0 and x[KEYS+1]==1:
            best=int(x[KEYS])
    if best is None:
        raise AssertionError("No signed evidence for requested key")
    return best

def world(seed):
    reports,groups,rng=_reports(seed)
    writer=PublicEventF4()
    events=[]
    for t,(key,value,signed,source) in enumerate(reports):
        frame=event(key,value,signed,source,t)
        writer.observe(frame)
        events.append(frame)
    public=np.stack(events)
    mem=writer.readout()
    assert public.shape==(WORLD_EVENTS,PUBLIC_FEATURES)
    assert mem.shape==(CAPACITY,EVENT_FEATURES)
    assert np.array_equal(mem[-WORLD_EVENTS:,:],public[:,:-1])
    digest=hashlib.sha256(public.tobytes()).hexdigest()
    rows=[]
    for signature,change in CELLS:
        key=int(rng.choice(groups[(signature,change)]))
        query=np.zeros(PUBLIC_FEATURES,dtype=np.float32)
        query[key]=1.
        query[-1]=1.
        hist=np.concatenate((public,query[None,:]))
        truth=_correct_from_public(public,key)
        # Simulator-only strata must be derivable from PUBLIC history.
        selected=[x for x in public if x[key]>0]
        assert bool(selected[-1][KEYS+1])==(signature=="signed")
        valid=[int(x[KEYS]) for x in selected if x[KEYS+1]==1]
        assert len(valid)>0
        assert (valid[-1]!=valid[0])==(change=="revised")
        assert truth==valid[-1]
        rows.append({
            "seed":seed,"cell":signature+"|"+change,"key":key,
            "truth":truth,
            "public":hist.copy(),"memory":mem.copy(),
            "query":query[:KEYS].copy(),
            "shared_history_sha256":digest,
            "latest_unsigned":signature=="unsigned",
            "revised":change=="revised",
        })
    assert len(rows)==4
    assert len({r["key"] for r in rows})==4
    assert len({r["shared_history_sha256"] for r in rows})==1
    assert all(np.array_equal(rows[0]["public"][:-1],r["public"][:-1]) for r in rows)
    return rows

def stale_memory(case):
    """Observer-only stale display: remove later signed entries for query key."""
    out=case["memory"].copy()
    key=case["key"]
    seen=False
    for i in range(CAPACITY):
        if out[i,key]>0 and out[i,KEYS+1]==1:
            if seen:
                out[i,:]=0.
            seen=True
    assert seen
    return out

def forged_credential(case):
    """Observer-only outside-distribution attack on public signature readout."""
    out=case["memory"].copy()
    if case["latest_unsigned"]:
        which=[i for i in range(CAPACITY) if out[i,case["key"]]>0]
        assert which
        assert out[which[-1],KEYS+1]==0
        out[which[-1],KEYS+1]=1.
    return out
