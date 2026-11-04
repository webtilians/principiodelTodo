"""F10-Q preregistered SQL fixture generator, separate from predictor.

These are synthetic records, written to ephemeral SQLite by the evaluator.
The MCP tool server itself has no write access and executes real SELECTs.
"""
from __future__ import annotations
import hashlib
import json
import sqlite3
import numpy as np

VARIANTS=("U_before","U_after","S_before","S_after")
SOURCES=("iid","source0","source1","alternating")
SPLIT_BASE={"train":29_100_000,"ood":29_400_000}
TRAIN_TASKS=(("PUBLIC",0),("PUBLIC",1),("MEMORY",0))

def fixtures(split,rep):
    if split not in SPLIT_BASE or rep not in range(4):
        raise ValueError("29300000+ confirmation is sealed")
    out=[]
    for j in range(128):
        seed=SPLIT_BASE[split]+10000*rep+j
        rng=np.random.default_rng(seed)
        b=int(rng.integers(2))
        public=int(rng.integers(2))
        n=9 if j%2==0 else 11
        mode=SOURCES[j%4]
        task,flip=TRAIN_TASKS[j%3] if split=="train" else ("MEMORY",1)
        for variant in VARIANTS:
            case_key=f"{split}-r{rep}-j{j}-{variant}"
            challenger_signed=variant in ("S_before","S_after")
            challenger_revision=45 if variant.endswith("_before") else 60
            current=1-b if variant=="S_after" else b
            invalid="missing" if split=="ood" and j%8==0 else (
                "null" if split=="ood" and j%8==1 else None)
            rows=[]
            for i in range(n):
                verified=0
                revision=i+1
                bit=int(rng.integers(2))
                if i==0:
                    verified=1
                    revision=50
                    bit=b
                elif i==1:
                    verified=int(challenger_signed)
                    revision=challenger_revision
                    bit=1-b
                elif i==2:
                    verified=0
                    revision=200
                    bit=1-current
                elif i in (3,4):
                    verified=1
                    revision=i+1
                if invalid=="missing":verified=0
                if invalid=="null" and (
                    (variant=="S_after" and i==1) or
                    (variant!="S_after" and i==0)):
                    bit=None
                rows.append({
                    "case_id":case_key,"entry_id":1+i,
                    "accredited":int(verified),"revision":revision,
                    "payload":bit,"source":mode,
                })
            truth=None if invalid else (
                (public if task=="PUBLIC" else current)^flip)
            identity={
                "split":split,"replica":rep,"seed":seed,
                "base_world":j,"variant":variant,"case_id":case_key,
                "source_mode":mode,"n":n,"task":task,"flip":flip,
                "public_bit":public,"invalid":invalid,
            }
            digest=hashlib.sha256(
                json.dumps({"record":identity,"rows":rows},
                           sort_keys=True).encode()).hexdigest()
            out.append({
                **identity,"truth":truth,"case_sha":digest,
                "rows":rows,
            })
    assert len(out)==512
    if split=="ood":
        assert sum(row["invalid"] is None for row in out)==384
        assert sum(row["invalid"] is not None for row in out)==128
    return out

def install(conn,cases):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS evidence "
        "(id INTEGER PRIMARY KEY, case_id TEXT NOT NULL, "
        "entry_id INTEGER, accredited INTEGER, revision INTEGER, "
        "payload INTEGER, source TEXT)")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_evidence "
        "ON evidence(case_id, accredited, revision)")
    values=[]
    for row in cases:
        values.extend((
            z["case_id"],z["entry_id"],z["accredited"],z["revision"],
            z["payload"],z["source"]) for z in row["rows"])
    conn.executemany(
        "INSERT INTO evidence(case_id, entry_id, accredited, revision, payload, source)"
        " VALUES(?,?,?,?,?,?)",values)
    conn.commit()
