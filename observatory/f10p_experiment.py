"""F10-P: single prospectively preregistered run, 4 independent replicas.

Actual experiences shared EXACTLY across all seven arms. Only EventEffects
public event output mask differs between E100, E50, E0, E_noX, E_shuffleX.
B shares the executed reward and I is an explicitly PRIVILEGED positive
control. Original F10-L skills frozen by SHA; no F10-N/F10-O retuning.
"""
from __future__ import annotations

import argparse,gzip,hashlib,json,time
from pathlib import Path

import numpy as np
import torch

from observatory.f10m_experiment import frozen_original_models,selection_cache
from observatory.f10n_memory import EffectMemory,RewardMemory,execute
from observatory.f10p_worlds import (
    dataset,PROGRAMS,TRAIN_PROGRAMS,HELDOUT_PROGRAM,context
)
from observatory.f10p_events import (
    CHANNELS,EventEffects,public_events,project
)

ARMS=CHANNELS+("B_reward","I_privileged")
MODES=("normal","shuffle","x_disabled","public_randomized")
PROGRAM_IDS={s:i for i,s in enumerate(PROGRAMS)}
TRAIN_IDS=np.array([PROGRAM_IDS[s] for s in TRAIN_PROGRAMS])
EPOCHS=16
TRAIN_CASES=3072
TRIALS_PER_REP=98304
assert len(PROGRAMS)==85 and len(TRAIN_IDS)==84
assert HELDOUT_PROGRAM==("S","R","X") and HELDOUT_PROGRAM not in TRAIN_PROGRAMS

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,data):
    Path(path).write_text(json.dumps(data,sort_keys=True,separators=(",",":")))

def train(rep,out,lroot,g0):
    selector,xor,lut,original=frozen_original_models(rep,lroot,g0)
    assert lut==(1,0)
    rows=dataset("train",rep)
    assert len(rows)==TRAIN_CASES
    assert {(r["task"],r["flip"]) for r in rows}=={
        ("PUBLIC",0),("PUBLIC",1),("MEMORY",0)}
    assert len({r["case_sha"] for r in rows})==len(rows)
    choices=selection_cache(rows,selector)
    learned={name:EventEffects() for name in CHANNELS}
    reward=RewardMemory()
    privileged=EffectMemory()
    n_program=np.zeros(len(PROGRAMS),dtype=np.int64)
    timer=time.perf_counter()
    mask_rng=np.random.default_rng(288000+rep)
    shuffle_rng=np.random.default_rng(289000+rep)
    with gzip.open(out/"executed_public_trials.jsonl.gz","wt") as stream:
        for epoch in range(EPOCHS):
            order=np.random.default_rng(285000+100*rep+epoch).permutation(
                TRAIN_CASES)
            rng=np.random.default_rng(286000+100*rep+epoch)
            for i in order:
                row=rows[int(i)]
                selected=int(choices[int(i)])
                for _ in range(2):
                    pid=int(rng.choice(TRAIN_IDS))
                    seq=PROGRAMS[pid]
                    assert seq!=HELDOUT_PROGRAM
                    actual=execute(row,seq,selected,lut)
                    events=public_events(actual)
                    projections={
                        arm:project(events,arm,mask_rng,shuffle_rng)
                        for arm in CHANNELS
                    }
                    for arm,projected in projections.items():
                        learned[arm].observe(projected,int(row["public_bit"]))
                    reward.observe(row,pid,actual)
                    privileged.observe(actual,int(row["public_bit"]))
                    n_program[pid]+=1
                    visible={
                        "epoch":epoch+1,
                        "seed":row["seed"],"variant":row["variant"],
                        "case_sha":row["case_sha"],
                        "source":row["task"],"flip":row["flip"],
                        "public_bit":row["public_bit"],
                        "program_id":pid,
                        "program":"".join(seq) or "EMPTY",
                        "answer":actual["answer"],"success":actual["success"],
                        "reward":actual["reward"],
                        "public_events":events,
                        "projected_events":{
                            name:projections[name] for name in CHANNELS},
                    }
                    # No actual["trace"], contents, truth, winning slot, before,
                    # after, or read_observed ever serialized here.
                    stream.write(json.dumps(visible,separators=(",",":"))+"\n")
        print("F10P_TRAIN_EXECUTED",rep,int(sum(n_program)),flush=True)
    assert n_program.sum()==TRIALS_PER_REP
    assert n_program[PROGRAM_IDS[HELDOUT_PROGRAM]]==0
    assert all(obj.episodes==TRIALS_PER_REP for obj in learned.values())
    assert reward.episodes==privileged.episodes==TRIALS_PER_REP
    snapshots={
        **{k:v.snapshot() for k,v in learned.items()},
        "B_reward":reward.snapshot(),
        "I_privileged":privileged.snapshot(),
    }
    for arm,state in snapshots.items():
        save(out/(arm+".json"),state)
    for name,digest in original["all_5_original_checkpoint_sha"].items():
        assert sha(lroot/(name+".pt"))==digest
    meta={
        "experiment":"F10-P fixed public tool output observation ladder",
        "state":"ALL_MEMORY_SHA_FROZEN_BEFORE_DEV_OOD",
        "replica":rep,"original_skills":original,
        "no_train_srx":True,
        "no_train_memory_flip1":True,
        "confirm_28300000_sealed":True,
        "train_cases":TRAIN_CASES,
        "epochs":EPOCHS,"trials_executed":TRIALS_PER_REP,
        "all_arms_share_exact_programs":True,
        "train_case_shas":[r["case_sha"] for r in rows],
        "program_counts":n_program.tolist(),
        "trained_memory_files":{
            name:{"path":name+".json","sha256":sha(out/(name+".json"))}
            for name in snapshots
        },
        "public_trial_log":{
            "path":"executed_public_trials.jsonl.gz",
            "sha256":sha(out/"executed_public_trials.jsonl.gz"),
        },
        "prereg_sha256":sha(
            Path(__file__).parents[1]/"docs/INFINITO_F10P_PREREG_2026-11-04.md"),
        "code_sha256":sha(__file__),
        "train_seconds":time.perf_counter()-timer,
    }
    save(out/"frozen.json",meta)
    print("F10P_MEMORY_SHA_FROZEN_PRE_HOLDOUT",rep,
          {arm:learned[arm].events_seen for arm in CHANNELS},flush=True)
    return meta,selector,lut

def evaluate(rep,out,meta,selector,lut):
    for info in meta["trained_memory_files"].values():
        assert sha(out/info["path"])==info["sha256"]
    event_memories={
        arm:EventEffects.from_snapshot(
            json.loads((out/(arm+".json")).read_text()))
        for arm in CHANNELS
    }
    reward=RewardMemory.from_snapshot(
        json.loads((out/"B_reward.json").read_text()))
    privileged=EffectMemory.from_snapshot(
        json.loads((out/"I_privileged.json").read_text()))
    rows=[]
    for split in ("dev","ood"):
        cases=dataset(split,rep)
        assert len(cases)==(768 if split=="dev" else 512)
        selected_slots=selection_cache(cases,selector)
        for arm in ARMS:
            for row,selected in zip(cases,selected_slots):
                if arm in CHANNELS:
                    program,estimated=event_memories[arm].plan(
                        row["task"],row["flip"],row["public_bit"])
                elif arm=="B_reward":
                    program,estimated=reward.rank(
                        row["task"],row["flip"],row["public_bit"],
                        "additive_reward_table")
                else:
                    program,estimated=privileged.plan(
                        row["task"],row["flip"],row["public_bit"])
                assert program in PROGRAMS
                outcomes={
                    m:execute(row,program,int(selected),lut,m)
                    for m in MODES
                }
                rows.append({
                    "replica":rep,"arm":arm,"split":split,
                    "seed":row["seed"],"base_world":row["base_world"],
                    "variant":row["variant"],"case_sha":row["case_sha"],
                    "task":row["task"],"flip":row["flip"],
                    "source_mode":row["source_mode"],"n":row["n"],
                    "public_bit":row["public_bit"],"truth":row["truth"],
                    "winner":row["winner"],"selected":int(selected),
                    "program":"".join(program) or "EMPTY",
                    "program_id":PROGRAM_IDS[program],
                    "predicted_score":estimated,
                    "outcomes":{
                        m:{k:v[k] for k in (
                            "answer","success","reward","invalid","operations")}
                        for m,v in outcomes.items()
                    }
                })
            print("F10P_FROZEN_EVAL",rep,split,arm,len(cases),flush=True)
    assert len(rows)==len(ARMS)*(768+512)
    save(out/"eval_rows.json",rows)
    assert all(sha(out/v["path"])==v["sha256"]
               for v in meta["trained_memory_files"].values())
    assert sha(out/meta["public_trial_log"]["path"])==(
        meta["public_trial_log"]["sha256"])
    print("F10P_FINISHED_ORIGINAL_FROZEN",rep,len(rows),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--replica",type=int,required=True,choices=range(4))
    ap.add_argument("--original",type=Path,required=True)
    ap.add_argument("--g0",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=False)
    original=json.loads(a.g0.read_text())
    meta,selector,lut=train(a.replica,a.out,a.original,original)
    evaluate(a.replica,a.out,meta,selector,lut)
