"""F10-N prospective 4-arm experiment: same executed trial history A/B/C.

Only ORIGINAL F10-L skill checkpoints are loaded/frozen. No trainable
PyTorch policy is used: explicit episodic reward memory and operation-effect
memory are updated from REAL executed trials, never gold program labels.
D chooses exploratory programs using uncertainty in its own effect memory.

Nothing in TRAIN executes SRX; MEMORY+FLIP1 absent from TRAIN AND DEV.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json,time
from pathlib import Path
import numpy as np
import torch

from observatory.f10m_experiment import (
    frozen_original_models,selection_cache
)
from observatory.f10n_worlds import (
    dataset,PROGRAMS,TRAIN_PROGRAMS,HELDOUT_PROGRAM,context
)
from observatory.f10n_memory import EffectMemory,RewardMemory,execute

ARM_NAMES=("reward_table","additive_reward_table",
           "learned_effects_uniform","learned_effects_directed")
MODES=("normal","shuffle","x_disabled","public_randomized")
EPOCHS=16
TRAIN_CASES=3072
EPISODES_PER_STREAM=98304
SEED_BASE=266000
PROGRAM_INDEX={program:i for i,program in enumerate(PROGRAMS)}
TRAIN_IDS=np.array([PROGRAM_INDEX[s] for s in TRAIN_PROGRAMS],np.int64)
assert len(TRAIN_IDS)==84 and HELDOUT_PROGRAM not in TRAIN_PROGRAMS

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,record):
    Path(path).write_text(json.dumps(record,sort_keys=True,separators=(",",":")))

def log_record(stream,epoch,row,pid,result,phase):
    # NEVER log label (truth), selected index winner, or unexecuted results.
    item={
        "phase":phase,"epoch":epoch,"case_sha":row["case_sha"],
        "seed":row["seed"],"variant":row["variant"],
        "source":row["task"],"flip":row["flip"],"public_bit":row["public_bit"],
        "program_id":int(pid),"program":"".join(PROGRAMS[int(pid)]) or "EMPTY",
        "reward":result["reward"],"success":result["success"],
        "invalid":result["invalid"],"steps":result["trace"],
    }
    stream.write(json.dumps(item,separators=(",",":"))+"\n")

def choose_directed(memory,row,rng):
    # Exactly 16 uniformly sampled candidates, excluding all heldout
    # programs. Score uses ONLY transition count coverage, no hidden truth.
    ids=rng.choice(TRAIN_IDS,size=16,replace=False)
    scored=[
        (memory.novelty(PROGRAMS[int(pid)],row["task"],row["public_bit"]),
         int(pid))
        for pid in ids
    ]
    best=max(score for score,_ in scored)
    ties=[pid for score,pid in scored if abs(score-best)<1e-12]
    return int(rng.choice(ties))

def train(rep,out,original_models,g0):
    selector,xor,xor_lut,original=frozen_original_models(
        rep,original_models,g0)
    assert xor_lut==(1,0)
    rows=dataset("train",rep)
    assert len(rows)==TRAIN_CASES
    assert all((r["task"],r["flip"]) in (
        ("PUBLIC",0),("PUBLIC",1),("MEMORY",0)) for r in rows)
    assert len(set(r["case_sha"] for r in rows))==len(rows)
    choices=selection_cache(rows,selector)
    reward_memory=RewardMemory()
    effect_uniform=EffectMemory()
    effect_directed=EffectMemory()
    counts_uniform=np.zeros(len(PROGRAMS),np.int64)
    counts_directed=np.zeros(len(PROGRAMS),np.int64)
    total_uniform_success=0
    total_directed_success=0
    start_time=time.perf_counter()
    with gzip.open(out/"uniform_trials.jsonl.gz","wt") as f_uniform, \
         gzip.open(out/"directed_trials.jsonl.gz","wt") as f_directed:
        for epoch in range(EPOCHS):
            order=np.random.default_rng(SEED_BASE+100*rep+epoch).permutation(len(rows))
            ru=np.random.default_rng(SEED_BASE+10000+100*rep+epoch)
            rd=np.random.default_rng(SEED_BASE+20000+100*rep+epoch)
            epoch_uniform=0
            epoch_directed=0
            for i in order:
                row=rows[int(i)]
                selected=choices[int(i)]
                # A/B/C see EXACTLY the same two actually executed programs.
                for _ in range(2):
                    pid=int(ru.choice(TRAIN_IDS))
                    assert PROGRAMS[pid]!=HELDOUT_PROGRAM
                    result=execute(row,PROGRAMS[pid],selected,xor_lut)
                    reward_memory.observe(row,pid,result)
                    effect_uniform.observe(result,row["public_bit"])
                    log_record(f_uniform,epoch+1,row,pid,result,"uniform")
                    counts_uniform[pid]+=1
                    epoch_uniform+=result["success"]
                # D executes one uniform and one uncertainty-directed
                # real trial, same world/episode count as A/B/C.
                pid_uniform=int(rd.choice(TRAIN_IDS))
                result=execute(row,PROGRAMS[pid_uniform],selected,xor_lut)
                effect_directed.observe(result,row["public_bit"])
                log_record(f_directed,epoch+1,row,pid_uniform,result,"directed_uniform")
                counts_directed[pid_uniform]+=1
                epoch_directed+=result["success"]

                pid_directed=choose_directed(effect_directed,row,rd)
                assert PROGRAMS[pid_directed]!=HELDOUT_PROGRAM
                result=execute(row,PROGRAMS[pid_directed],selected,xor_lut)
                effect_directed.observe(result,row["public_bit"])
                log_record(f_directed,epoch+1,row,pid_directed,result,"directed_uncertainty")
                counts_directed[pid_directed]+=1
                epoch_directed+=result["success"]
            total_uniform_success+=epoch_uniform
            total_directed_success+=epoch_directed
            if epoch in (0,3,7,11,15):
                print("F10N_TRAIN",rep,"epoch",epoch+1,
                      "uniform_success",round(epoch_uniform/(2*TRAIN_CASES),4),
                      "directed_success",round(epoch_directed/(2*TRAIN_CASES),4),
                      "effect_signatures",len(effect_uniform.counts),
                      len(effect_directed.counts),flush=True)
    assert reward_memory.episodes==EPISODES_PER_STREAM
    assert effect_uniform.episodes==EPISODES_PER_STREAM
    assert effect_directed.episodes==EPISODES_PER_STREAM
    assert counts_uniform.sum()==counts_directed.sum()==EPISODES_PER_STREAM
    assert counts_uniform[PROGRAM_INDEX[HELDOUT_PROGRAM]]==0
    assert counts_directed[PROGRAM_INDEX[HELDOUT_PROGRAM]]==0
    memories={
        "reward_table":reward_memory.snapshot(),
        "learned_effects_uniform":effect_uniform.snapshot(),
        "learned_effects_directed":effect_directed.snapshot(),
    }
    for name,record in memories.items():
        write_json(out/(name+".json"),record)
    original_sha=original["all_5_original_checkpoint_sha"]
    for arm,s in original_sha.items():
        assert sha(original_models/(arm+".pt"))==s

    meta={
        "experiment":"F10-N explicit episodic reward vs observed operation effects",
        "status":"MEMORIES_SHA_FROZEN_BEFORE_DEV_OOD",
        "replica":rep,"original_skills":original,
        "original_sha_verified_after_training":True,
        "no_trainable_f10l_or_f10m_weights":True,
        "no_program_truth_or_winner_targets":True,
        "train_heldout_srx_count":0,
        "unseen_memory_flip1_never_trained":True,
        "confirm_26300000_sealed":True,
        "epochs":EPOCHS,
        "train_examples":TRAIN_CASES,
        "train_contexts":{"PUBLIC_0":1024,"PUBLIC_1":1024,"MEMORY_0":1024},
        "two_programs_executed_per_case_per_epoch":2,
        "episodes_per_condition":EPISODES_PER_STREAM,
        "train_case_shas":[r["case_sha"] for r in rows],
        "original_program_catalog_size":len(PROGRAMS),
        "train_program_catalog_size":len(TRAIN_PROGRAMS),
        "heldout_program":"SRX",
        "memory_files":{
            name:{"file":name+".json","sha256":sha(out/(name+".json"))}
            for name in memories
        },
        "experience_log_files":{
            name:{"file":name+"_trials.jsonl.gz",
                  "sha256":sha(out/(name+"_trials.jsonl.gz"))}
            for name in ("uniform","directed")
        },
        "attempt_counts":{
            "uniform":counts_uniform.tolist(),
            "directed":counts_directed.tolist(),
        },
        "mean_train_success":{
            "uniform":total_uniform_success/EPISODES_PER_STREAM,
            "directed":total_directed_success/EPISODES_PER_STREAM,
        },
        "transition_signatures":{
            "uniform":len(effect_uniform.counts),
            "directed":len(effect_directed.counts)
        },
        "observed_action_steps":{
            "uniform":effect_uniform.steps,
            "directed":effect_directed.steps,
        },
        "training_seconds":time.perf_counter()-start_time,
    }
    write_json(out/"training.json",meta)
    print("F10N_ORIGINAL_MEMORY_CHECKPOINTS_SHA_FROZEN",rep,
          meta["transition_signatures"],flush=True)
    return meta,selector,xor_lut

def evaluate(rep,out,meta,selector,xor_lut):
    """ONLY after all three memory snapshots and hashes are frozen."""
    for v in meta["memory_files"].values():
        assert sha(out/v["file"])==v["sha256"]
    memory=RewardMemory.from_snapshot(json.loads(
        (out/"reward_table.json").read_text()))
    effects_c=EffectMemory.from_snapshot(json.loads(
        (out/"learned_effects_uniform.json").read_text()))
    effects_d=EffectMemory.from_snapshot(json.loads(
        (out/"learned_effects_directed.json").read_text()))
    rows=[]
    for split in ("dev","ood"):
        cases=dataset(split,rep)
        assert len(cases)==(768 if split=="dev" else 512)
        choices=selection_cache(cases,selector)
        for arm in ARM_NAMES:
            for case,selected in zip(cases,choices):
                if arm in ("reward_table","additive_reward_table"):
                    program,score=memory.rank(
                        case["task"],case["flip"],case["public_bit"],arm)
                else:
                    em=effects_c if arm=="learned_effects_uniform" else effects_d
                    program,score=em.plan(case["task"],case["flip"],
                                          case["public_bit"],include_heldout=True)
                assert program in PROGRAMS
                outcomes={
                    m:execute(case,program,selected,xor_lut,m)
                    for m in MODES
                }
                oracle=execute(case,("S","R","X"),selected,xor_lut,"normal")
                rows.append({
                    "split":split,"replica":rep,"arm":arm,
                    "seed":case["seed"],"base_world":case["base_world"],
                    "variant":case["variant"],"case_sha":case["case_sha"],
                    "task":case["task"],"flip":case["flip"],
                    "public_bit":case["public_bit"],"source_mode":case["source_mode"],
                    "n":case["n"],"winner":case["winner"],
                    "selected":selected,"truth":case["truth"],
                    "program":"".join(program) or "EMPTY",
                    "program_id":PROGRAM_INDEX[program],
                    "predicted_score":score,
                    "oracle_srx_accuracy_diagnostic":oracle["success"],
                    "outcomes":{
                        m:{k:result[k] for k in ("success","answer","reward",
                                                 "invalid","operations")}
                        for m,result in outcomes.items()
                    },
                })
            print("F10N_FROZEN_EVAL",rep,split,arm,len(cases),flush=True)
    assert len(rows)==4*(768+512)
    write_json(out/"rows.json",rows)
    for v in meta["memory_files"].values():
        assert sha(out/v["file"])==v["sha256"]
    print("F10N_ALL_SHA_FROZEN_HOLDOUT_SAVED",rep,len(rows),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--replica",type=int,choices=range(4),required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--g0",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=False)
    manifest=json.loads(a.g0.read_text())
    meta,selector,xor_lut=train(a.replica,a.out,a.original,manifest)
    evaluate(a.replica,a.out,meta,selector,xor_lut)

if __name__=="__main__":
    main()
