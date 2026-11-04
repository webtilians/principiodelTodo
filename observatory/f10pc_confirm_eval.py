"""F10-P-C: test ONLY original F10-P SHA-frozen memories on NEW 28300000+.

Mandatory original 8960-row full 4-intervention replay before generating
a single confirm world; never trains or updates a memory/model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from observatory import f10p_worlds as worlds
from observatory.f10p_events import CHANNELS, EventEffects
from observatory.f10n_memory import RewardMemory, EffectMemory, execute
from observatory.f10m_experiment import frozen_original_models, selection_cache

ARMS=CHANNELS+("B_reward","I_privileged")
MODES=("normal","shuffle","x_disabled","public_randomized")
PROGRAM_IDS={seq:i for i,seq in enumerate(worlds.PROGRAMS)}
F10P_RUN=41888234405
SKILLS_RUN=41685499939
SOURCE_COMMIT="9de6adb6ff0299b361d72258b9d988f832256201"
CONFIRM_BASE=28_300_000

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path,obj):
    Path(path).write_text(json.dumps(obj,sort_keys=True,separators=(",",":")))

def original_memories(folder,meta):
    result={}
    for arm in ARMS:
        original=meta["trained_memory_files"][arm]
        assert original["path"]==arm+".json"
        assert sha(folder/original["path"])==original["sha256"]
        snapshot=json.loads((folder/original["path"]).read_text())
        if arm in CHANNELS:
            result[arm]=EventEffects.from_snapshot(snapshot)
        elif arm=="B_reward":
            result[arm]=RewardMemory.from_snapshot(snapshot)
        else:
            result[arm]=EffectMemory.from_snapshot(snapshot)
        assert result[arm].episodes==98_304
    train=meta["public_trial_log"]
    assert sha(folder/train["path"])==train["sha256"]
    return result

def plan(arm,memories,source,flip,public_bit):
    model=memories[arm]
    if arm in CHANNELS:
        return model.plan(source,flip,public_bit)
    if arm=="B_reward":
        return model.rank(source,flip,public_bit,"additive_reward_table")
    return model.plan(source,flip,public_bit)

def reduced(actual):
    return {key:actual[key] for key in (
        "answer","success","reward","invalid","operations")}

def frozen_preflight(rep,folder,skills,g0,science,out):
    """Must complete BEFORE any 28300000 case/seed is created."""
    begin=time.perf_counter()
    meta=json.loads((folder/"frozen.json").read_text())
    assert meta["replica"]==rep
    assert meta["state"]=="ALL_MEMORY_SHA_FROZEN_BEFORE_DEV_OOD"
    assert meta["no_train_srx"] and meta["no_train_memory_flip1"]
    assert meta["confirm_28300000_sealed"]
    assert meta["trials_executed"]==98_304 and meta["train_cases"]==3072
    assert meta["epochs"]==16
    assert meta["all_arms_share_exact_programs"]
    assert len(meta["train_case_shas"])==3072
    assert len(set(meta["train_case_shas"]))==3072
    assert len(meta["program_counts"])==85
    assert sum(meta["program_counts"])==98_304
    assert meta["program_counts"][PROGRAM_IDS[("S","R","X")]]==0
    assert len(meta["trained_memory_files"])==7
    assert meta["original_skills"]["all_5_original_checkpoint_sha"]==(
        g0["replicas"][str(rep)]["model_sha256"])
    assert science["status"]=="F10P_OBSERVABLE_EFFECTS_PASS"
    assert science["gate_counts"]=={
        "G0_seen":3,"G1_unexecuted_composition":4,
        "G2_causal_F4_X":4,"G3_vs_reward":4,
        "G4_feedback_identification":4,
        "G5_E50_diagnostic":4,"G6_privileged_diagnostic":4}
    memories=original_memories(folder,meta)
    selector,xor,lut,skill_metadata=frozen_original_models(rep,skills,g0)
    assert tuple(lut)==(1,0)
    assert skill_metadata==meta["original_skills"]
    for name,digest in meta["original_skills"][
        "all_5_original_checkpoint_sha"].items():
        assert sha(skills/(name+".pt"))==digest
    stored=json.loads((folder/"eval_rows.json").read_text())
    assert len(stored)==7*(768+512)==8960
    original_case_shas=set(meta["train_case_shas"])
    case_map={}
    for split in ("dev","ood"):
        source=worlds.dataset(split,rep)
        assert len(source)==(768 if split=="dev" else 512)
        slots=selection_cache(source,selector)
        for row,slot in zip(source,slots):
            key=(split,row["seed"],row["variant"])
            assert key not in case_map
            assert row["case_sha"] not in original_case_shas
            original_case_shas.add(row["case_sha"])
            case_map[key]=(row,int(slot))
    assert len(original_case_shas)==3072+768+512
    decision_cache={}
    exact=0
    dev_counts={}
    for entry in stored:
        split=entry["split"]
        assert split in ("dev","ood")
        arm=entry["arm"]
        assert arm in ARMS
        key=(split,entry["seed"],entry["variant"])
        row,selected=case_map[key]
        assert all(entry[k]==row[k] for k in (
            "case_sha","base_world","task","flip","variant","n",
            "source_mode","public_bit","truth","winner","seed"))
        assert selected==entry["selected"]
        choice_key=(arm,row["task"],row["flip"],row["public_bit"])
        if choice_key not in decision_cache:
            decision_cache[choice_key]=plan(
                arm,memories,row["task"],row["flip"],row["public_bit"])
        program,score=decision_cache[choice_key]
        assert entry["program_id"]==PROGRAM_IDS[program]
        assert entry["program"]==("".join(program) or "EMPTY")
        assert abs(float(entry["predicted_score"])-score)<1e-10
        for mode in MODES:
            actual=execute(row,program,selected,lut,mode)
            assert entry["outcomes"][mode]==reduced(actual)
        if split=="dev" and arm=="E100":
            dkey=str(row["task"])+"_"+str(row["flip"])
            if dkey not in dev_counts:dev_counts[dkey]=[0,0]
            dev_counts[dkey][0]+=entry["outcomes"]["normal"]["success"]
            dev_counts[dkey][1]+=1
        exact+=1
    assert exact==8960 and len(dev_counts)==3
    dev={key:good/total for key,(good,total) in dev_counts.items()}
    assert all(total==256 for good,total in dev_counts.values())
    assert dev==science["replicas"][str(rep)]["dev_seen"]
    for name,digest in meta["original_skills"][
        "all_5_original_checkpoint_sha"].items():
        assert sha(skills/(name+".pt"))==digest
    for arm in ARMS:
        assert sha(folder/(arm+".json"))==(
            meta["trained_memory_files"][arm]["sha256"])
    preflight={
        "state":"ORIGINAL_F10P_REPLAY_PASS_BEFORE_ANY_CONFIRM_WORLD",
        "original_run":F10P_RUN,
        "original_source_commit":SOURCE_COMMIT,
        "original_skill_run":SKILLS_RUN,
        "replica":rep,
        "original_dev_ood_rows_replayed":exact,
        "original_interventions_replayed":exact*4,
        "original_unique_case_shas":len(original_case_shas),
        "original_case_shas_sha":hashlib.sha256(
            json.dumps(sorted(original_case_shas)).encode()).hexdigest(),
        "original_dev_E100":dev,
        "original_G0_pass":all(v>=.90 for v in dev.values()),
        "original_science_sha":sha(
            folder/"../../original-science/f10p-original-science.json")
            if (folder/"../../original-science/f10p-original-science.json").exists()
            else None,
        "original_frozen_meta_sha":sha(folder/"frozen.json"),
        "frozen_memories_sha256":{
            k:meta["trained_memory_files"][k]["sha256"] for k in ARMS},
        "original_checkpoint_sha256":meta["original_skills"][
            "all_5_original_checkpoint_sha"],
        "elapsed_seconds":time.perf_counter()-begin,
    }
    save(out/"original_replay_preconfirm.json",preflight)
    print("F10PC_ORIGINAL_REPLAY_PASS_PRE_CONFIRM",rep,exact,exact*4,dev,
          flush=True)
    return meta,memories,selector,lut,original_case_shas,preflight

def independently_new_seed_cases(rep,old_case_shas):
    """Only AFTER full original preflight: the same F10-P generator unchanged
    except in-memory replacement of OOD_BASE; original module is restored.
    """
    assert worlds.OOD_BASE==28_400_000
    worlds.OOD_BASE=CONFIRM_BASE
    try:
        cases=worlds.dataset("ood",rep)
    finally:
        worlds.OOD_BASE=28_400_000
    assert len(cases)==512
    assert {row["seed"] for row in cases}=={
        CONFIRM_BASE+10_000*rep+j for j in range(128)}
    assert {(r["task"],r["flip"]) for r in cases}=={("MEMORY",1)}
    case_shas={row["case_sha"] for row in cases}
    assert len(case_shas)==512
    assert not case_shas&old_case_shas
    assert all(row["n"] in (9,11) for row in cases)
    return cases

def confirm(rep,folder,skills,g0,science,out):
    out.mkdir(parents=True,exist_ok=False)
    meta,memories,selector,lut,old_shas,preflight=frozen_preflight(
        rep,folder,skills,g0,science,out)
    # Decisive temporal order: confirm data unsealed only BELOW this point.
    assert (out/"original_replay_preconfirm.json").exists()
    cases=independently_new_seed_cases(rep,old_shas)
    selected=selection_cache(cases,selector)
    rows=[]
    for arm in ARMS:
        for row,slot in zip(cases,selected):
            seq,score=plan(arm,memories,row["task"],row["flip"],
                           row["public_bit"])
            assert seq in worlds.PROGRAMS
            normal_and_interventions={
                m:reduced(execute(row,seq,int(slot),lut,m)) for m in MODES}
            rows.append({
                "replica":rep,"arm":arm,"split":"confirm",
                "seed":row["seed"],"base_world":row["base_world"],
                "variant":row["variant"],"n":row["n"],
                "source_mode":row["source_mode"],
                "task":row["task"],"flip":row["flip"],
                "public_bit":row["public_bit"],
                "case_sha":row["case_sha"],
                "truth":row["truth"],"winner":row["winner"],
                "selected":int(slot),
                "program":"".join(seq) or "EMPTY",
                "program_id":PROGRAM_IDS[seq],
                "predicted_score":float(score),
                "outcomes":normal_and_interventions,
            })
        print("F10PC_NEW_FROZEN_CONFIRM_ARM",rep,arm,512,flush=True)
    assert len(rows)==7*512==3584
    save(out/"confirm_rows.json",rows)
    # Immutable original inputs checked AFTER confirm evaluation, too.
    assert sha(folder/"frozen.json")==preflight["original_frozen_meta_sha"]
    for arm in ARMS:
        assert sha(folder/(arm+".json"))==(
            preflight["frozen_memories_sha256"][arm])
    for name,digest in preflight["original_checkpoint_sha256"].items():
        assert sha(skills/(name+".pt"))==digest
    confirmed={
        "state":"F10PC_CONFIRM_FROZEN_INPUTS_AND_PREDICTIONS_COMPLETE",
        "original_preflight_sha":sha(out/"original_replay_preconfirm.json"),
        "new_case_count":512,
        "new_world_count":128,
        "total_new_prediction_rows":len(rows),
        "new_case_shas_sha":hashlib.sha256(
            json.dumps(sorted(set(r["case_sha"] for r in rows))).encode()
        ).hexdigest(),
        "new_eval_rows_sha":sha(out/"confirm_rows.json"),
        "original_snapshot_sha_verified_after_confirm":True,
        "original_skill_sha_verified_after_confirm":True,
        "original_source_commit":SOURCE_COMMIT,
        "confirm_seed_base":CONFIRM_BASE,
        "prereg_sha":sha(Path(__file__).parents[1]/
            "docs/INFINITO_F10PC_PREREG_2026-11-04.md"),
        "confirm_eval_code_sha":sha(__file__),
    }
    save(out/"confirm_complete.json",confirmed)
    print("F10PC_CONFIRM_FROZEN_COMPLETE",rep,3584,flush=True)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--replica",type=int,required=True,
                        choices=(0,1,2,3))
    parser.add_argument("--original",required=True,type=Path)
    parser.add_argument("--skills",required=True,type=Path)
    parser.add_argument("--g0",required=True,type=Path)
    parser.add_argument("--science",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    a=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    g0=json.loads(a.g0.read_text())
    science=json.loads(a.science.read_text())
    confirm(a.replica,a.original,a.skills,g0,science,a.out)
