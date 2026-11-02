"""F10-D: EVALUATE FROZEN F10-C/F10-B CHECKPOINTS ONLY on 15200000+ worlds.

No optimizer, no training, no opening old worlds. Same publicly observed GRU
history and query under READ, SKIP, stale and forged F4 readouts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from observatory.f9_external_minigrid import file_sha
from observatory.f10b_temporal_evidence_task import stale_memory,forged_credential
from observatory.f10b_train_evidence_actors import EvidenceActor
from observatory.f10c_relational_actor_training import RelationalActor
from observatory.f10d_structural_transfer_worlds import (
    dev_seed,world,CELLS,SOURCE_MODES,CARDINALITIES,DEV_WORLDS,
    last_signed_public,last_any_observed,majority_observed,
)

MODES=("relational_read","relational_skip","relational_stale",
       "relational_forged","unary_read","unary_skip",
       "recurrent_only","f10b_frozen_read","f10b_frozen_recurrent",
       "public_last_signed_rule","public_last_any","public_majority")

def load_models(f10c_root,f10b_root,index):
    cdir=Path(f10c_root)
    bdir=Path(f10b_root)
    c_report=json.loads((cdir/"report.json").read_text())
    b_report=json.loads((bdir/"report.json").read_text())
    assert c_report["index"]==index and c_report["phase"]=="TRAIN_ONLY"
    assert c_report["train_worlds"]==512 and c_report["train_cases"]==2048
    assert not c_report["confirm_14300000_opened"]
    assert set(c_report["arms"])=={
      "relational_read","unary_read_control","recurrent_only_control"}
    assert b_report["index"]==index and b_report["stage"]=="TRAIN_ONLY"
    assert b_report["train_worlds"]==256 and b_report["train_cases"]==1024
    assert b_report["no_confirm_13300000"]
    models={}
    for prefix,root,report,items in (
        ("f10c",cdir,c_report,(
            ("relational_read",RelationalActor),
            ("unary_read_control",EvidenceActor),
            ("recurrent_only_control",EvidenceActor))),
        ("f10b",bdir,b_report,(
            ("memory_available",EvidenceActor),
            ("recurrent_only",EvidenceActor)))):
        for name,cls in items:
            metadata=report["arms"][name]
            checkpoint=root/metadata["checkpoint"]
            expected=metadata["sha256"]
            assert file_sha(checkpoint)==expected,("Frozen model mismatch",checkpoint)
            actor=cls()
            actor.load_state_dict(torch.load(checkpoint,map_location="cpu",weights_only=True))
            actor.eval()
            for p in actor.parameters():
                p.requires_grad_(False)
            models[prefix+"_"+name]=actor
    prov={
        "f10c_original_run":41447171697,
        "f10b_original_run":41443064754,
        "f10c_report_sha256":file_sha(cdir/"report.json"),
        "f10b_report_sha256":file_sha(bdir/"report.json"),
        "frozen_checkpoint_sha256":{
          pre+"_"+name:r["arms"][name]["sha256"]
          for pre,r in (("f10c",c_report),("f10b",b_report))
          for name in r["arms"]
        }
    }
    assert len(models)==5
    return models,prov

def prob(actor,case,mem):
    with torch.no_grad():
        hist=torch.from_numpy(case["public"][None])
        f4=torch.from_numpy(mem[None])
        return float(torch.sigmoid(actor(hist,f4)).item())

def is_correct(row,mode):
    return int(row["scores"][mode]>=.5)==row["truth"]

def summarize(rows):
    out={}
    for mode in MODES:
        out[mode]=sum(is_correct(row,mode) for row in rows)
    deltas=[int(is_correct(r,"relational_read"))-
            int(is_correct(r,"relational_skip")) for r in rows]
    return {
        "cases":len(rows),"correct":out,
        "D_relational":{str(n):sum(x==n for x in deltas) for n in (-1,0,1)},
    }

def cluster_ci(rows,positive,negative):
    groups={}
    for row in rows: groups.setdefault(row["seed"],[]).append(row)
    assert len(groups)==DEV_WORLDS and all(len(c)==4 for c in groups.values())
    deltas=np.array([
        sum(int(is_correct(r,positive))-int(is_correct(r,negative))
            for r in cases)/4 for cases in groups.values()],dtype=float)
    rng=np.random.default_rng(158000)
    bs=np.sort(np.array([
        np.mean(rng.choice(deltas,len(deltas),replace=True))
        for _ in range(2000)]))
    return {
       "estimate":float(np.mean(deltas)),
       "ci95":[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
       "unit":"world","samples":2000
    }

def execute(args):
    if args.index not in range(4):raise ValueError("Only 4 frozen original replicas")
    torch.set_num_threads(1)
    models,prov=load_models(args.f10c_source,args.f10b_source,args.index)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for j in range(DEV_WORLDS):
        seed=dev_seed(args.index,j)
        cases=world(seed,args.index,j)
        assert len({c["history_sha256"] for c in cases})==1
        for case in cases:
            m=case["memory"];blank=np.zeros_like(m)
            relational=models["f10c_relational_read"]
            unary=models["f10c_unary_read_control"]
            scores={
                "relational_read":prob(relational,case,m),
                "relational_skip":prob(relational,case,blank),
                "relational_stale":prob(relational,case,stale_memory(case)),
                "relational_forged":prob(relational,case,forged_credential(case)),
                "unary_read":prob(unary,case,m),
                "unary_skip":prob(unary,case,blank),
                "recurrent_only":prob(models["f10c_recurrent_only_control"],case,blank),
                "f10b_frozen_read":prob(models["f10b_memory_available"],case,m),
                "f10b_frozen_recurrent":prob(models["f10b_recurrent_only"],case,blank),
                "public_last_signed_rule":float(last_signed_public(case)),
                "public_last_any":float(last_any_observed(case)),
                "public_majority":float(majority_observed(case)),
            }
            assert set(scores)==set(MODES)
            assert all(0<=v<=1 for v in scores.values())
            assert int(scores["public_last_signed_rule"])==case["truth"]
            rows.append({
                "seed":seed,"cell":case["cell"],"key":case["key"],
                "truth":case["truth"],"record_count":case["record_count"],
                "source_mode":case["source_mode"],
                "history_sha256":case["history_sha256"],
                "scores":scores,
            })
        if j%16==0:print("F10D_NEW_UNSEEN_WORLD",args.index,j,seed,flush=True)
    assert len(rows)==256
    overall=summarize(rows)
    overall["read_skip_world_bootstrap"]=cluster_ci(
       rows,"relational_read","relational_skip")
    overall["relational_vs_unary_world_bootstrap"]=cluster_ci(
       rows,"relational_read","unary_read")
    by_cell={cell:summarize([r for r in rows if r["cell"]==cell]) for cell in CELLS}
    by_count={str(k):summarize([r for r in rows if r["record_count"]==k])
              for k in CARDINALITIES}
    by_source={mode:summarize([r for r in rows if r["source_mode"]==mode])
               for mode in SOURCE_MODES}
    cross={
        cell:{
            str(k):summarize([r for r in rows
                              if r["cell"]==cell and r["record_count"]==k])
            for k in CARDINALITIES}
        for cell in CELLS
    }
    assert all(v["cases"]==64 for v in by_cell.values())
    assert all(v["cases"]==64 for v in by_count.values())
    assert all(v["cases"]==64 for v in by_source.values())
    assert all(v["cases"]==16 for c in cross.values() for v in c.values())
    report={
        "experiment":"F10-D frozen policy transfer under provenance and version shifts",
        "stage":"STRUCTURAL_DEV_ONLY","index":args.index,
        "dev_world_seeds":[dev_seed(args.index,j) for j in range(DEV_WORLDS)],
        "worlds":DEV_WORLDS,"cases":len(rows),
        "frozen_original_checkpoints":True,
        "training_performed":False,"new_reader_trained":False,
        "original_confirmations_opened":False,
        "F10D_15300000_opened":False,
        "labels_only_from_public_signed_records":True,
        "same_history_for_four_queries":True,
        "recurrent_state_fixed_across_f4_interventions":True,
        "provenance":prov,
        "overall":overall,"by_cell":by_cell,"by_count":by_count,
        "by_source":by_source,"cross_cell_count":cross,"rows":rows,
    }
    (out/"dev-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("F10D_DEV_RESULT",args.index,json.dumps({
        "correct":overall["correct"],
        "read_skip_ci":overall["read_skip_world_bootstrap"],
        "relational_vs_unary_ci":overall["relational_vs_unary_world_bootstrap"],
        "by_count":{k:v["correct"] for k,v in by_count.items()},
        "by_cell":{k:v["correct"] for k,v in by_cell.items()},
        "by_source":{k:v["correct"] for k,v in by_source.items()},
    }),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--f10c-source",type=Path,required=True)
    p.add_argument("--f10b-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    execute(p.parse_args())
