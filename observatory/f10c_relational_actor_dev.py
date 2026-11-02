"""F10-C NEW independent DEV, compare trained controls and F10-B FROZEN.

No F10-B train/dev worlds opened; only load their original checkpoint bytes.
READ/SKIP/stale/forgery use the SAME relational actor and same GRU state.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from observatory.f9_external_minigrid import file_sha
from observatory.f10b_train_evidence_actors import EvidenceActor
from observatory.f10b_temporal_evidence_task import stale_memory,forged_credential
from observatory.f10c_relational_actor_training import RelationalActor,ARMS
from observatory.f10c_multi_version_worlds import world,seed_for,DEV_WORLDS,GROUP_NAMES

MODES=("read_clean","skip","read_stale","read_forged",
       "unary_retrained","recurrent_retrained",
       "f10b_frozen_read","f10b_frozen_recurrent")

def probability(model,case,memory):
    with torch.no_grad():
        x=torch.from_numpy(case["public"][None])
        m=torch.from_numpy(memory[None])
        return float(torch.sigmoid(model(x,m)).item())

def correct(c,name):
    return int((c["p"][name]>=.5)==c["truth"])

def by_rows(rows):
    scores={name:sum(correct(c,name) for c in rows) for name in MODES}
    effects=[correct(c,"read_clean")-correct(c,"skip") for c in rows]
    return {
        "cases":len(rows),"correct":scores,
        "D_clean":{str(v):sum(d==v for d in effects) for v in (-1,0,1)},
    }

def world_bootstrap(rows):
    groups={}
    for c in rows:groups.setdefault(c["seed"],[]).append(c)
    assert len(groups)==DEV_WORLDS and all(len(r)==4 for r in groups.values())
    changes=np.array([sum(correct(c,"read_clean")-correct(c,"skip")
                          for c in cases)/4 for cases in groups.values()])
    rng=np.random.default_rng(187000)
    res=np.sort(np.array([np.mean(rng.choice(changes,size=len(changes),replace=True))
                          for _ in range(2000)]))
    return {"mean":float(changes.mean()),
            "ci95":[float(np.quantile(res,.025)),float(np.quantile(res,.975))],
            "unit":"world","draws":2000}

def load_pair(args):
    trained=Path(args.trained)
    report=json.loads((trained/"report.json").read_text())
    assert report["phase"]=="TRAIN_ONLY" and report["index"]==args.index
    assert report["train_worlds"]==512 and report["train_cases"]==2048
    assert set(report["arms"])==set(ARMS)
    new={}
    for arm in ARMS:
        path=trained/report["arms"][arm]["checkpoint"]
        assert file_sha(path)==report["arms"][arm]["sha256"]
        model=RelationalActor() if arm=="relational_read" else EvidenceActor()
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        model.eval()
        new[arm]=model
    frozen_dir=Path(args.frozen_f10b)
    old=json.loads((frozen_dir/"report.json").read_text())
    assert old["stage"]=="TRAIN_ONLY" and old["index"]==args.index
    assert old["train_worlds"]==256 and old["train_cases"]==1024
    assert old["no_confirm_13300000"] and old["no_old_worlds"]
    assert set(old["arms"])=={"memory_available","recurrent_only"}
    frozen={}
    for arm in ("memory_available","recurrent_only"):
        path=frozen_dir/old["arms"][arm]["checkpoint"]
        assert file_sha(path)==old["arms"][arm]["sha256"]
        model=EvidenceActor()
        model.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        model.eval()
        frozen[arm]=model
    return new,frozen,{
        "training_report_sha256":file_sha(trained/"report.json"),
        "f10b_frozen_report_sha256":file_sha(frozen_dir/"report.json"),
        "f10b_frozen_checkpoint_shas":{
            k:old["arms"][k]["sha256"] for k in ("memory_available","recurrent_only")},
    }

def execute(args):
    if args.index not in range(4):raise ValueError("Four fixed actors only")
    torch.set_num_threads(1)
    new,old,source=load_pair(args)
    path=Path(args.output)
    path.mkdir(parents=True,exist_ok=False)
    rows=[]
    for j in range(DEV_WORLDS):
        seed=seed_for("dev",args.index,j)
        cases=world(seed)
        assert {c["cell"] for c in cases}==set(GROUP_NAMES)
        assert len({c["history_sha256"] for c in cases})==1
        for c in cases:
            m=c["memory"];z=np.zeros_like(m)
            p={
                "read_clean":probability(new["relational_read"],c,m),
                "skip":probability(new["relational_read"],c,z),
                "read_stale":probability(new["relational_read"],c,stale_memory(c)),
                "read_forged":probability(new["relational_read"],c,forged_credential(c)),
                "unary_retrained":probability(new["unary_read_control"],c,m),
                "recurrent_retrained":probability(new["recurrent_only_control"],c,z),
                "f10b_frozen_read":probability(old["memory_available"],c,m),
                "f10b_frozen_recurrent":probability(old["recurrent_only"],c,z),
            }
            assert set(p)==set(MODES) and all(0<=v<=1 for v in p.values())
            rows.append({"seed":seed,"cell":c["cell"],"key":c["key"],
                         "truth":c["truth"],"history_sha256":c["history_sha256"],
                         "matching_reports":c["number_of_matching_reports"],
                         "contradictory_reports":c["number_of_contradictions"],
                         "p":p})
        if j%16==0:print("F10C_DEV_WORLD",args.index,j,seed,flush=True)
    assert len(rows)==256
    by_cell={}
    for cell in GROUP_NAMES:
        records=[r for r in rows if r["cell"]==cell]
        assert len(records)==64
        by_cell[cell]=by_rows(records)
    all_stats=by_rows(rows)
    all_stats["read_skip_bootstrap"]=world_bootstrap(rows)
    report={
        "experiment":"F10-C relational resolution multiple contradictions",
        "phase":"DEV_ONLY","index":args.index,"worlds":DEV_WORLDS,
        "cases":len(rows),"source":source,
        "dev_world_seeds":[seed_for("dev",args.index,j) for j in range(DEV_WORLDS)],
        "f10b_weights_frozen":True,"f10b_worlds_reused":False,
        "f10c_confirm_14300000_opened":False,"read_head_trained":False,
        "same_public_and_recurrent_before_read_intervention":True,
        "labels_derived_from_public_signed_records":True,
        "full_four_query_histories_matched":True,
        "by_cell":by_cell,"overall":all_stats,"rows":rows,
    }
    (path/"dev-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("F10C_DEV_SUMMARY",args.index,json.dumps({
        "overall":all_stats,
        "cells":{k:{"D+":v["D_clean"]["1"],
                    "correct":v["correct"]} for k,v in by_cell.items()},
    }),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--trained",type=Path,required=True)
    p.add_argument("--frozen-f10b",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    execute(p.parse_args())
