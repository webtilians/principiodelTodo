"""F10-B: independent 13200000+ DEV evaluation, matched F4 interventions.

All rows are emitted, including negative READ and failed provenance/revision.
No training from DEV and no 13300000+ confirm entrypoint.
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch

from pathlib import Path

from observatory.f9_external_minigrid import file_sha
from observatory.f10b_temporal_evidence_task import (
    DEV_WORLDS, CELLS, KEYS, seed_for, world,
    stale_memory, forged_credential,
)
from observatory.f10b_train_evidence_actors import (
    EvidenceActor, ARMS, EPOCHS,
)

def _prob(model,history,memory):
    with torch.no_grad():
        x=torch.from_numpy(history[None,:,:])
        m=torch.from_numpy(memory[None,:,:])
        return float(torch.sigmoid(model(x,m)).item())

def _counts(rows,probname):
    outcomes=[int(c["probs"][probname]>=.5)==c["truth"] for c in rows]
    return {"cases":len(rows),"correct":int(sum(outcomes)),
            "accuracy":float(np.mean(outcomes))}

def _effects(rows,mode):
    # Paired within the same actor/world/history/query; do not approximate
    # with independently fitted actors.
    out=[]
    for row in rows:
        yes=int(row["probs"][mode]>=.5)==row["truth"]
        no=int(row["probs"]["skip"]>=.5)==row["truth"]
        out.append(int(yes)-int(no))
    return {str(k):sum(x==k for x in out) for k in (-1,0,1)}

def _bootstrap_world_delta(rows, n=2000):
    grouped={}
    for c in rows:
        grouped.setdefault(c["seed"],[]).append(c)
    assert len(grouped)==DEV_WORLDS and all(len(v)==4 for v in grouped.values())
    perworld=[]
    for cases in grouped.values():
        delta=sum((int(c["probs"]["read_clean"]>=.5)==c["truth"])-
                  (int(c["probs"]["skip"]>=.5)==c["truth"])
                  for c in cases)/4.
        perworld.append(delta)
    rnd=np.random.default_rng(155600)
    draws=np.array(perworld)
    bs=np.sort([float(np.mean(rnd.choice(draws,size=len(draws),replace=True)))
                for _ in range(n)])
    return {"mean":float(np.mean(draws)),
            "95ci":[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
            "resampling_unit":"world",
            "bootstrap_draws":n}

def run(args):
    if args.index not in range(4):
        raise ValueError("Only four independent DEV actors")
    torch.set_num_threads(1)
    source=Path(args.trained)
    meta=json.loads((source/"report.json").read_text())
    assert meta["stage"]=="TRAIN_ONLY" and meta["index"]==args.index
    assert meta["train_worlds"]==256 and meta["epochs"]==EPOCHS
    assert meta["train_cases"]==1024 and not meta["no_confirm_13300000"] is False
    assert set(meta["arms"])==set(ARMS)
    models={}
    for name in ARMS:
        path=source/meta["arms"][name]["checkpoint"]
        assert file_sha(path)==meta["arms"][name]["sha256"]
        m=EvidenceActor()
        m.load_state_dict(torch.load(path,map_location="cpu",weights_only=True))
        m.eval()
        models[name]=m
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for j in range(DEV_WORLDS):
        seed=seed_for("dev",args.index,j)
        cases=world(seed)
        assert len({x["shared_history_sha256"] for x in cases})==1
        assert {x["cell"] for x in cases}=={a+"|"+b for a,b in CELLS}
        for c in cases:
            hist=c["public"]
            memory=c["memory"]
            zero=np.zeros_like(memory)
            probs={
                "read_clean":_prob(models["memory_available"],hist,memory),
                "skip":_prob(models["memory_available"],hist,zero),
                "read_stale":_prob(models["memory_available"],hist,stale_memory(c)),
                "read_forged":_prob(models["memory_available"],hist,forged_credential(c)),
                "recurrent_only":_prob(models["recurrent_only"],hist,zero),
            }
            assert all(0<=v<=1 for v in probs.values())
            rows.append({
                "seed":seed,"cell":c["cell"],"key":c["key"],
                "truth":c["truth"],
                "history_sha256":c["shared_history_sha256"],
                "signed_last":not c["latest_unsigned"],"revised":c["revised"],
                "probs":probs,
                "D_clean":int((int(probs["read_clean"]>=.5)==c["truth"])-
                              (int(probs["skip"]>=.5)==c["truth"])),
            })
        print("F10B_NEW_DEV_WORLD",args.index,j,seed,flush=True)
    assert len(rows)==4*DEV_WORLDS
    overall={
        "predictions":{mode:_counts(rows,mode) for mode in
                       ("read_clean","skip","read_stale","read_forged","recurrent_only")},
        "D_clean":_effects(rows,"read_clean"),
        "D_stale":_effects(rows,"read_stale"),
        "D_forged":_effects(rows,"read_forged"),
        "delta_cluster_bootstrap":_bootstrap_world_delta(rows),
    }
    strata={}
    for signed,revision in CELLS:
        name=signed+"|"+revision
        subset=[r for r in rows if r["cell"]==name]
        assert len(subset)==DEV_WORLDS
        strata[name]={
            "predictions":{mode:_counts(subset,mode) for mode in
                           ("read_clean","skip","read_stale",
                            "read_forged","recurrent_only")},
            "D_clean":_effects(subset,"read_clean"),
            "D_stale":_effects(subset,"read_stale"),
            "D_forged":_effects(subset,"read_forged"),
        }
    report={
        "experiment":"F10-B temporal evidence causal READ",
        "phase":"DEV_ONLY","index":args.index,
        "train_report_sha256":file_sha(source/"report.json"),
        "worlds":DEV_WORLDS,"cases":len(rows),
        "world_seeds":[seed_for("dev",args.index,j) for j in range(DEV_WORLDS)],
        "shared_history_across_four_strata":True,
        "labels_from_public_signed_reports":True,
        "same_actor_and_hidden_for_read_skip":True,
        "no_confirmation_opened":True,"no_old_worlds_reused":True,
        "no_reader_gate_trained":True,
        "overall":overall,"by_cell":strata,"rows":rows,
    }
    (output/"dev-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    summary={
        "index":args.index,
        "accuracy":{k:v["accuracy"] for k,v in overall["predictions"].items()},
        "D_clean":overall["D_clean"],
        "bootstrap":overall["delta_cluster_bootstrap"],
        "cells":{k:{
            "D+":v["D_clean"]["1"],
            "D-":v["D_clean"]["-1"],
            "read_clean_correct":v["predictions"]["read_clean"]["correct"],
            "skip_correct":v["predictions"]["skip"]["correct"],
            "recurrent_only_correct":v["predictions"]["recurrent_only"]["correct"],
        } for k,v in strata.items()},
    }
    print("F10B_DEV_SUMMARY",json.dumps(summary,ensure_ascii=False),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--trained",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())

if __name__=="__main__":
    main()
