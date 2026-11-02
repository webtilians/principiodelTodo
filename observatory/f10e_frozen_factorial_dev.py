"""F10-E: audit frozen actors under public 2x2 credential x chronology.

Only original F10-C/B checkpoints. Factorial pairs alter PUBLIC evidence, so
GRU states differ ACROSS variants. READ vs SKIP within a variant shares the
identical public sequence/recurrent state and swaps only external F4.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from observatory.f10d_frozen_transfer_dev import load_models,prob
from observatory.f10e_public_factorial_worlds import (
    dev_seed,world,N_WORLDS,VARIANTS,COUNTS,TIMINGS,
    public_last_any,public_last_signed,
)
from observatory.f10b_temporal_evidence_task import KEYS

MODES=("rel_read","rel_skip","unary_read","gru_only",
       "f10b_read","f10b_gru_only","last_any","last_signed","first_signed")

def correctness(row,name):
    return int((row["pred"][name]>=0.5)==row["truth"])

def stats(rows):
    return {
      "cases":len(rows),
      "correct":{name:sum(correctness(r,name) for r in rows) for name in MODES},
      "D_read_skip":{str(v):sum(
          correctness(r,"rel_read")-correctness(r,"rel_skip")==v for r in rows)
          for v in (-1,0,1)},
    }

def bootstrap_world(rows):
    grouping={}
    for r in rows:grouping.setdefault(r["seed"],[]).append(r)
    assert len(grouping)==N_WORLDS and all(len(v)==4 for v in grouping.values())
    diffs=np.array([
      sum(correctness(r,"rel_read")-correctness(r,"rel_skip")
          for r in v)/4. for v in grouping.values()],dtype=float)
    rng=np.random.default_rng(169500)
    bs=np.sort(np.array([
        np.mean(rng.choice(diffs,size=len(diffs),replace=True))
        for _ in range(2000)]))
    return {"estimate":float(diffs.mean()),
            "ci95":[float(np.quantile(bs,.025)),float(np.quantile(bs,.975))],
            "unit":"world","draws":2000}

def contrasts(rows):
    groups={}
    for r in rows:groups.setdefault(r["seed"],{})[r["variant"]]=r
    assert len(groups)==N_WORLDS and all(set(v)==set(VARIANTS) for v in groups.values())
    out={name:0 for name in (
        "all_four_correct","credential_useful","precedence_useful",
        "credential_null_stable","precedence_null_stable",
        "always_initial_signed_baseline_correct")}
    # Unlike per-condition accuracy, these tests demand correct JOINT
    # decisions for matched counterfactual pairs and their null controls.
    for worldrows in groups.values():
        actual={v:int(row["pred"]["rel_read"]>=.5) for v,row in worldrows.items()}
        b=worldrows["U_before"]["base_value"]
        out["all_four_correct"]+=int(
            all(actual[v]==worldrows[v]["truth"] for v in VARIANTS))
        out["credential_useful"]+=int(
            actual["U_after"]==b and actual["S_after"]==1-b)
        out["precedence_useful"]+=int(
            actual["S_before"]==b and actual["S_after"]==1-b)
        out["credential_null_stable"]+=int(
            actual["U_before"]==b and actual["S_before"]==b)
        out["precedence_null_stable"]+=int(
            actual["U_before"]==b and actual["U_after"]==b)
        out["always_initial_signed_baseline_correct"]+=int(
            all(worldrows[v]["truth"]==b for v in VARIANTS))
    assert out["always_initial_signed_baseline_correct"]==0
    return out

def evaluate(args):
    if args.index not in range(4):raise ValueError("F10-E 4 original replicas only")
    torch.set_num_threads(1)
    models,provenance=load_models(args.f10c_source,args.f10b_source,args.index)
    output=Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for j in range(N_WORLDS):
        seed=dev_seed(args.index,j)
        variants=world(seed,args.index,j)
        assert len(variants)==4
        for case in variants:
            mem=case["memory"]
            zeros=np.zeros_like(mem)
            x=case["public"][:-1]
            q=case["key"]
            first=[v for v in x if v[q]>0 and v[KEYS+1]==1][0]
            preds={
                "rel_read":prob(models["f10c_relational_read"],case,mem),
                "rel_skip":prob(models["f10c_relational_read"],case,zeros),
                "unary_read":prob(models["f10c_unary_read_control"],case,mem),
                "gru_only":prob(models["f10c_recurrent_only_control"],case,zeros),
                "f10b_read":prob(models["f10b_memory_available"],case,mem),
                "f10b_gru_only":prob(models["f10b_recurrent_only"],case,zeros),
                "last_any":float(public_last_any(case)),
                "last_signed":float(public_last_signed(case)),
                "first_signed":float(first[KEYS]),
            }
            assert set(preds)==set(MODES)
            assert all(0<=v<=1 for v in preds.values())
            assert int(preds["last_signed"])==case["truth"]
            assert int(preds["first_signed"])==case["base_value"]
            rows.append({
                "seed":seed,"variant":case["variant"],
                "truth":case["truth"],"base_value":case["base_value"],
                "key":q,"record_count":case["record_count"],
                "timing":case["timing"],
                "challenge_source":case["challenge_source"],
                "confirm_source":case["confirm_source"],
                "challenge_signed":case["challenge_signed"],
                "challenge_before":case["challenge_before"],
                "history_sha256":case["history_sha256"],
                "pred":preds,
            })
        if j%18==0:
            print("F10E_NEW_FACTORIAL_WORLD",args.index,j,seed,flush=True)
    assert len(rows)==N_WORLDS*4
    overall=stats(rows)
    overall["world_cluster_ci"]=bootstrap_world(rows)
    overall["paired_contrasts"]=contrasts(rows)
    by_variant={name:stats([r for r in rows if r["variant"]==name])
                for name in VARIANTS}
    by_k={str(k):stats([r for r in rows if r["record_count"]==k])
          for k in COUNTS}
    by_timing={name:stats([r for r in rows if r["timing"]==name])
               for name in TIMINGS}
    by_initial={str(b):stats([r for r in rows if r["base_value"]==b])
                for b in (0,1)}
    assert all(v["cases"]==N_WORLDS for v in by_variant.values())
    assert all(v["cases"]==N_WORLDS//3*4 for v in by_k.values())
    assert all(v["cases"]==N_WORLDS//3*4 for v in by_timing.values())
    report={
        "experiment":"F10-E 2x2 public credential x temporal precedence",
        "stage":"CAUSAL_FACTORIAL_DEV_ONLY",
        "index":args.index,
        "n_base_worlds":N_WORLDS,
        "n_variants_per_world":4,
        "n_cases":len(rows),
        "seeds":[dev_seed(args.index,j) for j in range(N_WORLDS)],
        "all_models_frozen":True,"train_performed":False,
        "reader_head_trained":False,"old_worlds_revisited":False,
        "confirmation_16300000_opened":False,
        "truth_from_public_signed_evidence_only":True,
        "paired_factorial_changes_public_history":True,
        "read_skip_same_recurrent_history_within_variant":True,
        "source_sha_provenance":provenance,
        "overall":overall,"by_variant":by_variant,"by_k":by_k,
        "by_timing":by_timing,"by_initial_value":by_initial,
        "rows":rows,
    }
    (output/"dev-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("F10E_DEV_SUMMARY",args.index,json.dumps({
        "correct":overall["correct"],
        "D":overall["D_read_skip"],
        "contrasts":overall["paired_contrasts"],
        "ci":overall["world_cluster_ci"],
        "by_variant":{k:v["correct"] for k,v in by_variant.items()},
        "by_k":{k:v["correct"] for k,v in by_k.items()},
        "by_timing":{k:v["correct"] for k,v in by_timing.items()},
    }),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--f10c-source",type=Path,required=True)
    p.add_argument("--f10b-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    evaluate(p.parse_args())
