"""F9-X-B: post-hoc channel audit of previously evaluated F9-X exploratory cases.

NO ENVIRONMENT ROLLOUTS, no new-world access, no weights trained or changed.
Ranking units on exploratory cases is DESCRIPTIVE, not confirmatory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from observatory.f9v_frozen_confirm import load_dev_models, validate_original_implementation
from observatory.f9x_frozen_head_attribution import vector, score


def signature(values, eps=1e-8):
    a = np.asarray(values, np.float64)
    n = len(a)
    pos, neg = int((a > eps).sum()), int((a < -eps).sum())
    return {
        "n": n,
        "nonzero": pos + neg,
        "sign_consistency": float(max(pos, neg) / n) if n else 0.,
        "mean_signed_delta": float(a.mean()) if n else 0.,
        "mean_absolute_delta": float(np.abs(a).mean()) if n else 0.,
    }


def analyze(index, data, model):
    cases, oldrows = data["cases"], data["predictions"]
    assert data["index"] == index and data["stage"] == "EXPLORATORY_NEW_WORLDS"
    assert len(cases) == len(oldrows) == 128
    # Exact cached-factual check ensures analysis cannot substitute a new trained head.
    for i in range(0, 128, 7):
        old = oldrows[i]["lex_hc"]["logit"]
        new = score(model, vector(cases[i]))["logit"]
        assert abs(old - new) < 2e-5
    cue_pairs = []
    history_pairs = []
    for base in range(0, 128, 4):
        group = cases[base:base+4]
        assert len(set(c["seed"] for c in group)) == 1
        for a,b in ((base,base+1),(base+2,base+3)):
            assert cases[a]["cue"] != cases[b]["cue"]
            assert np.array_equal(cases[a]["public"],cases[b]["public"])
            cue_pairs.append((a,b))
        for a,b in ((base,base+2),(base+1,base+3)):
            assert cases[a]["regime"] != cases[b]["regime"]
            assert np.array_equal(cases[a]["public"],cases[b]["public"])
            history_pairs.append((a,b))
    allvecs = [vector(c) for c in cases]
    baseline = [r["lex_hc"]["logit"] for r in oldrows]
    out = {}
    for group, offset in (("H",148),("C",212)):
        unit_rows = []
        for u in range(64):
            cue_logit_change=[]
            history_logit_change=[]
            cue_flip=0
            history_flip=0
            cue_signals=[]
            history_signals=[]
            k=offset+u
            for a,b in cue_pairs:
                key,ball=(a,b) if cases[a]["cue"]=="key" else (b,a)
                cue_signals.append(float(allvecs[key][k]-allvecs[ball][k]))
            for a,b in history_pairs:
                normal,off=(a,b) if cases[a]["regime"]=="history_f4_normal" else (b,a)
                history_signals.append(float(allvecs[normal][k]-allvecs[off][k]))
            for pairs, changes, is_cue in (
                (cue_pairs,cue_logit_change,True),
                (history_pairs,history_logit_change,False),
            ):
                for a,b in pairs:
                    for recipient,donor in ((a,b),(b,a)):
                        x=allvecs[recipient].copy()
                        x[k]=allvecs[donor][k]
                        pred=score(model,x)
                        changes.append(abs(pred["logit"]-baseline[recipient]))
                        if (pred["read"] != oldrows[recipient]["lex_hc"]["read"]):
                            if is_cue:cue_flip+=1
                            else:history_flip+=1
            unit_rows.append({
                "channel": u,
                "cue_mean_abs_logit_effect": float(np.mean(cue_logit_change)),
                "cue_read_flips": cue_flip,
                "cue_signature": signature(cue_signals),
                "history_mean_abs_logit_effect": float(np.mean(history_logit_change)),
                "history_read_flips": history_flip,
                "history_signature": signature(history_signals),
            })
        by_cue=sorted(unit_rows,key=lambda z:(-z["cue_mean_abs_logit_effect"],z["channel"]))
        by_history=sorted(unit_rows,key=lambda z:(-z["history_mean_abs_logit_effect"],z["channel"]))
        out[group]={
            "all_64_units":unit_rows,
            "top_10_cue_causal":by_cue[:10],
            "top_10_history_causal":by_history[:10],
            "number_single_unit_cue_flippers":sum(v["cue_read_flips"]>0 for v in unit_rows),
            "number_single_unit_history_flippers":sum(v["history_read_flips"]>0 for v in unit_rows),
        }
    return out


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--f9x-report",type=Path,required=True)
    p.add_argument("--f9v-weights",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.index not in range(4):raise ValueError("Only original four actors")
    torch.set_num_threads(1)
    validate_original_implementation()
    models,old=load_dev_models(args.f9v_weights,args.index)
    report=json.loads(args.f9x_report.read_text())
    assert set(report["world_seeds"]) == {
        9500000+10000*args.index+j for j in range(32)}
    result={
        "stage":"F9-X-B POSTHOC SAME EXISTING 9500000 EXPLORATORY DATA",
        "index":args.index,
        "original_f9x_run":41366421641,
        "original_dev_run":41345497955,
        "head_weights_unchanged":True,
        "new_worlds_evaluated":0,
        "split_8300000_reused":False,
        "split_9600000_opened":False,
        "units":analyze(args.index,report,models["lex_hc"]),
    }
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists():raise FileExistsError("Cannot overwrite channel attribution")
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("F9XB_POSTHOC_CHANNEL_RESULT",args.index,json.dumps({
        g:{k:v for k,v in d.items() if k!="all_64_units"}
        for g,d in result["units"].items()
    },ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
