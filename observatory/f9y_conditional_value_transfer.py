"""F9-Y: independent conditional-value predictor and heldout cue/history combinations.

Only new 10100000 TRAIN and 10200000 DEV worlds. No F9-V/F9-W/F9-X
holdouts used; no 10300000 entry is implemented. Frozen F9-P/F4 source.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn

from observatory.f9_external_minigrid import ensure_dependencies
from observatory.f9q_train_cue_probes import validate_sources
from observatory.f9v_lexicographic_causal_f4 import one_seed_cases
from observatory.f9v_frozen_confirm import validate_original_implementation

TRAIN_BASE = 10_100_000
DEV_BASE = 10_200_000
SEALED_BASE = 10_300_000
TRAIN_N = 128
DEV_N = 32
N_ACTORS = 4
EPOCHS = 200
LR = .001
CELLS = (
    ("key", "history_f4_normal"),
    ("ball", "history_f4_normal"),
    ("key", "history_f4_postcue_off"),
    ("ball", "history_f4_postcue_off"),
)
FULL_HEADS = ("deploy_hc", "deploy_public", "controlled_hc", "controlled_nuisance")
LOCO_HEADS = ("loco_hc", "loco_nuisance")
MODES = FULL_HEADS + LOCO_HEADS


def seed_for(split, actor_index, j):
    limits = {"train": (TRAIN_BASE, TRAIN_N), "dev": (DEV_BASE, DEV_N)}
    if split not in limits or actor_index not in range(N_ACTORS):
        raise ValueError("F9-Y only exposes new train/dev worlds")
    base, n = limits[split]
    if not 0 <= j < n:
        raise ValueError("F9-Y out of split")
    return base + 10_000 * actor_index + j


def cell_index(case):
    return CELLS.index((case["cue"], case["regime"]))


def feature(case, name):
    public = np.asarray(case["public"], dtype=np.float32)
    hc = np.asarray(case["hc"], dtype=np.float32)
    assert public.shape == (148,) and hc.shape == (128,)
    if name in ("deploy_public", "controlled_nuisance", "loco_nuisance"):
        hc = np.zeros_like(hc)
    parts = [public, hc]
    if name.startswith("controlled_") or name.startswith("loco_"):
        cue, history = case["cue"], case["regime"]
        tags = np.array([
            float(cue == "key"), float(cue == "ball"),
            float(history == "history_f4_normal"),
            float(history == "history_f4_postcue_off")], dtype=np.float32)
        assert tags.sum() == 2.
        parts.append(tags)
    x = np.concatenate(parts)
    assert x.shape == (280 if len(parts) == 3 else 276,)
    assert np.isfinite(x).all()
    return x


def pair_features(cases, name):
    return torch.from_numpy(np.stack([feature(c, name) for c in cases]))


def train_head(cases, name, actor_index, fold):
    assert cases and actor_index in range(4)
    assert name in MODES
    dim = 280 if name.startswith(("controlled_", "loco_")) else 276
    torch.manual_seed(113_000 + actor_index + 100 * fold)
    model = nn.Sequential(nn.Linear(dim, 32), nn.Tanh(), nn.Linear(32, 1))
    x = pair_features(cases, name)
    target = torch.tensor([int(c["effect"] == 1) for c in cases], dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    for _ in range(EPOCHS):
        logits = model(x).flatten()
        loss = criterion(logits, target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, {
        "name": name, "fold": fold, "train_states": len(cases),
        "positive": int(target.sum().item()),
        "epochs": EPOCHS, "optimizer_steps": EPOCHS,
        "lr": LR, "pos_weight": 1.,
        "final_train_bce": float(loss.item()),
    }


def prob(model, x):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    with torch.no_grad():
        return float(torch.sigmoid(model(x)).squeeze().item())


def bce(label, p):
    p = min(max(float(p), 1e-7), 1-1e-7)
    return -math.log(p) if label else -math.log(1-p)


def summarize(cases, choices):
    assert len(cases) == len(choices)
    positive = [int(c["effect"] == 1) for c in cases]
    predicted = [int(p >= .5) for p in choices]
    outcomes = [c["read_branch"] if r else c["skip_branch"]
                for c, r in zip(cases, predicted)]
    return {
        "states": len(cases), "positive": sum(positive),
        "negative_or_neutral": len(cases)-sum(positive),
        "logloss": float(np.mean([bce(y, p) for y,p in zip(positive,choices)])),
        "brier": float(np.mean([(y-p)**2 for y,p in zip(positive,choices)])),
        "reads": sum(predicted),
        "read_positive": sum(r and c["effect"] == 1 for c,r in zip(cases,predicted)),
        "read_harmful": sum(r and c["effect"] == -1 for c,r in zip(cases,predicted)),
        "read_neutral": sum(r and c["effect"] == 0 for c,r in zip(cases,predicted)),
        "first_correct": sum(int(o["first_arm_correct"]) for o in outcomes),
        "auc": auc(positive, choices),
    }


def auc(labels, probs):
    positive=[p for y,p in zip(labels,probs) if y]
    negative=[p for y,p in zip(labels,probs) if not y]
    if not positive or not negative:return None
    wins = sum(float(a > b) + .5 * float(a == b)
               for a in positive for b in negative)
    return wins / (len(positive) * len(negative))


def shuffled_feature(cases, i, name, group):
    """Donor from a different world in SAME actor, cue and history stratum."""
    if group not in ("H", "C", "HC"):
        raise ValueError(group)
    c=cases[i]
    matching=[j for j,v in enumerate(cases) if cell_index(v)==cell_index(c)]
    assert len(matching)==DEV_N
    position=matching.index(i)
    donor=cases[matching[(position+1) % len(matching)]]
    assert donor["seed"] != c["seed"]
    x=feature(c,name).copy()
    other=feature(donor,name)
    if group=="H":x[148:212]=other[148:212]
    elif group=="C":x[212:276]=other[212:276]
    else:x[148:276]=other[148:276]
    assert np.array_equal(x[:148], feature(c,name)[:148])
    if len(x)==280:
        assert np.array_equal(x[276:],feature(c,name)[276:])
    return x


def execute(args):
    if args.index not in range(N_ACTORS):
        raise ValueError(args.index)
    torch.set_num_threads(1)
    validate_original_implementation()
    ensure_dependencies()
    paths, shas, provenance=validate_sources(args.bc_source,args.ppo_source,args.index)
    from sb3_contrib import RecurrentPPO
    actor=RecurrentPPO.load(str(paths["ppo_sham"]),device="cpu")
    actor.policy.set_training_mode(False)
    for p in actor.policy.parameters():
        p.requires_grad_(False)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    train=[]
    dev=[]
    for phase,n,destination in (("train",TRAIN_N,train),("dev",DEV_N,dev)):
        for j in range(n):
            seed=seed_for(phase,args.index,j)
            cases=one_seed_cases(actor,args.index,seed)
            assert len(cases)==4 and {cell_index(c) for c in cases}==set(range(4))
            assert {c["seed"] for c in cases}=={seed}
            assert all(c["public_fork_equal"] and
                       c["read_branch"]["read"]==1 and c["skip_branch"]["read"]==0
                       for c in cases)
            destination.extend(cases)
            print("F9Y_NEW_WORLD",phase,args.index,j,seed,flush=True)
    assert len(train)==4*TRAIN_N and len(dev)==4*DEV_N
    full={}
    training={}
    for name in FULL_HEADS:
        fold=4 if name.startswith("controlled_") else 5
        model,meta=train_head(train,name,args.index,fold)
        full[name]=model
        training[name]=meta
        torch.save(model.state_dict(),out/(name+".pt"))
    loco_hc={}
    loco_nuisance={}
    for cell in range(4):
        subset=[v for v in train if cell_index(v)!=cell]
        assert len(subset)==3*TRAIN_N
        for name,into in (("loco_hc",loco_hc),("loco_nuisance",loco_nuisance)):
            model,meta=train_head(subset,name,args.index,cell)
            into[cell]=model
            training[name+"_"+str(cell)]=meta
            torch.save(model.state_dict(),out/(name+"_"+str(cell)+".pt"))
    # Diagnostic cue+regime prior is fitted from TRAIN ONLY.
    priors={}
    for cell in range(4):
        items=[c for c in train if cell_index(c)==cell]
        p=sum(c["effect"]==1 for c in items)
        priors[cell]=(p+1)/(len(items)+2)
    rows=[]
    for i,c in enumerate(dev):
        fold=cell_index(c)
        probs={name:prob(model,feature(c,name)) for name,model in full.items()}
        probs["loco_hc"]=prob(loco_hc[fold],feature(c,"loco_hc"))
        probs["loco_nuisance"]=prob(loco_nuisance[fold],feature(c,"loco_nuisance"))
        probs["prior_cell"]=priors[fold]
        for name,model in (("controlled_hc",full["controlled_hc"]),
                           ("loco_hc",loco_hc[fold])):
            for group in ("H","C","HC"):
                probs[name+"_shuffle_"+group]=prob(
                    model,shuffled_feature(dev,i,name,group))
        rows.append(probs)
    names=list(rows[0])
    aggregate={n:summarize(dev,[r[n] for r in rows]) for n in names}
    by_cell={}
    for fold,(cue,regime) in enumerate(CELLS):
        idx=[i for i,c in enumerate(dev) if cell_index(c)==fold]
        by_cell[str(fold)]={
            "cue":cue,"regime":regime,"states":len(idx),
            "positive":sum(dev[i]["effect"]==1 for i in idx),
            "negative":sum(dev[i]["effect"]!=1 for i in idx),
            "read_harmful_available":sum(dev[i]["effect"]==-1 for i in idx),
            "models":{n:summarize([dev[i] for i in idx],[rows[i][n] for i in idx])
                      for n in names},
        }
    train_counts={str(cell):{
        "positive":sum(c["effect"]==1 for c in train if cell_index(c)==cell),
        "negative":sum(c["effect"]!=1 for c in train if cell_index(c)==cell),
    } for cell in range(4)}
    result={
        "experiment":"F9-Y conditional causal value / leave-one-cue-history-combination-out",
        "phase":"TRAIN_DEV_ONLY","index":args.index,
        "source_sha256":shas,"source_provenance":provenance,
        "train_worlds":TRAIN_N,"dev_worlds":DEV_N,
        "train_cases":train,"dev_cases":dev,
        "predictions":rows,"training":training,
        "train_counts_by_cell":train_counts,
        "dev_by_cell":by_cell,"dev_aggregate":aggregate,
        "new_train_seeds":sorted({c["seed"] for c in train}),
        "new_dev_seeds":sorted({c["seed"] for c in dev}),
        "heads_trained_on_DEV":False,
        "actor_retrained":False,"F4_writer_changed":False,
        "f9v_8300000_reused":False,
        "f9w_9300000_opened":False,
        "f9x_9600000_opened":False,
        "f9y_10300000_opened":False,
        "f9t_6300000_opened":False,"f9u_7300000_opened":False,
        "historic_900000_opened":False,
    }
    (out/"report.json").write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("F9Y_NEW_DEV_SUMMARY",args.index,json.dumps({
        "train_counts":train_counts,
        "dev_cells":{k:{"positive":v["positive"],"negative":v["negative"],
                          "loco_hc":v["models"]["loco_hc"]["logloss"],
                          "loco_nuisance":v["models"]["loco_nuisance"]["logloss"]}
                     for k,v in by_cell.items()},
        "deploy_hc":aggregate["deploy_hc"],
        "loco_hc":aggregate["loco_hc"],
        "loco_nuisance":aggregate["loco_nuisance"],
    },ensure_ascii=False),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    execute(p.parse_args())


if __name__=="__main__":
    main()
