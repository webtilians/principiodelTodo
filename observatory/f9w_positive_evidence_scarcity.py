"""F9-W: independently test scarce positive causal READ examples.

The F9-V simulator/actor/F4/teacher-prefix remain frozen and unchanged.
New TRAIN/DEV worlds only, disjoint from consumed F9-V 8300000+ confirmation.
"""
from __future__ import annotations
import argparse,json
from collections import Counter
from pathlib import Path

import torch
from torch import nn

from observatory.f9_external_minigrid import ensure_dependencies,file_sha
from observatory.f9q_train_cue_probes import validate_sources
from observatory.f9v_lexicographic_causal_f4 import (
    one_seed_cases,head_inputs,ReadHead,
)

TRAIN_BASE=9_100_000
DEV_BASE=9_200_000
SEALED_CONFIRM_BASE=9_300_000
TRAIN_WORLDS=128
SHORT_WORLDS=32
DEV_WORLDS=16
EPOCHS=200
LR=.001
ARMS=("short32_hc","long128_hc","short32_weighted_hc","long128_public")

def seed_for(split,index,j):
    if split not in ("train","dev") or index not in range(4):
        raise ValueError("F9-W only trains/evaluates on new TRAIN/DEV")
    n=TRAIN_WORLDS if split=="train" else DEV_WORLDS
    if not 0<=j<n:raise ValueError("Out-of-split seed")
    return (TRAIN_BASE if split=="train" else DEV_BASE)+10_000*index+j

def arm_data(all_cases,arm):
    if arm not in ARMS:raise ValueError(arm)
    return all_cases if arm.startswith("long128") else all_cases[:SHORT_WORLDS*4]

def fit_arm(all_cases,index,arm):
    if index not in range(4):raise ValueError(index)
    data=arm_data(all_cases,arm)
    torch.manual_seed(91_000+index)
    head=ReadHead()
    inputs=head_inputs(data,use_hc=arm!="long128_public")
    labels=torch.tensor([c["label"] for c in data],dtype=torch.float32)
    npos=int(labels.sum().item());nneg=len(data)-npos
    weight=nneg/npos if (arm=="short32_weighted_hc" and npos>0) else 1.
    criterion=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight,dtype=torch.float32))
    optimizer=torch.optim.Adam(head.parameters(),lr=LR)
    for epoch in range(EPOCHS):
        logits=head.net(inputs).squeeze(-1)
        loss=criterion(logits,labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    head.eval()
    for p in head.parameters():p.requires_grad_(False)
    return head,{"arm":arm,"train_cases":len(data),"n_positive":npos,
                 "n_negative":nneg,"pos_weight":weight,
                 "no_positive_train":npos==0,"epochs":EPOCHS,
                 "optimizer_steps":EPOCHS,"final_train_loss":float(loss.item())}

def predict(head,case,arm):
    x=head_inputs([case],use_hc=arm!="long128_public")[0]
    with torch.no_grad():
        probability=float(torch.sigmoid(head(x)).item())
    return int(probability>=.5),probability

def evaluate(models,cases):
    details=[]
    for c in cases:
        row={}
        for arm,head in models.items():
            read,prob=predict(head,c,arm)
            outcome=c["read_branch"] if read else c["skip_branch"]
            row[arm]={"read":read,"prob":prob,
                      "correct":int(outcome["first_arm_correct"]),
                      "side":outcome["first_arm"],
                      "label_match":int(read==c["label"])}
        details.append(row)
    summary={}
    for arm in ARMS:
        reads=sum(x[arm]["read"] for x in details)
        npos=sum(c["effect"]==1 for c in cases)
        selected_pos=sum(x[arm]["read"] and c["effect"]==1 for c,x in zip(cases,details))
        selected_neg=sum(x[arm]["read"] and c["effect"]==-1 for c,x in zip(cases,details))
        selected_neutral=sum(x[arm]["read"] and c["effect"]==0 for c,x in zip(cases,details))
        by_regime={}
        for regime in ("history_f4_normal","history_f4_postcue_off"):
            idx=[i for i,c in enumerate(cases) if c["regime"]==regime]
            pair=[idx[k:k+2] for k in range(0,len(idx),2)]
            assert all(len(p)==2 for p in pair)
            by_regime[regime]={
              "correct":sum(details[i][arm]["correct"] for i in idx),
              "reads":sum(details[i][arm]["read"] for i in idx),
              "pairs_both_correct":sum(all(details[i][arm]["correct"] for i in p) for p in pair),
            }
        summary[arm]={
            "cases":len(cases),"correct":sum(x[arm]["correct"] for x in details),
            "reads":reads,"precision_read":selected_pos/reads if reads else None,
            "recall_positive":selected_pos/npos if npos else None,
            "selected_positive":selected_pos,"positive_available":npos,
            "selected_harmful":selected_neg,"selected_neutral":selected_neutral,
            "label_match":sum(x[arm]["label_match"] for x in details),
            "regimes":by_regime,
        }
    return summary,details

def execute(args):
    if args.phase not in ("smoke","dev") or args.index not in range(4):
        raise ValueError("Never open F9-W confirmation")
    if args.phase=="dev" and args.train_worlds!=TRAIN_WORLDS:
        raise ValueError("F9-W TRAIN budget fixed at 128 worlds")
    if args.phase=="smoke" and args.train_worlds!=1:
        raise ValueError("Smoke uses one world only")
    torch.set_num_threads(1)
    ensure_dependencies()
    paths,shas,provenance=validate_sources(args.bc_source,args.ppo_source,args.index)
    from sb3_contrib import RecurrentPPO
    actor=RecurrentPPO.load(str(paths["ppo_sham"]),device="cpu")
    actor.policy.set_training_mode(False)
    for p in actor.policy.parameters():p.requires_grad_(False)
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):raise FileExistsError("No overwrites")
    train=[]
    for j in range(args.train_worlds):
        train.extend(one_seed_cases(actor,args.index,seed_for("train",args.index,j)))
    assert len(train)==4*args.train_worlds
    models={};meta={}
    for arm in ARMS:
        head,stats=fit_arm(train,args.index,arm)
        models[arm]=head;meta[arm]=stats
        torch.save(head.state_dict(),root/(arm+".pt"))
    dev=[]
    n=1 if args.phase=="smoke" else DEV_WORLDS
    split="train" if args.phase=="smoke" else "dev"
    for j in range(n):
        dev.extend(one_seed_cases(actor,args.index,seed_for(split,args.index,j)))
    summary,details=evaluate(models,dev)
    result={
        "experiment":"F9-W scarce positive causal READ evidence",
        "phase":args.phase,"index":args.index,
        "model_source_shas":shas,"model_source_provenance":provenance,
        "train_worlds":args.train_worlds,"dev_worlds":n,
        "training":meta,"dev_summary":summary,
        "train_effects":dict(Counter(str(c["effect"]) for c in train)),
        "dev_effects":dict(Counter(str(c["effect"]) for c in dev)),
        "train_samples":train,"dev_samples":dev,"dev_decisions":details,
        "old_8300000_confirmation_used":False,
        "f9w_9300000_confirmation_opened":False,
        "f9u_7300000_opened":False,"f9t_6300000_opened":False,
        "old_900000_holdout_opened":False,
        "actor_retrained":False,"F4_writer_changed":False,
        "uses_external_llm":False,
        "protocol_sha256":file_sha(Path("docs/INFINITO_F9W_POSITIVE_EVIDENCE_SCARCITY_PROTOCOL.md")),
        "source_code_sha256":file_sha(Path(__file__)),
    }
    (root/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print("F9W_RESULT",args.phase,args.index,json.dumps({
        "train_positive":{k:v["n_positive"] for k,v in meta.items()},
        "dev_effects":result["dev_effects"],
        "summary":summary},ensure_ascii=False),flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=["smoke","dev"],required=True)
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--train-worlds",type=int,required=True)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    execute(p.parse_args())

if __name__=="__main__":main()
