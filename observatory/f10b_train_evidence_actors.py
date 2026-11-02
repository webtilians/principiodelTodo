"""F10-B: train equal-size GRU actors with/without external public event log.

No privileged label, target, validity bit or source-oracle enters the actor:
only temporal public report vectors and observation-only FIFO F4 are inputs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from observatory.f9_external_minigrid import file_sha
from observatory.f10b_temporal_evidence_task import (
    KEYS, CAPACITY, EVENT_FEATURES, PUBLIC_FEATURES, WORLD_EVENTS,
    TRAIN_WORLDS, seed_for, world,
)

ARMS=("memory_available","recurrent_only")
EPOCHS=12
BATCH=64
LR=0.002
WIDTH=64

class EvidenceActor(nn.Module):
    """Shared key-aware retrieval geometry; source trust and age LEARNED."""
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(PUBLIC_FEATURES,WIDTH,batch_first=True)
        # No hardcoded signed/recency priorities. Sources are randomized.
        self.provenance_score=nn.Linear(4,1)
        nn.init.zeros_(self.provenance_score.weight)
        nn.init.zeros_(self.provenance_score.bias)
        self.head=nn.Sequential(nn.Linear(WIDTH+3,WIDTH),
                                nn.Tanh(),nn.Linear(WIDTH,1))

    def forward(self,history,memory):
        assert history.ndim==3 and history.shape[1:]==(WORLD_EVENTS+1,PUBLIC_FEATURES)
        assert memory.ndim==3 and memory.shape[1:]==(CAPACITY,EVENT_FEATURES)
        _,last=self.gru(history)
        hidden=last[-1]
        query=history[:,-1,:KEYS]
        assert query.shape==(len(history),KEYS)
        match=(memory[:,:,:KEYS]*query[:,None,:]).sum(-1)
        occupied=memory[:,:,:KEYS].sum(-1)>0
        # Structural key match is explicit; 4 provenance+time components are
        # learned from TRAIN, never oracle-selected at DEV.
        meta=memory[:,:,KEYS+1:KEYS+5]
        score=14.*match+ self.provenance_score(meta).squeeze(-1)
        score=score.masked_fill(~occupied,-1e5)
        weights=torch.softmax(score,dim=1)
        weights=torch.where(occupied,weights,torch.zeros_like(weights))
        read_value=(weights*memory[:,:,KEYS]).sum(1)
        read_signed=(weights*memory[:,:,KEYS+1]).sum(1)
        has_memory=occupied.any(1).float()
        x=torch.cat((hidden,read_value[:,None],
                     read_signed[:,None],has_memory[:,None]),dim=1)
        return self.head(x).flatten()

def dataset(index):
    cases=[]
    for j in range(TRAIN_WORLDS):
        seed=seed_for("train",index,j)
        generated=world(seed)
        assert len(generated)==4
        cases.extend(generated)
        print("F10B_NEW_TRAIN_WORLD",index,j,seed,flush=True)
    assert len(cases)==TRAIN_WORLDS*4
    return cases

def train(args):
    if args.index not in range(4):
        raise ValueError("Only four independent F10-B training replicas")
    torch.set_num_threads(1)
    torch.manual_seed(151000+args.index)
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    cases=dataset(args.index)
    x=torch.from_numpy(np.stack([c["public"] for c in cases]))
    m=torch.from_numpy(np.stack([c["memory"] for c in cases]))
    y=torch.from_numpy(np.array([c["truth"] for c in cases],np.float32))
    assert x.shape==(1024,WORLD_EVENTS+1,PUBLIC_FEATURES)
    assert m.shape==(1024,CAPACITY,EVENT_FEATURES)
    assert y.shape==(1024,) and set(y.numpy())=={0.,1.}
    report={
        "experiment":"F10-B temporal evidence / provenance-dependent memory",
        "stage":"TRAIN_ONLY","index":args.index,
        "split":"13100000+","train_worlds":TRAIN_WORLDS,
        "train_cases":len(cases),"epochs":EPOCHS,
        "batch_size":BATCH,"lr":LR,"gru_width":WIDTH,
        "f4_capacity":CAPACITY,"events_per_world":WORLD_EVENTS,
        "no_old_worlds":True,"no_confirm_13300000":True,
        "no_read_gate_head":True,
        "same_history_across_four_queries":True,
        "source_provenance_public":True,
        "actor_source":"NEW RANDOM WEIGHTS not F9-P/F10-A",
        "arms":{},
    }
    init_states=[]
    orders=[np.random.default_rng(151500+args.index*100+e).permutation(len(y))
            for e in range(EPOCHS)]
    for arm in ARMS:
        torch.manual_seed(151000+args.index)
        model=EvidenceActor()
        initial=torch.cat([t.detach().flatten() for t in model.parameters()]).numpy()
        init_states.append(initial.copy())
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        mask_memory=torch.zeros_like(m) if arm=="recurrent_only" else m
        losses=[]
        for e,order in enumerate(orders):
            model.train()
            total=0.
            for start in range(0,len(order),BATCH):
                idx=order[start:start+BATCH]
                pred=model(x[idx],mask_memory[idx])
                loss=nn.functional.binary_cross_entropy_with_logits(pred,y[idx])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                total+=float(loss.detach())*len(idx)
            losses.append(total/len(order))
            print("F10B_TRAIN_EPOCH",args.index,arm,e+1,
                  round(losses[-1],6),flush=True)
        model.eval()
        with torch.no_grad():
            logits=model(x,mask_memory)
            hits=int(((logits>=0).float()==y).sum().item())
        path=out/(arm+".pt")
        torch.save(model.state_dict(),path)
        report["arms"][arm]={
            "checkpoint":path.name,"sha256":file_sha(path),
            "train_accuracy":hits/len(y),
            "trained_epochs":EPOCHS,"updates":EPOCHS*(len(y)//BATCH),
            "training_losses":losses,
            "parameters":sum(q.numel() for q in model.parameters()),
        }
        (out/"report.json").write_text(json.dumps(report,indent=2))
        print("F10B_ACTOR_TRAINED",args.index,arm,
              json.dumps({"sha256":report["arms"][arm]["sha256"],
                          "accuracy":report["arms"][arm]["train_accuracy"]}),
              flush=True)
    assert np.array_equal(init_states[0],init_states[1])
    assert report["arms"][ARMS[0]]["parameters"]==report["arms"][ARMS[1]]["parameters"]
    print("F10B_NEW_ACTOR_TRAINING_COMPLETE",args.index,flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    train(p.parse_args())

if __name__=="__main__":
    main()
