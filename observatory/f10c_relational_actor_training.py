"""F10-C: learnable pairwise evidence resolver and trained same-world controls."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from observatory.f9_external_minigrid import file_sha
from observatory.f10b_train_evidence_actors import EvidenceActor
from observatory.f10b_temporal_evidence_task import (
    KEYS,EVENT_FEATURES,PUBLIC_FEATURES,CAPACITY,WORLD_EVENTS,
)
from observatory.f10c_multi_version_worlds import world,seed_for,TRAIN_WORLDS

ARMS=("relational_read","unary_read_control","recurrent_only_control")
EPOCHS=12
BATCH=64
LR=.002
WIDTH=64

class RelationalActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(PUBLIC_FEATURES,WIDTH,batch_first=True)
        # Public key matching is structural; priority by signed/age is learned.
        self.unary=nn.Linear(4,1)
        self.pairwise=nn.Sequential(nn.Linear(7,16),nn.Tanh(),nn.Linear(16,1))
        self.head=nn.Sequential(nn.Linear(WIDTH+3,WIDTH),
                                nn.Tanh(),nn.Linear(WIDTH,1))

    def forward(self,history,memory):
        assert history.ndim==3 and history.shape[1:]==(WORLD_EVENTS+1,PUBLIC_FEATURES)
        assert memory.ndim==3 and memory.shape[1:]==(CAPACITY,EVENT_FEATURES)
        batch=len(history)
        _,state=self.gru(history)
        query=history[:,-1,:KEYS]
        match=(memory[:,:,:KEYS]*query[:,None,:]).sum(-1)
        # Every queried key has exactly 5 public reports, selected by KEY ONLY.
        ids=torch.topk(match,5,dim=1).indices
        selected=memory.gather(1,ids[:,:,None].expand(-1,-1,EVENT_FEATURES))
        valid=(selected[:,:,:KEYS]*query[:,None,:]).sum(-1)>0
        sign=selected[:,:,KEYS+1]
        moment=selected[:,:,KEYS+2]
        val=selected[:,:,KEYS]
        src=selected[:,:,KEYS+3:KEYS+5]
        same_src=(src[:,:,None,:]*src[:,None,:,:]).sum(-1)
        n=selected.shape[1]
        si=sign[:,:,None].expand(-1,-1,n)
        sj=sign[:,None,:].expand(-1,n,-1)
        vi=val[:,:,None].expand(-1,-1,n)
        vj=val[:,None,:].expand(-1,n,-1)
        td=moment[:,:,None]-moment[:,None,:]
        rel=torch.stack((si,sj,td,vi,vj,same_src,si*sj),dim=-1)
        score=self.pairwise(rel).squeeze(-1)
        peer_valid=(valid[:,:,None]&valid[:,None,:])
        rank=(score*peer_valid.float()).sum(-1)/peer_valid.sum(-1).clamp(min=1)
        own=self.unary(selected[:,:,KEYS+1:KEYS+5]).squeeze(-1)
        logits=(rank+own)*4.
        logits=logits.masked_fill(~valid,-1e5)
        weights=torch.softmax(logits,dim=1)
        weights=torch.where(valid,weights,torch.zeros_like(weights))
        read_value=(weights*val).sum(1)
        read_sign=(weights*sign).sum(1)
        available=valid.any(1).float()
        features=torch.cat((state[-1],read_value[:,None],
                            read_sign[:,None],available[:,None]),dim=1)
        return self.head(features).flatten()


def run(args):
    if args.index not in range(4):
        raise ValueError("Only four prespecified independent replicas")
    torch.set_num_threads(1)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for j in range(TRAIN_WORLDS):
        seed=seed_for("train",args.index,j)
        rows+=world(seed)
        if j%64==0:
            print("F10C_NEW_TRAIN_WORLDS",args.index,j,flush=True)
    assert len(rows)==2048
    x=torch.from_numpy(np.stack([c["public"] for c in rows]))
    m=torch.from_numpy(np.stack([c["memory"] for c in rows]))
    y=torch.from_numpy(np.array([c["truth"] for c in rows],np.float32))
    assert x.shape==(2048,61,PUBLIC_FEATURES)
    assert m.shape==(2048,64,EVENT_FEATURES)
    assert all(c["number_of_matching_reports"]==5 for c in rows)
    orders=[np.random.default_rng(171500+args.index*100+epoch).permutation(len(y))
            for epoch in range(EPOCHS)]
    report={
        "experiment":"F10-C multi-contradiction relational retrieval",
        "phase":"TRAIN_ONLY","index":args.index,
        "train_worlds":TRAIN_WORLDS,
        "train_cases":len(rows),"split":"14100000+",
        "epochs":EPOCHS,"batch":BATCH,"learning_rate":LR,
        "input_history_length":61,"f4_capacity":CAPACITY,
        "queries_per_world":4,"matching_reports":5,
        "no_old_training_worlds":True,"no_read_gate":True,
        "confirm_14300000_opened":False,"arms":{},
    }
    init_unary=None
    for arm in ARMS:
        torch.manual_seed(171000+args.index)
        model=RelationalActor() if arm=="relational_read" else EvidenceActor()
        initial_unary=torch.cat([t.detach().flatten() for t in model.parameters()]).numpy()
        if arm=="unary_read_control":init_unary=initial_unary.copy()
        if arm=="recurrent_only_control":assert np.array_equal(init_unary,initial_unary)
        masked=torch.zeros_like(m) if arm=="recurrent_only_control" else m
        optim=torch.optim.Adam(model.parameters(),lr=LR)
        losses=[]
        for epoch,order in enumerate(orders):
            model.train()
            average=0.
            for start in range(0,len(order),BATCH):
                idx=order[start:start+BATCH]
                pred=model(x[idx],masked[idx])
                loss=nn.functional.binary_cross_entropy_with_logits(pred,y[idx])
                optim.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optim.step()
                average+=float(loss.detach())*len(idx)
            losses.append(average/len(y))
            print("F10C_TRAIN_EPOCH",args.index,arm,epoch+1,
                  round(losses[-1],6),flush=True)
        model.eval()
        with torch.no_grad():
            score=model(x,masked)
            hits=int(((score>=0).float()==y).sum())
        path=out/(arm+".pt")
        torch.save(model.state_dict(),path)
        report["arms"][arm]={
            "checkpoint":path.name,"sha256":file_sha(path),
            "parameters":sum(p.numel() for p in model.parameters()),
            "losses":losses,"steps":EPOCHS*len(y)//BATCH,
            "accuracy_train":hits/len(y)}
        (out/"report.json").write_text(json.dumps(report,indent=2))
        print("F10C_NEW_TRAINED",args.index,arm,
              "accuracy",report["arms"][arm]["accuracy_train"],flush=True)
    print("F10C_ALL_ACTORS_TRAIN_DONE",args.index,flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
