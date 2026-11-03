"""F10-L phase one: TRAIN two primitives independently, freeze, evaluate DEV only.

No call into F10-L OOD generator. Composition is a separate phase-two
entrypoint, gated by an INDEPENDENT DEV audit across four replicas.
"""
from __future__ import annotations

import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli,Categorical

from observatory.f10l_worlds import xor_data,select_data,FEATURES,MAX_SLOTS

XOR_ARMS=("xor_onpolicy","xor_two_actions")
SEL_ARMS=("sel_onpolicy","sel_uniform_credit","sel_uniform_pair")
ALL_ARMS=XOR_ARMS+SEL_ARMS
EPOCHS=8
BATCH=64
LR=.003
ENTROPY=.002
EMA=.90

class XorActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self,x):
        return self.net(x).flatten()

class SelectActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Linear(FEATURES,1)
    def forward(self,features,mask):
        logits=self.net(features).squeeze(-1)
        return logits.masked_fill(mask<.5,-1e9)

def sha(file):
    return hashlib.sha256(Path(file).read_bytes()).hexdigest()

def reward(chosen,correct):
    """Visible ONLY after performing the action, not a supervised loss."""
    return torch.where(chosen==correct,torch.ones_like(chosen,dtype=torch.float32),
                       -torch.ones_like(chosen,dtype=torch.float32))

def xor_tensors(data):
    return (
        torch.from_numpy(np.stack([r["input"] for r in data])),
        torch.tensor([r["truth"] for r in data],dtype=torch.float32),
    )

def sel_tensors(data):
    return (
        torch.from_numpy(np.stack([r["features"] for r in data])),
        torch.from_numpy(np.stack([r["mask"] for r in data])),
        torch.tensor([r["winner"] for r in data],dtype=torch.long),
    )

def uniform_distinct_pairs(mask,indices,rep,epoch):
    a=[];b=[]
    for i in indices:
        # The EXACT same behavior actions in B and C, independent of model.
        rng=np.random.default_rng(236_000+100_000*rep+10_000*epoch+int(i))
        positions=np.flatnonzero(mask[int(i)].numpy()>.5)
        pair=rng.choice(positions,size=2,replace=False)
        a.append(int(pair[0]));b.append(int(pair[1]))
    return torch.tensor(a,dtype=torch.long),torch.tensor(b,dtype=torch.long)

def train(rep,out):
    train_x=xor_data("train",rep)
    train_s=select_data("train",rep)
    assert len(train_x)==len(train_s)==4096
    x,y=xor_tensors(train_x)
    f,mask,win=sel_tensors(train_s)
    summary={
        "experiment":"INFINITO F10-L local reward-credit primitives",
        "status":"FROZEN_BEFORE_DEV_AND_ANY_OOD",
        "replica":rep,
        "train_xor_case_hashes":[r["hash"] for r in train_x],
        "train_select_case_hashes":[r["hash"] for r in train_s],
        "train_xor_balance":{str(j):sum(
            1 for z in train_x if z["bit"]*2+z["flip"]==j) for j in range(4)},
        "train_composition_exposure":0,
        "train_xor_observations":4096,
        "train_select_observations":4096,
        "confirm_23300000_unopened":True,
        "lr":LR,"epochs":EPOCHS,"batch":BATCH,"entropy":ENTROPY,
        "ema":EMA,"arms":{},
    }
    assert all(i==1024 for i in summary["train_xor_balance"].values())
    for arm in ALL_ARMS:
        is_xor=arm in XOR_ARMS
        torch.manual_seed((235_000 if is_xor else 235_500)+rep)
        actor=XorActor() if is_xor else SelectActor()
        optimizer=torch.optim.Adam(actor.parameters(),lr=LR)
        ema=0.0
        total_episodes=0
        curves=[]
        first_weights_sha=hashlib.sha256(
            b"".join(param.detach().numpy().tobytes() for param in actor.parameters())
        ).hexdigest()
        t0=time.perf_counter()
        for epoch in range(EPOCHS):
            order=np.random.default_rng(
                (238_000 if is_xor else 238_500)+100*rep+epoch
            ).permutation(4096)
            correct=0
            rew=0.
            epoch_loss=0.
            for start in range(0,4096,BATCH):
                ids=order[start:start+BATCH]
                if is_xor:
                    logits=actor(x[ids])
                    dist=Bernoulli(logits=logits)
                    if arm=="xor_onpolicy":
                        a,b=dist.sample(),dist.sample()
                        ra,rb=reward(a,y[ids]),reward(b,y[ids])
                        loss=-.5*(
                            (ra.detach()-ema)*dist.log_prob(a)+
                            (rb.detach()-ema)*dist.log_prob(b)
                        ).mean()-ENTROPY*dist.entropy().mean()
                    else:
                        a,b=torch.zeros_like(logits),torch.ones_like(logits)
                        ra,rb=reward(a,y[ids]),reward(b,y[ids])
                        # Both possible actions ACTUALLY executed. This
                        # gives a label-equivalent counterfactual in binary
                        # deterministic XOR, explicitly noted in prereg.
                        loss=-((rb-ra).detach()*torch.sigmoid(logits)).mean()
                else:
                    logits=actor(f[ids],mask[ids])
                    dist=Categorical(logits=logits)
                    if arm=="sel_onpolicy":
                        a,b=dist.sample(),dist.sample()
                    else:
                        a,b=uniform_distinct_pairs(mask,ids,rep,epoch)
                    ra,rb=reward(a,win[ids]),reward(b,win[ids])
                    if arm=="sel_uniform_pair":
                        loss=-.5*((ra-rb).detach()*
                            (dist.log_prob(a)-dist.log_prob(b))).mean()
                    else:
                        loss=-.5*(
                            (ra.detach()-ema)*dist.log_prob(a)+
                            (rb.detach()-ema)*dist.log_prob(b)
                        ).mean()
                    loss=loss-ENTROPY*dist.entropy().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),1.)
                optimizer.step()
                cur_mean=float(torch.cat((ra,rb)).detach().mean())
                ema=EMA*ema+(1-EMA)*cur_mean
                correct+=int(((ra>0).sum()+(rb>0).sum()).item())
                rew+=float((ra+rb).sum().detach())
                epoch_loss+=float(loss.detach())*len(ids)
                total_episodes+=2*len(ids)
            curves.append({
                "epoch":epoch+1,"episodes":total_episodes,
                "executed_success":correct/(4096*2),
                "executed_mean_reward":rew/(4096*2),
                "loss":epoch_loss/4096,
            })
            print("F10L_TRAIN",rep,arm,"epoch",epoch+1,
                  "success",round(curves[-1]["executed_success"],5),
                  "episodes",total_episodes,flush=True)
        assert total_episodes==65536
        cp=out/(arm+".pt")
        torch.save(actor.state_dict(),cp)
        summary["arms"][arm]={
            "checkpoint":cp.name,"sha256":sha(cp),
            "initial_weights_sha256":first_weights_sha,
            "parameters":sum(p.numel() for p in actor.parameters()),
            "epochs":EPOCHS,"updates":512,"training_states_seen":32768,
            "executed_environment_episodes":total_episodes,
            "loss_supervised_winner_or_answer":False,
            "behavior":"on_policy" if arm.endswith("onpolicy")
                       else ("uniform_two_distinct" if arm.startswith("sel")
                             else "forced_zero_and_one"),
            "curves":curves,"seconds":time.perf_counter()-t0,
        }
        (out/"train.json").write_text(json.dumps(summary,indent=2))
    assert set(summary["arms"])==set(ALL_ARMS)
    return summary

def evaluate_dev(rep,out,summary):
    # This is the first generation of DEV; all five SHAs are already frozen.
    assert len(summary["arms"])==5
    for arm in ALL_ARMS:
        assert sha(out/(arm+".pt"))==summary["arms"][arm]["sha256"]
    cases_x=xor_data("dev",rep)
    cases_s=select_data("dev",rep)
    assert len(cases_x)==len(cases_s)==512
    x,y=xor_tensors(cases_x)
    f,mask,w=sel_tensors(cases_s)
    rows=[]
    for arm in ALL_ARMS:
        is_xor=arm in XOR_ARMS
        actor=XorActor() if is_xor else SelectActor()
        actor.load_state_dict(torch.load(out/(arm+".pt"),map_location="cpu",weights_only=True))
        actor.eval()
        with torch.no_grad():
            for start in range(0,512,64):
                if is_xor:
                    part=cases_x[start:start+64]
                    logits=actor(x[start:start+64])
                    chosen=(logits>=0).long().tolist()
                    for j,c in enumerate(part):
                        rows.append({
                            "replica":rep,"arm":arm,"task":"xor",
                            "seed":c["seed"],"base_world":c["base_world"],
                            "hash":c["hash"],"bit":c["bit"],"flip":c["flip"],
                            "truth":c["truth"],"action":chosen[j],
                        })
                else:
                    part=cases_s[start:start+64]
                    logits=actor(f[start:start+64],mask[start:start+64])
                    selected=logits.argmax(-1).tolist()
                    for j,c in enumerate(part):
                        rows.append({
                            "replica":rep,"arm":arm,"task":"select",
                            "seed":c["seed"],"base_world":c["base_world"],
                            "variant":c["variant"],"hash":c["hash"],
                            "n":c["n"],"source_mode":c["source_mode"],
                            "winner":c["winner"],"selected":selected[j],
                        })
        print("F10L_DEV_FROZEN",rep,arm,flush=True)
    assert len(rows)==5*512
    (out/"dev.json").write_text(json.dumps(rows,separators=(",",":")))
    print("F10L_DEV_WRITTEN_ALL_SHA_UNCHANGED",rep,len(rows),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--replica",required=True,type=int,choices=range(4))
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=False)
    frozen=train(a.replica,a.out)
    evaluate_dev(a.replica,a.out,frozen)

if __name__=="__main__":
    main()
