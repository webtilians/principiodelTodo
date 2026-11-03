"""F10-J experience-only structured information exploration.

Five SAME-architecture arms. All A/B/C/D use only outcomes of executed
actions, no winner or truth supervised targets; E has privileged labels.
B/C/D forced gate actions are interventions and DO NOT masquerade as
on-policy gate samples. Gate value is fitted from READ−SKIP sampled outcome.
"""
from __future__ import annotations

import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Bernoulli,Categorical

from observatory.f10i_actor import ExperienceActor,count_parameters
from observatory.f10j_worlds import dataset,KEYS,EVENTS,FEATURES

ARMS=("on_policy","forced_gate","forced_gate_slot_explore","paired_credit","supervised_ceiling")
MODES=("read","force_read","force_skip","shuffle","erase_sign","erase_time")
EPOCHS=20
BATCH=64
LR=.002
GAMMA=.99
READ_COST=.05
ENTROPY=.005
EMA_MOMENTUM=.95
SLOT_EPS=.5
REWARD_NOISE=.10

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def data_tensors(cases):
    return (
        torch.from_numpy(np.stack([c["history"] for c in cases])),
        torch.from_numpy(np.stack([c["memory"] for c in cases])),
        torch.tensor([c["truth"] for c in cases],dtype=torch.float32),
        torch.tensor([c["winner"] for c in cases],dtype=torch.long),
        torch.tensor([float(c["informative"]) for c in cases]),
        torch.tensor([c["delay"] for c in cases],dtype=torch.float32),
    )

def rollout(model,hist,mem,gate=None,slot_mix=False):
    h,inv,valid,logits,gate_logits=model.encode(hist,mem)
    gates=Bernoulli(logits=gate_logits)
    if gate is None:
        g=gates.sample()
    else:
        g=torch.full_like(gate_logits,float(gate))
    probs=torch.softmax(logits,dim=-1)
    if slot_mix:
        uniform=valid.to(probs.dtype)/valid.sum(dim=-1,keepdim=True)
        probs=(1-SLOT_EPS)*probs+SLOT_EPS*uniform
    selector=Categorical(probs=probs)
    idx=selector.sample()
    value=mem[torch.arange(len(mem)),idx,KEYS]*g
    action_logits,_=model.interpret(h,inv,value,g)
    action_dist=Bernoulli(logits=action_logits)
    act=action_dist.sample()
    return {
        "gate":g,"slot":idx,"action":act,
        "gate_log_prob":gates.log_prob(g),
        "slot_log_prob":selector.log_prob(idx),
        "action_log_prob":action_dist.log_prob(act),
        "gate_probability":torch.sigmoid(gate_logits),
        "entropy":gates.entropy()+g*selector.entropy()+action_dist.entropy(),
    }

def reward(s,truth,delay,flip):
    # Truth is used ONLY by the simulated ENVIRONMENT and privileged E.
    correct=(s["action"]==truth).float()
    observed_correct=torch.where(flip,1-correct,correct)
    return (2*observed_correct-1)*GAMMA**delay-READ_COST*s["gate"]

def expert_loss(model,hist,mem,winner,truth,informative):
    h,inv,valid,selection_logits,gate_logit=model.encode(hist,mem)
    g=informative
    value=mem[torch.arange(len(mem)),winner,KEYS]*g
    action_logits,_=model.interpret(h,inv,value,g)
    selector_loss=nn.functional.cross_entropy(
        selection_logits,winner,reduction="none")
    return (
        nn.functional.binary_cross_entropy_with_logits(gate_logit,g)+
        (selector_loss*g).sum()/g.sum().clamp_min(1.)+
        nn.functional.binary_cross_entropy_with_logits(action_logits,truth)
    )

def train(rep,out):
    cases=dataset("train",rep)
    assert len(cases)==2048
    h,m,truth,winner,informative,delay=data_tensors(cases)
    assert h.shape==(2048,49,30) and m.shape==(2048,48,29)
    assert sum(c["informative"] for c in cases)==1024
    assert torch.all(h[:,:EVENTS,KEYS]==0)
    meta={
        "experiment":"F10-J information-value exploration from consequences",
        "status":"FROZEN_BEFORE_DEV_OOD",
        "replica":rep,"train_worlds":512,"train_cases":2048,
        "train_hashes":[c["hash"] for c in cases],
        "same_parameter_model":"observatory.f10i_actor.ExperienceActor",
        "truth_used_for_rl_targets":False,
        "winner_used_for_rl_targets":False,
        "labels_only_for_privileged_ceiling":True,
        "confirm_21300000_unopened":True,
        "epochs":EPOCHS,"batch":BATCH,"lr":LR,"gamma":GAMMA,
        "read_cost":READ_COST,"entropy":ENTROPY,
        "ema_momentum":EMA_MOMENTUM,"slot_explore_epsilon":SLOT_EPS,
        "train_reward_flip_probability":REWARD_NOISE,
        "two_executed_episodes_per_state":True,
        "torch":torch.__version__,"numpy":np.__version__,
        "python":platform.python_version(),"arms":{},
    }
    for arm in ARMS:
        torch.manual_seed(215000+rep)
        model=ExperienceActor()
        model.train()
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        ema=0.
        episodes=0
        curves=[]
        t0=time.perf_counter()
        for epoch in range(EPOCHS):
            order=np.random.default_rng(216000+100*rep+epoch).permutation(len(cases))
            total_reward=0.
            total_acc=0.
            total_reads=0.
            total_loss=0.
            for start in range(0,len(order),BATCH):
                ids=order[start:start+BATCH]
                hh,mm=h[ids],m[ids]
                yy,dd=truth[ids],delay[ids]
                # Exactly one COMMON exogenous corruption per state, shared
                # across both executed actions and all arms.
                noise_seed=218000+rep*100000+epoch*1000+start//BATCH
                flip=torch.tensor(
                    np.random.default_rng(noise_seed).random(len(ids))<REWARD_NOISE,
                    dtype=torch.bool)
                if arm in ("on_policy","supervised_ceiling"):
                    a=rollout(model,hh,mm)
                    b=rollout(model,hh,mm)
                else:
                    a=rollout(model,hh,mm,gate=1,
                              slot_mix=arm in ("forced_gate_slot_explore","paired_credit"))
                    b=rollout(model,hh,mm,gate=0,
                              slot_mix=arm in ("forced_gate_slot_explore","paired_credit"))
                ra=reward(a,yy,dd,flip)
                rb=reward(b,yy,dd,flip)
                episodes+=2*len(ids)
                total_reward+=float((ra+rb).detach().sum())
                total_acc+=int((a["action"]==yy).sum().item()+
                               (b["action"]==yy).sum().item())
                total_reads+=int(a["gate"].sum().item()+b["gate"].sum().item())

                if arm=="supervised_ceiling":
                    loss=expert_loss(model,hh,mm,winner[ids],yy,informative[ids])
                elif arm=="on_policy":
                    scores=(a["gate_log_prob"]+a["gate"]*a["slot_log_prob"]+
                            a["action_log_prob"],
                            b["gate_log_prob"]+b["gate"]*b["slot_log_prob"]+
                            b["action_log_prob"])
                    objective=-.5*((ra.detach()-ema)*scores[0]+
                                   (rb.detach()-ema)*scores[1]).mean()
                    loss=objective-ENTROPY*.5*(a["entropy"]+b["entropy"]).mean()
                else:
                    delta=(ra-rb).detach()
                    # Gate learned from real randomized information intervention.
                    # Under a Bernoulli gate policy dE[R]/dtheta is
                    # E[R_read-R_skip]*d sigmoid(gate_logit)/dtheta.
                    gate_objective=-(delta*a["gate_probability"]).mean()
                    if arm=="paired_credit":
                        objective=-.5*(delta*(a["slot_log_prob"]+a["action_log_prob"]-
                                                 b["action_log_prob"])).mean()
                    else:
                        objective=-.5*(
                            (ra.detach()-ema)*(a["slot_log_prob"]+a["action_log_prob"])+
                            (rb.detach()-ema)*b["action_log_prob"]
                        ).mean()
                    loss=gate_objective+objective-ENTROPY*.5*(
                        a["entropy"]+b["entropy"]).mean()

                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                batch_mean=float(torch.cat((ra,rb)).detach().mean())
                ema=EMA_MOMENTUM*ema+(1-EMA_MOMENTUM)*batch_mean
                total_loss+=float(loss.detach())*len(ids)
            assert episodes==(epoch+1)*2*len(cases)
            curves.append({
                "epoch":epoch+1,"environment_rollouts":episodes,
                "mean_observed_reward":total_reward/(2*len(cases)),
                "action_accuracy_clean_diagnostic":total_acc/(2*len(cases)),
                "sampled_read_fraction":total_reads/(2*len(cases)),
                "objective":total_loss/len(cases),
            })
            if epoch in (0,4,9,14,19):
                print("F10J_TRAIN",rep,arm,epoch+1,
                      "return",round(curves[-1]["mean_observed_reward"],4),
                      "clean_accuracy",round(curves[-1]["action_accuracy_clean_diagnostic"],4),
                      "reads",round(curves[-1]["sampled_read_fraction"],4),
                      flush=True)
        cp=out/(arm+".pt")
        torch.save(model.state_dict(),cp)
        meta["arms"][arm]={
            "checkpoint":cp.name,"sha256":sha(cp),
            "parameters":count_parameters(model),
            "optimizer_steps":640,"train_presentations":40960,
            "executed_environment_episodes":episodes,
            "privileged_labels":arm=="supervised_ceiling",
            "exploration":"on_policy" if arm in ("on_policy","supervised_ceiling")
               else "one_observed_READ_and_one_observed_SKIP",
            "slot_behavior_epsilon":SLOT_EPS if arm in ("forced_gate_slot_explore","paired_credit") else 0.,
            "curves":curves,"wall_seconds":time.perf_counter()-t0,
        }
        (out/"training.json").write_text(json.dumps(meta,indent=2))
    assert len(meta["arms"])==5
    print("F10J_ALL_5_ARMS_FROZEN_BEFORE_HOLDOUT",rep,flush=True)
    return meta

def evaluate(rep,out,meta):
    rows=[]
    for split in ("dev","ood"):
        cases=dataset(split,rep)
        assert len(cases)==512
        h,m,truth,winner,informative,delay=data_tensors(cases)
        assert torch.all(h[:,:EVENTS,KEYS]==0)
        for arm in ARMS:
            cp=out/(arm+".pt")
            assert sha(cp)==meta["arms"][arm]["sha256"]
            model=ExperienceActor()
            model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True))
            model.eval()
            with torch.no_grad():
                for start in range(0,len(cases),64):
                    batch=cases[start:start+64]
                    hh,mm=h[start:start+64],m[start:start+64]
                    preds={mode:model.predict(hh,mm,mode) for mode in MODES}
                    for j,case in enumerate(batch):
                        outs={mode:{k:float(v[j]) if k.endswith("probability") else int(v[j])
                                    for k,v in pred.items()}
                              for mode,pred in preds.items()}
                        row={
                            "replica":rep,"arm":arm,"split":split,
                            "seed":case["seed"],"variant":case["variant"],
                            "informative":case["informative"],"k":case["k"],
                            "signed_count":case["signed_count"],
                            "signed_changes":case["signed_changes"],
                            "source_mode":case["source_mode"],
                            "hash":case["hash"],"winner":case["winner"],
                            "truth":case["truth"],"delay":case["delay"],
                            "xor":case["xor"],"public_bit":case["public_bit"],
                            "predictions":outs,
                            "baseline_public":case["public_bit"]^case["xor"],
                        }
                        rows.append(row)
            assert sha(cp)==meta["arms"][arm]["sha256"]
        print("F10J_HOLDOUT_SPLIT_COMPLETE",rep,split,len(cases),flush=True)
    assert len(rows)==2*5*512
    (out/"rows.json").write_text(json.dumps(rows,separators=(",",":")))
    print("F10J_FROZEN_RESULTS_SAVED",rep,len(rows),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--replica",type=int,choices=range(4),required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.out.mkdir(parents=True,exist_ok=False)
    meta=train(args.replica,args.out)
    evaluate(args.replica,args.out,meta)

if __name__=="__main__":
    main()
