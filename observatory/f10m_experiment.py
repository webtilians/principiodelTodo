"""INFINITO F10-M: learn CHOICE OF COMPLETE OPERATION PROGRAM by real trials.

Original F10-L selector and XOR are SHA-authenticated and FROZEN. Only the
new program-selection controller is trainable. The controller observes
(source, flip, public bit) and gets observed returns for TWO actually
executed chosen programs; never an optimal-program label or reward for
programs not executed.
"""
from __future__ import annotations

import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from observatory.f10l_train import SelectActor,XorActor,sha as sha_file
from observatory.f10m_worlds import (
    dataset,PROGRAMS,program_features,evaluate_program,context,VARIANTS,
)

ORIGINAL_RUN_ID=41685499939
ORIGINAL_CODE_SHA="b5c32cbfb50a24414607e3a24e30e139ca92fede"
ARMS=("global_bandit","context_mlp","factorized","paired_factorized")
EVAL_MODES=("normal","shuffle","x_disabled","public_randomized")
EPS=.20
EPOCHS=16
BATCH=64
LR=.01
ENTROPY=.002
EMA=.90
N_PROGRAMS=85

class GlobalBandit(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit=nn.Parameter(torch.zeros(N_PROGRAMS))
    def forward(self,obs):
        return self.logit.expand(len(obs),-1)

class ContextMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3,32),nn.Tanh(),nn.Linear(32,N_PROGRAMS))
    def forward(self,obs):
        return self.net(obs)

class Factorized(nn.Module):
    def __init__(self):
        super().__init__()
        self.theta=nn.Parameter(torch.zeros(3,33))
        self.register_buffer("features",torch.from_numpy(program_features()))
    def forward(self,obs):
        # 3 additive components: base + source effect + flip effect,
        # deliberately NO source*flip interaction or optimal-program labels.
        weights=self.theta[0].unsqueeze(0)+(
            obs[:,0:1]*self.theta[1].unsqueeze(0))+(
            obs[:,1:2]*self.theta[2].unsqueeze(0))
        return weights@self.features.T

def make_actor(arm):
    if arm=="global_bandit":return GlobalBandit()
    if arm=="context_mlp":return ContextMLP()
    if arm in ("factorized","paired_factorized"):return Factorized()
    raise ValueError(arm)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def frozen_original_models(rep,original_dir,g0):
    """Hash-authenticate the ORIGINAL 2016? No: 2026 F10-L artifacts."""
    meta=json.loads((original_dir/"train.json").read_text())
    assert meta["replica"]==rep and meta["train_composition_exposure"]==0
    assert meta["status"]=="FROZEN_BEFORE_DEV_AND_ANY_OOD"
    assert g0["technical"]=="F10L_PHASE1_TECHNICAL_COMPLETE"
    assert g0["eligible_for_composition"]
    assert g0["composition_23400000_not_opened"]
    assert len(meta["arms"])==5
    for name,info in meta["arms"].items():
        cp=original_dir/(name+".pt")
        assert sha_file(cp)==info["sha256"]
        assert info["sha256"]==g0["replicas"][str(rep)]["model_sha256"][name]
        assert info["updates"]==512
        assert info["executed_environment_episodes"]==65536
        assert not info["loss_supervised_winner_or_answer"]
    selected=SelectActor()
    xor=XorActor()
    selected.load_state_dict(torch.load(original_dir/"sel_uniform_pair.pt",
                                         map_location="cpu",weights_only=True))
    xor.load_state_dict(torch.load(original_dir/"xor_two_actions.pt",
                                    map_location="cpu",weights_only=True))
    selected.eval()
    xor.eval()
    for model in (selected,xor):
        for p in model.parameters():
            p.requires_grad_(False)
    with torch.no_grad():
        inp=torch.tensor([[-1.,1.],[1.,1.]],dtype=torch.float32)
        original_logits=xor(inp)
        xor_lut=tuple(int(z>=0) for z in original_logits)
    return selected,xor,xor_lut,{
        "original_run_id":ORIGINAL_RUN_ID,
        "original_code_sha":ORIGINAL_CODE_SHA,
        "original_checkpoint_sha":{
            a:meta["arms"][a]["sha256"]
            for a in ("sel_uniform_pair","xor_two_actions")
        },
        "all_5_original_checkpoint_sha":{
            a:meta["arms"][a]["sha256"] for a in meta["arms"]
        },
        "pretrained_xor_bit_flip1_lut":xor_lut,
    }

def selection_cache(rows,selector):
    """Cache only selected slot from real F10-L model, never winning truth."""
    all_choices=[]
    selector.eval()
    with torch.no_grad():
        for start in range(0,len(rows),256):
            group=rows[start:start+256]
            f=torch.from_numpy(np.stack([c["features"] for c in group]))
            valid=torch.from_numpy(np.stack([c["mask"] for c in group]))
            picked=selector(f,valid).argmax(-1).tolist()
            all_choices.extend(int(x) for x in picked)
    assert len(all_choices)==len(rows)
    assert all(c["mask"][slot]>.5 for c,slot in zip(rows,all_choices))
    return all_choices

def behavior_distribution(logits):
    pure=torch.softmax(logits,dim=-1)
    mixed=(1-EPS)*pure+EPS/N_PROGRAMS
    return pure,mixed

def execute_sample(rows,choices,ids,program_ids,xor_lut):
    """Only execute the actual sampled program per state. No gold program."""
    return [
        evaluate_program(rows[int(i)],PROGRAMS[int(p)],choices[int(i)],xor_lut)
        for i,p in zip(ids,program_ids)
    ]

def train(rep,out,rows,choices,xor_lut,original):
    assert len(rows)==3072
    assert all((r["task"],r["flip"]) in (("PUBLIC",0),("PUBLIC",1),("MEMORY",0))
               for r in rows)
    assert len(set(r["case_sha"] for r in rows))==3072
    obs=torch.from_numpy(np.stack([context(r) for r in rows]))
    metadata={
        "experiment":"F10-M experience-only composition-sequence discovery",
        "status":"FROZEN_BEFORE_DEV_AND_OOD",
        "replica":rep,
        "no_pretrained_model_training":True,
        "no_optimal_program_labels":True,
        "no_unexecuted_program_reward_queries":True,
        "unseen_memory_flip1_never_train":True,
        "confirm_25300000_sealed":True,
        "train_worlds_per_context":256,
        "train_case_hashes":[r["case_sha"] for r in rows],
        "train_context_counts":{
            f"{a}_{b}":sum(r["task"]==a and r["flip"]==b for r in rows)
            for a,b in (("PUBLIC",0),("PUBLIC",1),("MEMORY",0))
        },
        "original_skills":original,
        "program_count":len(PROGRAMS),
        "program_list":["".join(p) or "EMPTY" for p in PROGRAMS],
        "epochs":EPOCHS,"batch":BATCH,"lr":LR,"epsilon":EPS,
        "entropy":ENTROPY,"ema":EMA,
        "torch":torch.__version__,"numpy":np.__version__,
        "python":platform.python_version(),
        "arms":{},
    }
    assert list(metadata["train_context_counts"].values())==[1024,1024,1024]
    for arm in ARMS:
        torch.manual_seed(255000+rep)
        actor=make_actor(arm)
        optimizer=torch.optim.Adam(actor.parameters(),lr=LR)
        ema=0.
        executed=0
        all_curves=[]
        t0=time.perf_counter()
        initial_sha=hashlib.sha256(
            b"".join(p.detach().numpy().tobytes() for p in actor.parameters())
        ).hexdigest()
        for epoch in range(EPOCHS):
            idx=np.random.default_rng(256000+100*rep+epoch).permutation(3072)
            rewards_sum=0.
            accuracy_sum=0
            invalid_sum=0
            mean_loss=0.
            for start in range(0,3072,BATCH):
                ids=idx[start:start+BATCH]
                logits=actor(obs[ids])
                pure,mixed=behavior_distribution(logits)
                sampler=Categorical(probs=mixed)
                a=sampler.sample()
                b=sampler.sample()
                ea=execute_sample(rows,choices,ids,a.tolist(),xor_lut)
                eb=execute_sample(rows,choices,ids,b.tolist(),xor_lut)
                ra=torch.tensor([x["return"] for x in ea],dtype=torch.float32)
                rb=torch.tensor([x["return"] for x in eb],dtype=torch.float32)
                executed+=2*len(ids)
                entropy=-(pure*torch.log(pure.clamp_min(1e-12))).sum(-1).mean()
                if arm=="paired_factorized":
                    delta=(ra-rb).detach()
                    loss=-.5*(delta*(sampler.log_prob(a)-
                                      sampler.log_prob(b))).mean()-ENTROPY*entropy
                else:
                    loss=-.5*((ra.detach()-ema)*sampler.log_prob(a)+
                               (rb.detach()-ema)*sampler.log_prob(b)).mean()
                    loss=loss-ENTROPY*entropy
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(),1.)
                optimizer.step()
                ema=EMA*ema+(1-EMA)*float(torch.cat((ra,rb)).mean())
                rewards_sum+=float(ra.sum()+rb.sum())
                accuracy_sum+=sum(x["success"] for x in ea+eb)
                invalid_sum+=sum(x["invalid"] for x in ea+eb)
                mean_loss+=float(loss.detach())*len(ids)
            assert executed==(epoch+1)*2*len(rows)
            all_curves.append({
                "epoch":epoch+1,"executed_episodes":executed,
                "sampled_correct_fraction":accuracy_sum/(2*len(rows)),
                "sampled_return":rewards_sum/(2*len(rows)),
                "invalid_per_episode":invalid_sum/(2*len(rows)),
                "objective":mean_loss/len(rows),
            })
            if epoch in (0,3,7,11,15):
                print("F10M_TRAIN",rep,arm,epoch+1,
                      "correct",round(all_curves[-1]["sampled_correct_fraction"],4),
                      "return",round(all_curves[-1]["sampled_return"],4),
                      "invalid",round(all_curves[-1]["invalid_per_episode"],4),
                      flush=True)
        cp=out/(arm+".pt")
        torch.save(actor.state_dict(),cp)
        metadata["arms"][arm]={
            "checkpoint":cp.name,"sha256":sha(cp),
            "initial_trainable_sha":initial_sha,
            "parameters":sum(p.numel() for p in actor.parameters()),
            "epochs":EPOCHS,"updates":768,
            "executed_environment_episodes":executed,
            "train_presentations":EPOCHS*len(rows),
            "no_supervised_labels":True,
            "curves":all_curves,
            "wall_seconds":time.perf_counter()-t0,
        }
        (out/"training.json").write_text(json.dumps(metadata,indent=2))
    assert len(metadata["arms"])==4
    return metadata

def evaluate(rep,out,meta,selector,xor_lut):
    """DEV/OOD are FIRST generated AFTER all four SHA checkpoints written."""
    assert all((out/(arm+".pt")).exists() for arm in ARMS)
    assert all(sha(out/(arm+".pt"))==meta["arms"][arm]["sha256"]
               for arm in ARMS)
    result=[]
    for split in ("dev","ood"):
        cases=dataset(split,rep)
        assert len(cases)==(768 if split=="dev" else 512)
        choices=selection_cache(cases,selector)
        contexts=torch.from_numpy(np.stack([context(c) for c in cases]))
        for arm in ARMS:
            model=make_actor(arm)
            cp=out/(arm+".pt")
            assert sha(cp)==meta["arms"][arm]["sha256"]
            model.load_state_dict(torch.load(cp,weights_only=True,map_location="cpu"))
            model.eval()
            with torch.no_grad():
                logits=model(contexts)
                program_ids=logits.argmax(-1).tolist()
            for i,(case,pid) in enumerate(zip(cases,program_ids)):
                sequence=PROGRAMS[pid]
                outcomes={mode:evaluate_program(case,sequence,choices[i],
                                                 xor_lut,mode)
                          for mode in EVAL_MODES}
                result.append({
                    "replica":rep,"arm":arm,"split":split,
                    "seed":case["seed"],"base_world":case["base_world"],
                    "case_sha":case["case_sha"],
                    "task":case["task"],"flip":case["flip"],
                    "public_bit":case["public_bit"],
                    "source_mode":case["source_mode"],"n":case["n"],
                    "variant":case["variant"],
                    "truth":case["truth"],"winner":case["winner"],
                    "frozen_selected":choices[i],
                    "program_id":pid,
                    "program":"".join(sequence) or "EMPTY",
                    "outcomes":{key:{
                        "answer":v["answer"],"success":v["success"],
                        "return":v["return"],"invalid":v["invalid"],
                        "operations":v["operations"],"read":v["read"],
                    } for key,v in outcomes.items()},
                })
            assert sha(cp)==meta["arms"][arm]["sha256"]
        print("F10M_FROZEN_EVAL",rep,split,len(cases),flush=True)
    assert len(result)==4*(768+512)
    (out/"rows.json").write_text(json.dumps(result,separators=(",",":")))
    print("F10M_CHECKPOINTS_UNTOUCHED_AND_EVAL_COMPLETE",rep,len(result),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--replica",type=int,choices=range(4),required=True)
    p.add_argument("--original",type=Path,required=True)
    p.add_argument("--g0",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=False)
    g0=json.loads(a.g0.read_text())
    selector,xor,xor_lut,original=frozen_original_models(
        a.replica,a.original,g0)
    # F10-L bit XOR flip1 was learned perfectly on all four replicas.
    assert xor_lut==(1,0), "Original frozen XOR not the predicted swap"
    original_pre_sha={k:sha(a.original/(k+".pt"))
                      for k in original["all_5_original_checkpoint_sha"]}
    train_rows=dataset("train",a.replica)
    selected=selection_cache(train_rows,selector)
    frozen=train(a.replica,a.out,train_rows,selected,xor_lut,original)
    evaluate(a.replica,a.out,frozen,selector,xor_lut)
    assert all(sha(a.original/(k+".pt"))==v
               for k,v in original_pre_sha.items())
    print("F10M_ORIGINAL_F10L_20_SKILL_SHAS_UNCHANGED",a.replica,flush=True)

if __name__=="__main__":
    main()
