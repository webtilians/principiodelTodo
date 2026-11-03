"""F10-K reward-only primitive skills -> frozen zero-shot composition.

Only the environment_reward function and privileged D receive true answer.
REINFORCE A/B/C use ONLY a sampled action, sampled slot, past EMA baseline
and reward received after executing that action. No M composition in TRAIN.
"""
from __future__ import annotations
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical,Bernoulli

from observatory.f10k_worlds import dataset,TYPES,FEAT,MAX_SLOTS

ARMS=("joint_shared","staged_shared","staged_unshared","supervised_ceiling")
MODES=("read","no_flip","shuffle","oracle_slot")
TOTAL_EPISODES=24576
BATCH=64
LR=.003
ENTROPY=.002
EMA=.90

class SkillActor(nn.Module):
    def __init__(self,unshared=False):
        super().__init__()
        self.unshared=unshared
        # Score only metadata; NEVER the F4 value.
        self.selector=nn.Linear(FEAT,1)
        self.processor=nn.Sequential(nn.Linear(2,16),nn.Tanh(),nn.Linear(16,1))
        if unshared:
            self.memory_processor=nn.Sequential(
                nn.Linear(2,16),nn.Tanh(),nn.Linear(16,1))

    def compute(self,features,values,valid,public,flip,task,force_slot=None,
                no_flip=False,shuffle=False,deterministic=False):
        n=len(task)
        logits=self.selector(features).squeeze(-1)
        logits=logits.masked_fill(valid<.5,-1e9)
        # Public cases have no slots: insert a dummy distribution of one
        # valid placeholder, never used for processing or credit.
        public_case=(task==0)
        if public_case.any():
            logits=torch.where(public_case[:,None],
                               torch.cat((torch.zeros_like(logits[:,:1]),
                                          torch.full_like(logits[:,1:],-1e9)),dim=1),
                               logits)
        slot_distribution=Categorical(logits=logits)
        slot=(logits.argmax(-1) if deterministic else slot_distribution.sample())
        if force_slot is not None:
            slot=force_slot
        input_values=values
        if shuffle:
            input_values=values.clone()
            for i in range(n):
                if task[i]>0:
                    positions=torch.where(valid[i]>.5)[0]
                    input_values[i,positions]=values[i,positions.roll(1)]
        chosen=input_values[torch.arange(n),slot]
        observed=torch.where(public_case,(2*public-1),chosen)
        task_flip=torch.zeros_like(flip) if no_flip else flip
        processor_input=torch.stack((observed,2*task_flip-1),dim=1)
        logit=self.processor(processor_input).flatten()
        if self.unshared:
            memlogit=self.memory_processor(processor_input).flatten()
            logit=torch.where(public_case,logit,memlogit)
        action_dist=Bernoulli(logits=logit)
        action=(logit>=0).float() if deterministic else action_dist.sample()
        has_selection=(task==2)|(task==3)
        return {
            "action":action,"slot":slot,
            "action_logprob":action_dist.log_prob(action),
            "slot_logprob":slot_distribution.log_prob(slot)*has_selection,
            "entropy":action_dist.entropy()+
                       slot_distribution.entropy()*has_selection,
            "action_probability":torch.sigmoid(logit),
        }

def tensorize(rows):
    return (
        torch.from_numpy(np.stack([r["features"] for r in rows])),
        torch.from_numpy(np.stack([r["values"] for r in rows])),
        torch.from_numpy(np.stack([r["valid"] for r in rows])),
        torch.tensor([r["public_bit"] for r in rows],dtype=torch.float32),
        torch.tensor([r["flip"] for r in rows],dtype=torch.float32),
        torch.tensor([TYPES.index(r["task"]) for r in rows],dtype=torch.long),
        torch.tensor([r["truth"] for r in rows],dtype=torch.float32),
        torch.tensor([r["winner"] for r in rows],dtype=torch.long),
    )

def env_reward(action,truth):
    """Simulated environment ONLY; truth is not passed into actor/gradient."""
    return torch.where(action==truth,torch.ones_like(truth),-torch.ones_like(truth))

def order_for(arm,rep):
    # Exactly 8192 presentations per primitive type, equal in EVERY arm.
    p=np.arange(0,2048,dtype=np.int64)
    i=np.arange(2048,4096,dtype=np.int64)
    s=np.arange(4096,6144,dtype=np.int64)
    rng=np.random.default_rng(226000+rep)
    if arm=="joint_shared":
        sequence=np.tile(np.arange(6144,dtype=np.int64),4)
        rng.shuffle(sequence)
        return sequence
    # Curriculum: basic processors P+I -> version selection S -> mixed replay.
    phase1=np.concatenate((p,i))
    phase2=np.tile(s,2)
    phase3=np.concatenate((np.tile(p,3),np.tile(i,3),np.tile(s,2)))
    for phase in (phase1,phase2,phase3):
        rng.shuffle(phase)
    return np.concatenate((phase1,phase2,phase3))

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def train(rep,out):
    cases=dataset("train",rep)
    assert len(cases)==6144
    arrays=tensorize(cases)
    assert len(set(r["case_sha"] for r in cases))==6144
    meta={
        "experiment":"F10-K reward-learned primitives and never-trained composition",
        "status":"ALL_CHECKPOINTS_FROZEN_BEFORE_DEV_AND_OOD",
        "replica":rep,"train_counts":{"P":2048,"I":2048,"S":2048},
        "train_case_sha":[r["case_sha"] for r in cases],
        "world_namespace":{"train":22100000,"dev":22200000,"ood":22400000},
        "confirm_22300000_unopened":True,
        "never_train_memory_plus_flip_one":True,
        "lr":LR,"entropy":ENTROPY,"ema":EMA,"batch":BATCH,
        "numpy":np.__version__,"torch":torch.__version__,
        "python":platform.python_version(),"arms":{}
    }
    for arm in ARMS:
        torch.manual_seed(225000+rep)
        model=SkillActor(unshared=arm=="staged_unshared")
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        sequence=order_for(arm,rep)
        assert len(sequence)==TOTAL_EPISODES
        bytype=np.bincount(arrays[5][sequence].numpy(),minlength=4)
        assert bytype.tolist()==[8192,8192,8192,0]
        ema={0:0.,1:0.,2:0.}
        curves=[]
        t0=time.perf_counter()
        for batchid in range(0,TOTAL_EPISODES,BATCH):
            idx=sequence[batchid:batchid+BATCH]
            f,v,valid,p,flip,task,y,w=(a[idx] for a in arrays)
            outp=model.compute(f,v,valid,p,flip,task)
            received=env_reward(outp["action"],y)
            if arm=="supervised_ceiling":
                oracle=model.compute(f,v,valid,p,flip,task,force_slot=w)
                selected_logit=torch.logit(oracle["action_probability"].clamp(1e-6,1-1e-6))
                classification=nn.functional.binary_cross_entropy_with_logits(selected_logit,y)
                ce=(model.selector(f).squeeze(-1).masked_fill(valid<.5,-1e9))
                match=(task==2)
                selection=(nn.functional.cross_entropy(ce[match],w[match])
                           if match.any() else torch.tensor(0.,dtype=torch.float32))
                loss=classification+selection
            else:
                baseline=torch.tensor([ema[int(x)] for x in task],dtype=torch.float32)
                advantage=(received.detach()-baseline)
                loss=-(advantage*(outp["action_logprob"]+
                                   outp["slot_logprob"])).mean()-ENTROPY*outp["entropy"].mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            optimizer.step()
            for k in (0,1,2):
                mask=task==k
                if mask.any():
                    ema[k]=EMA*ema[k]+(1-EMA)*float(received[mask].mean())
            if (batchid//BATCH+1)%24==0:
                curves.append({
                    "updates":batchid//BATCH+1,
                    "observed_return":float(received.detach().mean()),
                    "sample_action_accuracy":float((outp["action"]==y).float().mean()),
                    "loss":float(loss.detach()),
                    "episodes_so_far":batchid+BATCH,
                })
        cp=out/(arm+".pt")
        torch.save(model.state_dict(),cp)
        meta["arms"][arm]={
            "checkpoint":cp.name,"sha256":sha(cp),
            "parameters":sum(x.numel() for x in model.parameters()),
            "rl_no_winner_or_truth_target":arm!="supervised_ceiling",
            "used_supervised_labels":arm=="supervised_ceiling",
            "episodes":TOTAL_EPISODES,"updates":TOTAL_EPISODES//BATCH,
            "episode_counts":{"P":8192,"I":8192,"S":8192,"M":0},
            "curves":curves,"seconds":time.perf_counter()-t0,
        }
        print("F10K_TRAIN_FROZEN",rep,arm,
              "sha",meta["arms"][arm]["sha256"][:12],
              "params",meta["arms"][arm]["parameters"],
              "episodes",TOTAL_EPISODES,flush=True)
        (out/"train.json").write_text(json.dumps(meta,indent=2))
    return meta

def evaluate(rep,out,meta):
    result=[]
    # Never instantiate a holdout generator until ALL FOUR models saved.
    assert all((out/(a+".pt")).exists() for a in ARMS)
    for split in ("dev","ood"):
        rows=dataset(split,rep)
        arrays=tensorize(rows)
        for arm in ARMS:
            cp=out/(arm+".pt")
            assert sha(cp)==meta["arms"][arm]["sha256"]
            model=SkillActor(unshared=(arm=="staged_unshared"))
            model.load_state_dict(torch.load(cp,weights_only=True,map_location="cpu"))
            model.eval()
            with torch.no_grad():
                for st in range(0,len(rows),128):
                    batch=rows[st:st+128]
                    f,v,valid,p,flip,task,y,w=(a[st:st+128] for a in arrays)
                    preds={mode:model.compute(
                        f,v,valid,p,flip,task,deterministic=True,
                        no_flip=mode=="no_flip",shuffle=mode=="shuffle",
                        force_slot=w if mode=="oracle_slot" else None)
                        for mode in MODES}
                    for j,row in enumerate(batch):
                        result.append({
                            "replica":rep,"split":split,"arm":arm,
                            "seed":row["seed"],"task":row["task"],"variant":row["variant"],
                            "case_sha":row["case_sha"],
                            "source_mode":row["source_mode"],
                            "n":row["n"],"flip":row["flip"],
                            "truth":row["truth"],"winner":row["winner"],
                            "public_bit":row["public_bit"],
                            "predictions":{
                                mode:{"action":int(o["action"][j]),
                                      "slot":int(o["slot"][j]),
                                      "p_action":float(o["action_probability"][j])}
                                for mode,o in preds.items()
                            },
                        })
            print("F10K_EVAL",rep,arm,split,len(rows),flush=True)
    assert len(result)==4*(768+512)
    (out/"eval.json").write_text(json.dumps(result,separators=(",",":")))
    print("F10K_ALL_FROZEN_EVALUATIONS",rep,len(result),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--replica",type=int,choices=range(4),required=True)
    parser.add_argument("--out",type=Path,required=True)
    a=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    a.out.mkdir(parents=True,exist_ok=False)
    meta=train(a.replica,a.out)
    evaluate(a.replica,a.out,meta)

if __name__=="__main__":
    main()
