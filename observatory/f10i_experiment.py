"""F10-I fixed-budget on-policy two-rollout experience experiment.

A: standard REINFORCE (EMA reward baseline).
B: paired on-policy REINFORCE with identical initial world and 2 trajectories.
C: B plus prediction of noisy POST-action verifier and intrinsic prediction bonus.
D: privileged supervised ceiling using same actor, never reported as RL success.

No old worlds or checkpoints imported; all five? FOUR arms frozen before DEV/OOD.
"""
from __future__ import annotations

import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn

from observatory.f10i_actor import ExperienceActor,count_parameters
from observatory.f10i_worlds import dataset,KEYS

ARMS=("rl_standard","rl_pair","rl_pair_predict","supervised_ceiling")
MODES=("read","force_read","force_skip","shuffle","erase_sign","erase_time")
EPOCHS=20
BATCH=64
LR=.002
GAMMA=.99
READ_COST=.05
ENTROPY=.005
EMA_MOMENTUM=.95
SENSOR_SUPERVISION=.30
INTRINSIC_WEIGHT=.10

def _reject_json_value(value):\n    raise TypeError(f"Unrecognized JSON value type: {type(value).__name__}")\n\ndef sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensors(cases):
    return (
        torch.from_numpy(np.stack([c["history"] for c in cases])),
        torch.from_numpy(np.stack([c["memory"] for c in cases])),
        torch.tensor([c["truth"] for c in cases],dtype=torch.float32),
        torch.tensor([c["winner"] for c in cases],dtype=torch.long),
        torch.tensor([c["sensor"] for c in cases],dtype=torch.float32),
        torch.tensor([c["delay"] for c in cases],dtype=torch.float32)
    )

def reward(sample,truth,delay):
    """Environment calculates outcome only AFTER sampled final action."""
    correct=(sample["action"]==truth).float()
    delayed=(2*correct-1.)*GAMMA**delay
    return delayed-READ_COST*sample["read"]

def train(rep,out):
    cases=dataset("train",rep)
    assert len(cases)==2048
    h,m,y,w,sensor,delay=tensors(cases)
    assert h.shape==(2048,49,30) and m.shape==(2048,48,29)
    report={
        "experiment":"INFINITO F10-I experiential paired reward, no winner labels in RL",
        "stage":"FROZEN_BEFORE_DEV_OOD",
        "replica":rep,"train_worlds":512,"train_cases":len(cases),
        "train_hashes":[c["hash"] for c in cases],
        "train_sensors_from_post_action_environment":True,
        "episodes_per_example_epoch":2,
        "epochs":EPOCHS,"batch":BATCH,"lr":LR,
        "read_cost":READ_COST,"gamma":GAMMA,
        "entropy":ENTROPY,"ema_momentum":EMA_MOMENTUM,
        "prediction_loss_weight":SENSOR_SUPERVISION,
        "prediction_intrinsic_weight":INTRINSIC_WEIGHT,
        "torch":torch.__version__,"numpy":np.__version__,
        "python":platform.python_version(),
        "no_old_training_or_checkpoint":True,
        "no_confirm_20300000":True,
        "arms":{},
    }
    for arm in ARMS:
        torch.manual_seed(205000+rep)
        model=ExperienceActor()
        model.train()
        opt=torch.optim.Adam(model.parameters(),lr=LR)
        ema=0.
        losses=[]
        learning_curve=[]
        interactions=0
        t0=time.perf_counter()
        for epoch in range(EPOCHS):
            order=np.random.default_rng(206000+100*rep+epoch).permutation(len(cases))
            total_loss=0.
            total_reward=0.
            total_correct=0.
            total_reads=0.
            for start in range(0,len(order),BATCH):
                indices=order[start:start+BATCH]
                hist,mem=h[indices],m[indices]
                truth,late=y[indices],delay[indices]
                # All arms consume exactly 2 independently sampled COMPLETE
                # episodes per world state. Only D has privileged loss targets.
                a=model.sample(hist,mem)
                b=model.sample(hist,mem)
                ra=reward(a,truth,late)
                rb=reward(b,truth,late)
                interactions+=2*len(indices)
                total_reward+=float((ra+rb).detach().sum())
                total_correct+=int(((a["action"]==truth).sum()+
                                    (b["action"]==truth).sum()).item())
                total_reads+=int((a["read"].sum()+b["read"].sum()).item())
                if arm=="supervised_ceiling":
                    loss=model.expert_objective(hist,mem,w[indices],truth)
                elif arm=="rl_standard":
                    # EMA baseline uses ONLY outcomes from PREVIOUS minibatches.
                    objective=-.5*(
                        ((ra-ema).detach()*a["log_prob"]).mean()+
                        ((rb-ema).detach()*b["log_prob"]).mean())
                    entropy=.5*(a["entropy"].mean()+b["entropy"].mean())
                    loss=objective-ENTROPY*entropy
                    batch_mean=float(torch.cat((ra,rb)).detach().mean())
                    ema=EMA_MOMENTUM*ema+(1-EMA_MOMENTUM)*batch_mean
                else:
                    if arm=="rl_pair_predict":
                        verifier=sensor[indices]
                        la=nn.functional.binary_cross_entropy_with_logits(
                            a["sensor_logits"],verifier,reduction="none")
                        lb=nn.functional.binary_cross_entropy_with_logits(
                            b["sensor_logits"],verifier,reduction="none")
                        # Intrinsic advantage from ONLY observed future
                        # verification; never access winner/truth labels here.
                        rpa=ra-INTRINSIC_WEIGHT*la.detach()
                        rpb=rb-INTRINSIC_WEIGHT*lb.detach()
                    else:
                        rpa,rpb=ra,rb
                    objective=-.5*((rpa-rpb).detach()*(
                        a["log_prob"]-b["log_prob"])).mean()
                    entropy=.5*(a["entropy"].mean()+b["entropy"].mean())
                    loss=objective-ENTROPY*entropy
                    if arm=="rl_pair_predict":
                        loss=loss+SENSOR_SUPERVISION*.5*(la.mean()+lb.mean())
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step()
                total_loss+=float(loss.detach())*len(indices)
            assert interactions==(epoch+1)*2*len(cases)
            learning_curve.append({
                "epoch":epoch+1,
                "mean_training_return":total_reward/(2*len(cases)),
                "sampled_action_accuracy":total_correct/(2*len(cases)),
                "read_fraction":total_reads/(2*len(cases)),
                "loss":total_loss/len(cases),
                "cumulative_rollouts":interactions,
            })
            losses.append(total_loss/len(cases))
            if (epoch+1)%5==0:
                print("F10I_TRAIN",rep,arm,epoch+1,
                      "reward",round(learning_curve[-1]["mean_training_return"],5),
                      "accuracy",round(learning_curve[-1]["sampled_action_accuracy"],5),
                      "reads",round(learning_curve[-1]["read_fraction"],5),
                      "loss",round(losses[-1],5),flush=True)
        checkpoint=out/(arm+".pt")
        torch.save(model.state_dict(),checkpoint)
        report["arms"][arm]={
            "checkpoint":checkpoint.name,
            "sha256":sha(checkpoint),
            "parameters":count_parameters(model),
            "optimizer_steps":EPOCHS*len(cases)//BATCH,
            "train_state_presentations":EPOCHS*len(cases),
            "environment_rollouts":interactions,
            "environment_verifier_observations":interactions,
            "supervision":"expert_index_and_action" if arm=="supervised_ceiling"
                           else ("environment_reward_and_noisy_verifier"
                               if arm=="rl_pair_predict" else "terminal_environment_reward_only"),
            "used_winner_label_in_loss":arm=="supervised_ceiling",
            "used_action_label_in_loss":arm=="supervised_ceiling",
            "losses":losses,
            "learning_curve":learning_curve,
            "seconds":time.perf_counter()-t0,
        }
        (out/"training.json").write_text(json.dumps(report,indent=2))
    assert all((out/(arm+".pt")).exists() for arm in ARMS)
    return report

def evaluate(rep,out,training):
    rows=[]
    for split in ("dev","ood"):
        cases=dataset(split,rep)
        assert len(cases)==512
        h,m,y,w,sensor,delay=tensors(cases)
        for arm in ARMS:
            cp=out/(arm+".pt")
            assert sha(cp)==training["arms"][arm]["sha256"]
            model=ExperienceActor()
            model.load_state_dict(torch.load(cp,map_location="cpu",weights_only=True))
            model.eval()
            with torch.no_grad():
                for start in range(0,len(cases),64):
                    batch=cases[start:start+64]
                    hh,mm=h[start:start+64],m[start:start+64]
                    preds={}
                    for mode in MODES:
                        o=model.predict(hh,mm,mode)
                        preds[mode]={k:v.tolist() for k,v in o.items()}
                    for j,case in enumerate(batch):
                        mem=case["memory"]
                        query=case["q"]
                        valid=np.flatnonzero(mem[:,query]>.5)
                        latest_all=valid[np.argmax(mem[valid,KEYS+2])]
                        coin=int(np.random.default_rng(case["seed"]+101).integers(2))
                        baselines={
                            "last_storage":int(mem[valid[-1],KEYS]>0)^case["inv"],
                            "latest_unsigned":int(mem[latest_all,KEYS]>0)^case["inv"],
                            "coin":coin,
                            "last_signed_rule":int(mem[case["winner"],KEYS]>0)^case["inv"],
                        }
                        row={
                            "replica":rep,"arm":arm,"split":split,
                            "seed":case["seed"],"variant":case["variant"],
                            "k":case["k"],"signed_count":case["signed_count"],
                            "signed_value_changes":case["signed_value_changes"],
                            "source_mode":case["source_mode"],
                            "hash":case["hash"],"winner":case["winner"],
                            "truth":case["truth"],"sensor":case["sensor"],
                            "delay":case["delay"],
                            "prediction":{mode:preds[mode]["action"][j] for mode in MODES},
                            "greedy_read":preds["read"]["read"][j],
                            "read_probability":preds["read"]["read_probability"][j],
                            "selected_slot":preds["read"]["selected"][j],
                            "baselines":baselines,
                        }
                        assert baselines["last_signed_rule"]==case["truth"]
                        rows.append(row)
            assert sha(cp)==training["arms"][arm]["sha256"]
        print("F10I_SPLIT_COMPLETE",rep,split,len(cases),flush=True)
    assert len(rows)==4*2*512
    (out/"rows.json").write_text(json.dumps(rows,separators=(",",":"),\n        default=lambda value: value.item() if isinstance(value,np.generic) else _reject_json_value(value)))
    print("F10I_ALL_FROZEN_RESULTS",rep,len(rows),flush=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--index",type=int,choices=range(4),required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=False)
    trained=train(a.index,out)
    evaluate(a.index,out,trained)

if __name__=="__main__":
    main()
