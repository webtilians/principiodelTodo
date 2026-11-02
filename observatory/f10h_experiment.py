"""F10-H fixed-budget 4-way auxiliary supervision factorial (and GRU-only).

No F10-G datasets, checkpoints, DEV or OOD are imported. The architecture is
EXACTLY F10-G Actor('unary') (same 7458 learned parameters, no model change).
"""
from __future__ import annotations
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from observatory.f10g_models import Actor,count,gru_width
from observatory.f10h_worlds import dataset,KEYS,VARIANTS

ARMS=("joint","selection_only","interpretation_only","both","gru_only")
MODES=("read","skip","shuffle","erase_sign","erase_time","oracle_patch")
EPOCHS=20
BATCH=64
LR=.002

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def tensors(cases):
    return (
      torch.from_numpy(np.stack([r["history"] for r in cases])),
      torch.from_numpy(np.stack([r["memory"] for r in cases])),
      torch.tensor([r["truth"] for r in cases],dtype=torch.float32),
      torch.tensor([r["winner"] for r in cases],dtype=torch.long)
    )

def model_for(arm):
    return Actor("gru",gru_width()) if arm=="gru_only" else Actor("unary")

def train(replica,out):
    cases=dataset("train",replica)
    assert len(cases)==2048
    x,m,y,winner=tensors(cases)
    assert x.shape==(2048,49,30) and m.shape==(2048,48,29)
    report={
       "experiment":"F10-H four-way supervision factorial",
       "stage":"TRAIN_COMPLETED_BEFORE_DEV_OR_OOD",
       "replica":replica,"preregistered_train_worlds":512,
       "train_cases":len(cases),"epochs":EPOCHS,"batch":BATCH,"lr":LR,
       "torch":torch.__version__,"numpy":np.__version__,
       "python":platform.python_version(),
       "train_hashes":[case["hash"] for case in cases],
       "arms":{},
       "no_previous_checkpoint_or_training_examples":True,
       "confirm_19300000_opened":False,
    }
    for arm in ARMS:
        torch.manual_seed(195000+replica)
        model=model_for(arm)
        optimizer=torch.optim.Adam(model.parameters(),lr=LR)
        epochs_losses=[]
        t0=time.perf_counter()
        for epoch in range(EPOCHS):
            order=np.random.default_rng(196000+100*replica+epoch).permutation(len(y))
            total=0.
            for start in range(0,len(order),BATCH):
                ids=order[start:start+BATCH]
                score,logits,weights,hidden=model(x[ids],m[ids])
                loss=nn.functional.binary_cross_entropy_with_logits(score,y[ids])
                if arm in ("selection_only","both"):
                    loss=loss+nn.functional.cross_entropy(logits,winner[ids])
                if arm in ("interpretation_only","both"):
                    value=m[ids,winner[ids],KEYS]
                    isolated=model.interpret(
                        hidden,value,torch.ones_like(value),x[ids,-1,-1])
                    loss=loss+nn.functional.binary_cross_entropy_with_logits(isolated,y[ids])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),1.)
                optimizer.step()
                total+=float(loss.detach())*len(ids)
            epochs_losses.append(total/len(y))
            if (epoch+1)%5==0:
                print("F10H_TRAIN",replica,arm,epoch+1,
                      round(epochs_losses[-1],5),flush=True)
        path=out/(arm+".pt")
        torch.save(model.state_dict(),path)
        report["arms"][arm]={
            "parameters":count(model),
            "steps":EPOCHS*len(y)//BATCH,
            "examples":EPOCHS*len(y),
            "seconds":time.perf_counter()-t0,
            "losses":epochs_losses,
            "checkpoint":path.name,
            "sha256":file_sha(path),
            "decision_supervision":True,
            "selection_supervision":arm in ("selection_only","both"),
            "interpretation_supervision":arm in ("interpretation_only","both"),
        }
        (out/"training.json").write_text(json.dumps(report,indent=2))
    # All 5 final checkpoints exist on disk before the first OOD/DEV world.
    assert all((out/(arm+".pt")).exists() for arm in ARMS)
    return report

def evaluate(replica,out,metadata):
    rows=[]
    for split in ("dev","ood"):
        cases=dataset(split,replica)
        assert len(cases)==512
        x,m,y,winner=tensors(cases)
        for arm in ARMS:
            checkpoint=out/(arm+".pt")
            assert file_sha(checkpoint)==metadata["arms"][arm]["sha256"]
            model=model_for(arm)
            model.load_state_dict(torch.load(checkpoint,weights_only=True,map_location="cpu"))
            model.eval()
            with torch.no_grad():
                for start in range(0,len(cases),64):
                    batch=cases[start:start+64]
                    xx,mm,ww=x[start:start+64],m[start:start+64],winner[start:start+64]
                    pred={}
                    chosen=winner_weight=None
                    for mode in MODES:
                        patch=(mm[torch.arange(len(mm)),ww,KEYS]
                               if mode=="oracle_patch" else None)
                        score,logits,weights,_=model(xx,mm,mode,patch)
                        pred[mode]=(score>=0).int().tolist()
                        if mode=="read":
                            chosen=logits.argmax(-1).tolist()
                            winner_weight=weights[torch.arange(len(mm)),ww].tolist()
                    for offset,case in enumerate(batch):
                        public=case["memory"]
                        query=case["query"]
                        inv=case["inversion"]
                        slots=np.flatnonzero(public[:,query]>.5)
                        latest=slots[np.argmax(public[slots,KEYS+2])]
                        baselines={
                           "last_position":int(public[slots[-1],KEYS]>0)^inv,
                           "latest_ignore_signature":int(public[latest,KEYS]>0)^inv,
                           "last_signed_public":int(public[case["winner"],KEYS]>0)^inv,
                           "reproducible_coin":int(np.random.default_rng(case["seed"]+97).integers(2)),
                        }
                        rows.append({
                            "replica":replica,"split":split,"arm":arm,
                            "seed":case["seed"],"variant":case["variant"],
                            "k":case["k"],"signed_count":case["signed_count"],
                            "signed_value_changes":case["signed_value_changes"],
                            "source_mode":case["source_mode"],
                            "hash":case["hash"],"truth":case["truth"],
                            "winner":case["winner"],"selected":chosen[offset],
                            "winner_weight":winner_weight[offset],
                            "predictions":{mode:pred[mode][offset] for mode in MODES},
                            "baselines":baselines,
                        })
            assert file_sha(checkpoint)==metadata["arms"][arm]["sha256"]
        print("F10H_SPLIT_COMPLETE",replica,split,len(cases),flush=True)
    assert len(rows)==512*2*len(ARMS)
    (out/"rows.json").write_text(json.dumps(rows,separators=(",",":")))
    print("F10H_EVAL_COMPLETE",replica,"rows",len(rows),flush=True)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    args.output.mkdir(parents=True,exist_ok=False)
    metadata=train(args.index,args.output)
    evaluate(args.index,args.output,metadata)

if __name__=="__main__":
    main()
