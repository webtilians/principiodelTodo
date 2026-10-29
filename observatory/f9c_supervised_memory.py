"""F9-C supervised POSITIVE CONTROL: learn to remember visible MiniGrid objects.

This does NOT make PPO or LSTM RL competent at original navigation. Labels
are used only by supervised TRAINING loss, never supplied at inference. Inputs
are the original partial symbolic images, converted into generic visible
key/ball occupancy channels for every pixel (no oracle cue/goal inputs).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from observatory.f9_external_minigrid import file_sha, seed_for
from observatory.f9c_visible_cue import make_env


def encode(observation):
    """Observable key/ball pixels ONLY; no world coordinates or hidden labels."""
    import numpy as np
    from minigrid.core.constants import OBJECT_TO_IDX
    values = np.rint(observation[:147].reshape(49,3)[:,0] *
                     max(OBJECT_TO_IDX.values())).astype(int)
    return np.stack((values == OBJECT_TO_IDX["key"],
                     values == OBJECT_TO_IDX["ball"]),axis=-1).astype("float32").reshape(-1)


def frames_and_label(seed, size=7, reverse=False):
    import numpy as np
    env = make_env(size, reverse_cue=reverse)
    try:
        start, _ = env.reset(seed=seed)
        fork, _, done, _, _ = env.step(0)
        if done:
            raise ValueError("Memory assay ended before decision")
        truth = env.inner.labels["correct_side"]
        assert truth in ("upper", "lower")
        return np.stack((encode(start),encode(fork)),axis=0), (
            0 if truth == "upper" else 1)
    finally:
        env.close()


def make_dataset(split, count, size=7, *, paired=False):
    import numpy as np
    samples, labels = [], []
    for j in range(count):
        seed = seed_for(split, 0, j)
        modes = (False,True) if paired else (False,)
        for reverse in modes:
            x, y = frames_and_label(seed,size,reverse)
            samples.append(x)
            labels.append(y)
    return np.stack(samples), np.array(labels,dtype="int64")


def evaluate(model, size, episodes):
    import torch
    model.eval()
    records = []
    with torch.no_grad():
        for j in range(episodes):
            seed=seed_for("development",0,j)
            pair=[]
            for reverse in (False,True):
                x, y=frames_and_label(seed,size,reverse)
                scores=model(torch.tensor(x,dtype=torch.float32).unsqueeze(0))
                guess=int(torch.argmax(scores,dim=-1).item())
                pair.append({"seed":seed,"reverse":reverse,"expected":y,
                             "predicted":guess,"success":int(guess==y)})
            assert pair[0]["expected"] != pair[1]["expected"]
            assert (pair[0]["predicted"] == pair[1]["predicted"]) in (True,False)
            records.append(pair)
    original = [x[0] for x in records]
    counter = [x[1] for x in records]
    return {
        "episodes": episodes,
        "original_success": statistics.mean(x["success"] for x in original),
        "inverted_success": statistics.mean(x["success"] for x in counter),
        "decision_flips": sum(a["predicted"] != b["predicted"]
                              for a,b in records),
        "correct_on_both": sum(a["success"] and b["success"]
                               for a,b in records),
        "pairs": records,
    }


def build_models():
    import torch
    from torch import nn

    class CurrentOnly(nn.Module):
        def __init__(self):
            super().__init__()
            self.net=nn.Sequential(nn.Linear(98,64),nn.ReLU(),
                                   nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
        def forward(self,x):
            return self.net(x[:,-1,:])

    class Temporal(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn=nn.GRU(input_size=98,hidden_size=64,batch_first=True)
            self.head=nn.Sequential(nn.Linear(64,32),nn.ReLU(),nn.Linear(32,2))
        def forward(self,x):
            hidden,_=self.rnn(x)
            return self.head(hidden[:,-1,:])

    return {"current_only":CurrentOnly(), "gru_memory":Temporal()}


def run(train_worlds=1200,epochs=25, out=None):
    import numpy as np
    import torch
    from torch import nn

    torch.set_num_threads(1)
    torch.manual_seed(319)
    rng=np.random.default_rng(319)
    x,y=make_dataset("train",train_worlds,size=7,paired=True)
    counts={str(k):int((y==k).sum()) for k in (0,1)}
    if min(counts.values()) < train_worlds//2:
        raise ValueError("Training labels insufficiently balanced")
    x_tensor=torch.tensor(x,dtype=torch.float32)
    y_tensor=torch.tensor(y,dtype=torch.long)
    models=build_models()
    history={}
    summary={}
    out=Path(out)
    out.mkdir(parents=True,exist_ok=True)
    for name,model in models.items():
        opt=torch.optim.Adam(model.parameters(),lr=0.002)
        losses=[]
        for epoch in range(epochs):
            model.train()
            order=rng.permutation(len(x))
            epoch_loss=[]
            for start in range(0,len(order),128):
                indices=order[start:start+128]
                logits=model(x_tensor[indices])
                loss=nn.functional.cross_entropy(logits,y_tensor[indices])
                opt.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss.append(float(loss.item()))
            losses.append(statistics.mean(epoch_loss))
            if epoch in (0,4,9,epochs-1):
                print("SUPERVISED_TRAIN",name,"epoch",epoch+1,
                      "loss",round(losses[-1],6),flush=True)
        path=out/f"f9c_supervised_{name}.pth"
        torch.save(model.state_dict(),path)
        s7=evaluate(model,7,100)
        s13=evaluate(model,13,30)
        history[name]=losses
        summary[name]={"S7":s7,"S13":s13,
                        "trainable_parameters":sum(p.numel() for p in model.parameters()),
                        "checkpoint_path":path.name,"checkpoint_sha256":file_sha(path)}
        print("SUPERVISED_RESULT",name,"S7",s7["original_success"],
              "flips",s7["decision_flips"],"/100",
              "S13",s13["original_success"],"flips",s13["decision_flips"],"/30",
              flush=True)
    report={
        "experiment":"F9-C supervised observation-only feature memory positive control",
        "training":"supervised crossentropy with labels used only in training",
        "phase":"development_diagnostic",
        "not_original_navigation_benchmark":True,
        "not_PPO_or_RecurrentPPO":True,
        "not_INFINITO":True,
        "features":"98 generic binary key/ball visible partial-image pixel channels, both frames",
        "labels_train_only":"which native MiniGrid Memory exit matches initially visible object",
        "train_seeds_start":1000,
        "training_worlds":train_worlds,"training_pairs":2*train_worlds,
        "epochs":epochs,"training_label_distribution":counts,
        "summary":summary,"train_loss_history":history,
        "heldout_opened":False,
        "source_sha256":file_sha(Path(__file__)),
        "limitations":"Controlled teleport + two actions, supervised oracle labels in TRAINING only; positive control for memory learnability, not demonstration of reinforcement learning or navigating MiniGrid; one split/seed, no holdout.",
    }
    (out/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-worlds",type=int,default=1200)
    p.add_argument("--epochs",type=int,default=25)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not 300<=args.train_worlds<=5000 or not 1<=args.epochs<=100:
        p.error("train worlds 300..5000, epochs 1..100")
    run(args.train_worlds,args.epochs,args.output)


if __name__=="__main__":
    main()
