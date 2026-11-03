"""F10-L STAGE TWO (only when independent G0 authorized it).

Loads SHA-frozen primitive policies; no optimizer, no training, no other
model update. Only at this point creates never-seen OOD composition worlds.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch

from observatory.f10l_worlds import select_data
from observatory.f10l_train import XorActor,SelectActor,sha,sel_tensors

def run(rep,stage1):
    m=json.loads((stage1/"train.json").read_text())
    assert m["status"]=="FROZEN_BEFORE_DEV_AND_ANY_OOD"
    assert m["train_composition_exposure"]==0
    assert m["confirm_23300000_unopened"]
    models={}
    for arm in ("xor_onpolicy","xor_two_actions",
                "sel_onpolicy","sel_uniform_credit","sel_uniform_pair"):
        cp=stage1/(arm+".pt")
        assert sha(cp)==m["arms"][arm]["sha256"]
        actor=XorActor() if arm.startswith("xor") else SelectActor()
        actor.load_state_dict(torch.load(cp,weights_only=True,map_location="cpu"))
        actor.eval()
        models[arm]=actor
    cases=select_data("ood",rep)
    assert len(cases)==512
    f,mask,w=sel_tensors(cases)
    contents=torch.from_numpy(np.stack([c["contents"] for c in cases]))
    flip=torch.ones(len(cases),dtype=torch.float32)
    with torch.no_grad():
        s=models["sel_uniform_pair"](f,mask).argmax(-1)
        s_a=models["sel_onpolicy"](f,mask).argmax(-1)
        idx=torch.arange(len(cases))
        value=contents[idx,s]
        value_a=contents[idx,s_a]
        oracle_value=contents[idx,w]
        normal_input=torch.stack((value,2*flip-1),dim=1)
        normal=(models["xor_two_actions"](normal_input)>=0).long()
        alt_xor=(models["xor_onpolicy"](normal_input)>=0).long()
        alt_selector=(models["xor_two_actions"](
            torch.stack((value_a,2*flip-1),dim=1))>=0).long()
        oracle_slot=(models["xor_two_actions"](
            torch.stack((oracle_value,2*flip-1),dim=1))>=0).long()
        no_flip=(models["xor_two_actions"](
            torch.stack((value,-torch.ones_like(flip)),dim=1))>=0).long()
        shuffled=contents.clone()
        for i in range(len(cases)):
            valid=torch.where(mask[i]>.5)[0]
            shuffled[i,valid]=contents[i,valid.roll(1)]
        shuffled_value=shuffled[idx,s]
        shuffle=(models["xor_two_actions"](
            torch.stack((shuffled_value,2*flip-1),dim=1))>=0).long()
        # Deterministic symbolic interpretation of the selected bit.
        symbolic_selected=((value>0).long()^1)
        # Only evaluator/diagnostic path may see the winner.
        symbolic_oracle=((oracle_value>0).long()^1)
    rows=[]
    for i,c in enumerate(cases):
        assert int(symbolic_oracle[i])==(c["status"]^1)
        rows.append({
            "replica":rep,"hash":c["hash"],"seed":c["seed"],
            "base_world":c["base_world"],"variant":c["variant"],
            "source_mode":c["source_mode"],"n":c["n"],
            "signed_count":c["signed_count"],
            "winner":c["winner"],"selected":int(s[i]),
            "selected_onpolicy":int(s_a[i]),
            "truth":c["status"]^1,
            "predictions":{
                "composed":int(normal[i]),"shuffle":int(shuffle[i]),
                "no_flip":int(no_flip[i]),"oracle_slot":int(oracle_slot[i]),
                "xor_onpolicy_selector_paired":int(alt_xor[i]),
                "xor_paired_selector_onpolicy":int(alt_selector[i]),
                "symbolic_xor_learned_selector":int(symbolic_selected[i]),
                "symbolic_oracle":int(symbolic_oracle[i]),
            },
        })
    for arm in models:
        assert sha(stage1/(arm+".pt"))==m["arms"][arm]["sha256"]
    output={
        "experiment":"INFINITO F10-L zero-shot composition, STAGE TWO",
        "replica":rep,"status":"EVALUATED_ONLY_AFTER_G0",
        "no_optimizer_or_training_in_stage2":True,
        "confirm_23300000_unopened":True,
        "checkpoint_sha":{a:m["arms"][a]["sha256"] for a in models},
        "rows":rows,
    }
    print("F10L_FROZEN_OOD",rep,"cases",len(rows),
          "composition_accuracy",sum(r["predictions"]["composed"]==r["truth"]
                                     for r in rows)/len(rows),flush=True)
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--replica",type=int,required=True,choices=range(4))
    p.add_argument("--stage1",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    result=run(args.replica,args.stage1)
    args.out.write_text(json.dumps(result,separators=(",",":")))
