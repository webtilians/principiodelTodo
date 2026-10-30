"""Re-evaluate original F9-H checkpoints; no training and no heldout."""
import argparse
import json
from pathlib import Path
from sb3_contrib import RecurrentPPO
from observatory import f9i_selective_navigation as nav
from observatory.f9_external_minigrid import file_sha
import torch

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,choices=range(4),required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True,exist_ok=True)
    nav.EVENT_CAPACITY=64
    model=RecurrentPPO.load(args.checkpoint,device="cpu")
    sha=file_sha(args.checkpoint)
    result={"checkpoint_sha256":sha,"source":"F9-H run 40925227783",
            "seed_index":args.index,"heldout_opened":False,"metrics":{}}
    for size in (7,13):
        for control in ("normal","corrupt_initial"):
            result["metrics"][f"S{size}/{control}"]=nav.evaluate(
                model,"fifo",args.index,size,10,control,args.output/"traces.jsonl")
    assert sha==file_sha(args.checkpoint)
    (args.output/"diagnosis.json").write_text(json.dumps(result,indent=2))
    print("F9I_F9H_DIAGNOSIS",json.dumps(result),flush=True)

if __name__=="__main__":
    main()
