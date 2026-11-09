"""F12-P-H: interleave frozen prompts across workers to detect carryover/order effects.
Preregistered at e7fb7ab1a87a before implementation.
"""
from __future__ import annotations
import argparse
import json
import platform
from pathlib import Path
from observatory import f12o_repeatability_gate as frozen
from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256

STUDY="F12-P-H"
ORDERS=((0,1,2,3,4,5),(5,4,3,2,1,0),(2,5,0,3,1,4))
REPEATS=2
WORKERS=2

def anchors():
    return frozen.build_anchors()

def run_worker(model,worker,out=None):
    if worker not in range(WORKERS):raise ValueError("Unregistered worker")
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    source=anchors()
    rows=[]
    for order_id,order in enumerate(ORDERS):
        for cycle in range(REPEATS):
            for position,anchor_index in enumerate(order):
                a=source[anchor_index]
                before=model.metadata()
                raw=model.chat(frozen.previous.SYSTEM,a["prompt"])
                after=model.metadata()
                rows.append({
                    "worker":worker,"order_id":order_id,"cycle":cycle,
                    "position":position,"anchor_index":anchor_index,"anchor_id":a["id"],
                    "prompt_sha256":a["prompt_sha256"],
                    "system_sha256":a["system_sha256"],
                    "raw":raw,"raw_sha256":frozen.sha(raw),
                    "action":frozen.previous.parse_action(raw),
                    "prompt_tokens":after["prompt_tokens"]-before["prompt_tokens"],
                    "output_tokens":after["completion_tokens"]-before["completion_tokens"],
                    "inference_seconds":after["inference_seconds"]-before["inference_seconds"],
                })
    result={
        "study":STUDY,"worker":worker,"model":model.metadata(),
        "anchors":source,"rows":rows,
        "machine":{"platform":platform.platform(),"machine":platform.machine(),
                   "processor":platform.processor(),"python":platform.python_version()},
        "status":"WORKER_ONLY_NO_SCIENTIFIC_RESULT",
    }
    if out is not None:
        (out/f"f12p_history_worker{worker}.json").write_text(
            json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True))
    return result

def aggregate(workers,out=None):
    if len(workers)!=2 or {w["worker"] for w in workers}!={0,1}:
        raise ValueError("Exactly two independent history workers required")
    source=anchors()
    source_map={a["id"]:a for a in source}
    rows=[r for w in workers for r in w["rows"]]
    expected={(worker,oid,cyc,pos)
              for worker in range(2) for oid in range(3)
              for cyc in range(2) for pos in range(6)}
    actual=[(r["worker"],r["order_id"],r["cycle"],r["position"]) for r in rows]
    complete=len(rows)==72 and len(set(actual))==72 and set(actual)==expected
    model_ok=all(
        w["model"].get("real_model") is True
        and w["model"].get("weights_sha256")==FIXED_MODEL_SHA256
        and w["model"].get("revision")=="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
        and w["model"].get("backend_version")=="0.3.36"
        and w["model"].get("temperature")==0.0
        and w["model"].get("n_threads")==4
        for w in workers)
    anchor_ok=all(
        [(a["id"],a["prompt"],a["prompt_sha256"],a["system_sha256"]) for a in w["anchors"]]
        ==[(a["id"],a["prompt"],a["prompt_sha256"],a["system_sha256"]) for a in source]
        for w in workers)
    raw_by_anchor={a["id"]:[] for a in source}
    valid=True
    for r in rows:
        if (r["anchor_id"] not in source_map or
            r["order_id"] not in range(3) or
            r["position"] not in range(6) or
            r["anchor_index"]!=ORDERS[r["order_id"]][r["position"]] or
            r["anchor_id"]!=source[r["anchor_index"]]["id"] or
            r["prompt_sha256"]!=source_map[r["anchor_id"]]["prompt_sha256"] or
            r["system_sha256"]!=source_map[r["anchor_id"]]["system_sha256"] or
            r["raw_sha256"]!=frozen.sha(r["raw"]) or
            r["action"] not in frozen.previous.ACTIONS):
            valid=False
        if r["anchor_id"] in raw_by_anchor:
            raw_by_anchor[r["anchor_id"]].append(r["raw"])
    equal={name:len(values)==12 and len(set(values))==1
           for name,values in raw_by_anchor.items()}
    gates={
        "H0_real_72_interleaved_frozen_prompts":complete and model_ok and anchor_ok and valid,
        "H1_literal_output_invariant_to_order_and_worker":all(equal.values()),
    }
    if not model_ok and not any(w["model"].get("real_model") for w in workers):
        status="FAKE_TEST_ONLY"
    elif not gates["H0_real_72_interleaved_frozen_prompts"]:
        status="INVALID_HISTORY"
    elif not gates["H1_literal_output_invariant_to_order_and_worker"]:
        status="HISTORY_VARIANCE_DETECTED"
    else:status="HISTORY_PASS_FOR_ANCHORS"
    result={
        "study":STUDY,"status":status,"gates":gates,
        "per_anchor_equal":equal,
        "unique_raw_outputs":{k:len(set(v)) for k,v in raw_by_anchor.items()},
        "by_anchor":{name:[
            {k:r[k] for k in ("worker","order_id","cycle","position","raw","raw_sha256","action")}
            for r in rows if r["anchor_id"]==name]
            for name in raw_by_anchor},
        "models":[w["model"] for w in workers],
        "machines":[w["machine"] for w in workers],
        "transfer_worlds_evaluated":False,
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12p_history_gate.json").write_text(
            json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,choices=(0,1))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12p_history"))
    args=p.parse_args()
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12p_history_worker*.json"))
        if len(files)!=2:raise ValueError("Missing history worker")
        result=aggregate([json.loads(f.read_text()) for f in files],args.out)
        print(json.dumps({k:result[k] for k in ("study","status","gates","unique_raw_outputs")},
                         sort_keys=True))
        return 0 if result["status"]=="HISTORY_PASS_FOR_ANCHORS" else 2
    if args.worker is None:p.error("Specify --worker or --aggregate-dir")
    report=run_worker(FrozenGGUF(),args.worker,args.out)
    print(json.dumps({"study":STUDY,"worker":args.worker,
                      "inferences":len(report["rows"]),"model":report["model"]},sort_keys=True))
    return 0

if __name__=="__main__":raise SystemExit(main())
