"""F12-O-R: frozen Qwen 7B raw-answer repeatability before new-world trial.

Preregistered in b18d66e25f2c before any F12-O inference.
The six anchors reuse F12-N worlds ONLY as diagnostics, never as confirmation.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
from pathlib import Path

from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256
from observatory import f12n_public_error_cursor as previous

STUDY="F12-O-R"
ANCHOR_WORLDS=(0,7,21)
REPEATS=5
WORKERS=2
O21_START_SHA="b3a52c881af625e9a56166841a334998ec6a366bbb957cb1eabf3c23f099e105"
GOAL="Use the hidden bit, invert it, and verify success."

def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()

def build_anchors():
    worlds=previous.frozen_preflight()
    anchors=[]
    for wid in ANCHOR_WORLDS:
        w=worlds[wid]
        plan=w["plan"]
        prefix=[]
        world=previous.World(wid,*previous.BIT_PAIRS[wid],"changed")
        for action in plan["plan"][:3]:
            prefix.append({"origin":"controlled_prefix","action":action,
                           "response":world.execute(action)})
        if prefix[-1]["response"]!={"error":"DISABLED_TOOL"}:
            raise ValueError("Not a genuine public disabled error")
        for condition,events in (("start",[]),("disabled",prefix)):
            prompt=previous.prompt_for(GOAL,w["context"],plan,events,"F0")
            anchors.append({
                "id":f"O{wid}-{condition}",
                "prompt":prompt,
                "prompt_sha256":sha(prompt),
                "system_sha256":sha(previous.SYSTEM),
            })
    match=next(a for a in anchors if a["id"]=="O21-start")
    if match["prompt_sha256"]!=O21_START_SHA:
        raise ValueError("F12-N O21 start source-SHA mismatch; no inference")
    if len(anchors)!=6 or len({a["prompt_sha256"] for a in anchors})!=6:
        raise ValueError("Invalid 6-anchor preregistration")
    return anchors

def run_worker(model,worker,out=None):
    if worker not in range(WORKERS):raise ValueError("Invalid worker number")
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    anchors=build_anchors()
    rows=[]
    for a in anchors:
        for rep in range(REPEATS):
            before=model.metadata()
            raw=model.chat(previous.SYSTEM,a["prompt"])
            after=model.metadata()
            parsed=previous.parse_action(raw)
            rows.append({
                "anchor_id":a["id"],"worker":worker,"repetition":rep,
                "prompt_sha256":a["prompt_sha256"],
                "system_sha256":a["system_sha256"],
                "raw":raw,"raw_sha256":sha(raw),"parsed_action":parsed,
                "prompt_tokens":after.get("prompt_tokens",0)-before.get("prompt_tokens",0),
                "completion_tokens":after.get("completion_tokens",0)-before.get("completion_tokens",0),
                "inference_seconds":after.get("inference_seconds",0)-before.get("inference_seconds",0),
            })
    result={
        "study":STUDY,"worker":worker,"model":model.metadata(),
        "machine":{"platform":platform.platform(),"machine":platform.machine(),
                   "processor":platform.processor(),"python":platform.python_version()},
        "anchors":anchors,"replays":rows,
        "status":"WORKER_ONLY_NOT_SCIENTIFIC_RESULT",
    }
    if out is not None:
        (out/f"f12o_replay_worker{worker}.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))
    return result

def aggregate(workers,out=None):
    if len(workers)!=WORKERS or {x["worker"] for x in workers}!={0,1}:
        raise ValueError("Requires exactly two unique replay workers")
    frozen=build_anchors()
    expected={(a["id"],worker,rep) for a in frozen
              for worker in range(WORKERS) for rep in range(REPEATS)}
    rows=[r for w in workers for r in w["replays"]]
    identities=[(r["anchor_id"],r["worker"],r["repetition"]) for r in rows]
    valid_rows=(len(rows)==60 and len(set(identities))==60 and set(identities)==expected)
    valid_model=all(
        w["model"].get("real_model") is True
        and w["model"].get("weights_sha256")==FIXED_MODEL_SHA256
        and w["model"].get("revision")=="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
        and w["model"].get("backend_version")=="0.3.36"
        and w["model"].get("temperature")==0.0
        and w["model"].get("n_threads")==4
        for w in workers)
    anchor_match=all(
        [(a["id"],a["prompt"],a["prompt_sha256"],a["system_sha256"]) for a in w["anchors"]]
        ==[(a["id"],a["prompt"],a["prompt_sha256"],a["system_sha256"]) for a in frozen]
        for w in workers)
    by_id={a["id"]:[] for a in frozen}
    valid_responses=True
    for r in rows:
        if r["anchor_id"] not in by_id:valid_responses=False;continue
        a=next(a for a in frozen if a["id"]==r["anchor_id"])
        if (not r["raw"] or r["raw_sha256"]!=sha(r["raw"])
            or r["parsed_action"] not in previous.ACTIONS
            or r["prompt_sha256"]!=a["prompt_sha256"]
            or r["system_sha256"]!=a["system_sha256"]):
            valid_responses=False
        by_id[r["anchor_id"]].append(r["raw"])
    literal={id_:len(set(raw))==1 and len(raw)==10 for id_,raw in by_id.items()}
    gates={
        "R0_exact_60_real_inferences_locked_sources":(
            valid_rows and valid_model and anchor_match and valid_responses),
        "R1_literal_raw_equality_across_processes":all(literal.values()),
    }
    status=("REPLAY_PASS" if all(gates.values()) else
            "FAKE_TEST_ONLY" if not valid_model and
                not any(w["model"].get("real_model") for w in workers)
            else "REPLAY_VARIANCE_DETECTED" if gates["R0_exact_60_real_inferences_locked_sources"]
            else "INVALID_REPLAY")
    report={
        "study":STUDY,"status":status,"gates":gates,
        "per_anchor_literal_equal":literal,
        "unique_raw_outputs":{k:len(set(v)) for k,v in by_id.items()},
        "outputs_by_anchor":{
            k:[{"worker":r["worker"],"repetition":r["repetition"],
                "raw":r["raw"],"raw_sha256":r["raw_sha256"]}
               for r in rows if r["anchor_id"]==k] for k in by_id},
        "machines":[w["machine"] for w in workers],
        "models":[w["model"] for w in workers],
        "factor_worlds_evaluated":False,
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12o_replay_gate.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,choices=(0,1))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12o_replay"))
    args=p.parse_args()
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12o_replay_worker*.json"))
        if len(files)!=2:raise ValueError("Missing replay worker artifact")
        result=aggregate([json.loads(x.read_text()) for x in files],args.out)
        print(json.dumps({k:result[k] for k in
                          ("study","status","gates","per_anchor_literal_equal","unique_raw_outputs")},
                         sort_keys=True))
        return 0 if result["status"]=="REPLAY_PASS" else 2
    if args.worker is None:p.error("Specify --worker or --aggregate-dir")
    result=run_worker(FrozenGGUF(),args.worker,args.out)
    print(json.dumps({"study":STUDY,"worker":args.worker,
                      "model":result["model"],"replays":len(result["replays"])},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
