"""F12-R: reproduce F12-P O21=q2 at position 2 with 12-call warm history.

Preregistered and arithmetic-corrected at f5098b6a6b6bb8ce0cdf004e5e687ee3741d2edd before code.
DO NOT import or execute F12-P transfer worlds here.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256
from observatory import f12o_repeatability_gate as frozen

STUDY="F12-R"
SOURCE_SHA="b3a52c881af625e9a56166841a334998ec6a366bbb957cb1eabf3c23f099e105"
SYSTEM=frozen.previous.SYSTEM
WORKERS=(0,1)
HISTORIES=("H0","HP","HR")
POSITIONS=(2,4)
INSTANCES=("reuse","fresh")
CELLS=tuple((history,position,instance)
            for history in HISTORIES for position in POSITIONS
            for instance in INSTANCES)
NORMAL=("O0-start","O0-disabled","O7-start","O7-disabled","O21-start","O21-disabled")
INVERSE=tuple(reversed(NORMAL))
EXPECTED_MODEL_PARAMS={
    "n_threads":4,"n_ctx":4096,"n_batch_requested":256,
    "seed_requested":0,"temperature":0.0,"top_p_requested":1.0,
    "max_tokens":96,"chat_format":"chatml",
}

def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def anchors_by_id():
    records=frozen.build_anchors()
    if len(records)!=6 or len({a["id"] for a in records})!=6:
        raise ValueError("Frozen anchors invalid")
    out={a["id"]:a for a in records}
    if out["O21-start"]["prompt_sha256"]!=SOURCE_SHA:
        raise ValueError("Frozen O21 prompt SHA changed")
    if any(a["system_sha256"]!=sha(SYSTEM) or
           a["prompt_sha256"]!=sha(a["prompt"])
           for a in records):
        raise ValueError("Frozen source checksums invalid")
    return out

def cell_attributes(index):
    if index not in range(12):raise ValueError("Cell must be 0..11")
    return CELLS[index]

def schedule(history_mode,position):
    if history_mode not in HISTORIES or position not in POSITIONS:
        raise ValueError("Unregistered factor values")
    warm={
        "H0":(),
        "HP":NORMAL*2,
        "HR":INVERSE*2,
    }[history_mode]
    # F is the immediately preceding anchor in both block positions.
    prefix=("O21-disabled",) if position==2 else (
        "O0-start","O7-start","O21-disabled")
    return warm,prefix

def cache_observables(model):
    """Observable state proxies; llama-cpp may not expose KV cache-hit counts."""
    generator=getattr(model,"generator",None)
    if generator is None:
        return {
            "n_tokens_observed":None,"cache_object_type":None,
            "n_ctx_observed":None,"kv_cache_hits_available":False,
            "kv_cache_hits":None,
        }
    tokens=getattr(generator,"n_tokens",None)
    if tokens is not None:
        try:tokens=int(tokens)
        except (TypeError,ValueError):tokens=None
    cache=getattr(generator,"cache",None)
    cache_type=type(cache).__name__ if cache is not None else None
    n_ctx=getattr(generator,"n_ctx",None)
    if callable(n_ctx):
        try:n_ctx=int(n_ctx())
        except (TypeError,ValueError,RuntimeError):n_ctx=None
    else:n_ctx=None
    return {
        "n_tokens_observed":tokens,
        "cache_object_type":cache_type,
        "n_ctx_observed":n_ctx,
        # No stable per-query KV hit counter in the pinned FrozenGGUF wrapper.
        "kv_cache_hits_available":False,
        "kv_cache_hits":None,
    }

def model_is_frozen(metadata):
    return (
        metadata.get("real_model") is True
        and metadata.get("weights_sha256")==FIXED_MODEL_SHA256
        and metadata.get("revision")=="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
        and metadata.get("backend_version")=="0.3.36"
        and metadata.get("temperature")==0.0
        and metadata.get("n_threads")==4
        and metadata.get("n_ctx")==4096
        and metadata.get("max_tokens")==96
    )

def do_call(model,anchor,kind,index):
    before=model.metadata()
    state_before=cache_observables(model)
    raw=model.chat(SYSTEM,anchor["prompt"])
    after=model.metadata()
    state_after=cache_observables(model)
    action=frozen.previous.parse_action(raw)
    return {
        "kind":kind,"index":index,"anchor_id":anchor["id"],
        "prompt_sha256":anchor["prompt_sha256"],
        "system_sha256":anchor["system_sha256"],
        "raw":raw,"raw_sha256":sha(raw),"action":action,
        "prompt_tokens":after["prompt_tokens"]-before["prompt_tokens"],
        "completion_tokens":after["completion_tokens"]-before["completion_tokens"],
        "inference_seconds":after["inference_seconds"]-before["inference_seconds"],
        "model_calls_before":before["calls"],"model_calls_after":after["calls"],
        "cache_before":state_before,"cache_after":state_after,
    }

def run_cell(index,worker,model_factory=FrozenGGUF):
    """One cell only, intended to run in a fresh OS child process."""
    if worker not in WORKERS:raise ValueError("Unknown worker")
    history_mode,position,mode=cell_attributes(index)
    known=anchors_by_id()
    warm,prefix=schedule(history_mode,position)
    model=model_factory()
    prior_rows=[]
    try:
        for j,name in enumerate(warm):
            prior_rows.append(do_call(model,known[name],"warmup",j))
        for j,name in enumerate(prefix,start=len(warm)):
            prior_rows.append(do_call(model,known[name],"block_prefix",j))
        prior_metadata=model.metadata()
        new_instance=(mode=="fresh")
        if new_instance:
            # Releases the llama context, but not the OS process/native library.
            del model
            gc.collect()
            target_model=model_factory()
        else:
            target_model=model
        try:
            target=do_call(target_model,known["O21-start"],"target",len(prior_rows))
            target_metadata=target_model.metadata()
        finally:
            del target_model
            gc.collect()
    finally:
        if "model" in locals():
            del model
    return {
        "study":STUDY,"worker":worker,"cell_index":index,
        "history_mode":history_mode,"position":position,
        "instance_mode":mode,
        "warmup_schedule":list(warm),"block_prefix_schedule":list(prefix),
        "prior_schedule":list(warm+prefix),
        "same_instance_used_for_target":not new_instance,
        "model_instance_init_count":2 if new_instance else 1,
        "child_pid":os.getpid(),"parent_pid":os.getppid(),
        "platform":{
            "system":platform.system(),"platform":platform.platform(),
            "machine":platform.machine(),"processor":platform.processor(),
            "python":platform.python_version(),
            "cpu_count":os.cpu_count(),
            "pythonhashseed":os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads":os.environ.get("OMP_NUM_THREADS"),
        },
        "constructor_parameters":EXPECTED_MODEL_PARAMS,
        "prefix_model":prior_metadata,"target_model":target_metadata,
        "prior_calls":prior_rows,"target_call":target,
        "transfer_worlds_evaluated":False,"product_integrated":False,
        "status":"CELL_ONLY_NO_DIAGNOSTIC_CONCLUSION",
    }

def run_worker(worker,out=None,python_executable=None):
    if worker not in WORKERS:raise ValueError("Unknown worker")
    if out is None:raise ValueError("Output folder required for child-process provenance")
    out.mkdir(parents=True,exist_ok=True)
    py=python_executable or sys.executable
    order=list(range(12)) if worker==0 else list(reversed(range(12)))
    results=[]
    for index in order:
        subprocess.run([
            py,"-m","observatory.f12r_o21_position2_reproduction",
            "--trial",str(index),"--worker",str(worker),"--out",str(out)
        ],check=True)
        f=out/f"f12r_cell_w{worker}_c{index}.json"
        if not f.exists():raise ValueError("Child process artifact missing")
        results.append(json.loads(f.read_text()))
    if len(results)!=12 or len({r["child_pid"] for r in results})!=12:
        raise ValueError("Must use a separate OS process for every cell")
    report={
        "study":STUDY,"worker":worker,"cell_order":order,
        "cells":results,
        "model_integrity":all(model_is_frozen(c["target_model"]) for c in results),
        "status":"WORKER_ONLY_NO_DIAGNOSTIC_CONCLUSION",
    }
    (out/f"f12r_worker{worker}.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
    return report

def validate_cell(c,source):
    try:
        index=c["cell_index"]
        history_mode,position,mode=cell_attributes(index)
        warm,prefix=schedule(history_mode,position)
        prior=c["prior_calls"]
        target=c["target_call"]
        metadata=c["target_model"]
        expected_names=list(warm+prefix)+["O21-start"]
        if not (
            c["history_mode"]==history_mode and
            c["position"]==position and c["instance_mode"]==mode and
            c["warmup_schedule"]==list(warm) and
            c["block_prefix_schedule"]==list(prefix) and
            c["prior_schedule"]==list(warm+prefix) and
            c["same_instance_used_for_target"]==(mode=="reuse") and
            c["model_instance_init_count"]==(2 if mode=="fresh" else 1) and
            len(prior)==len(warm)+len(prefix) and len(prefix)==position-1 and
            c["transfer_worlds_evaluated"] is False and
            c["product_integrated"] is False and
            c["constructor_parameters"]==EXPECTED_MODEL_PARAMS and
            all(model_is_frozen(x) for x in (c["prefix_model"],metadata)) and
            c["prefix_model"]["calls"]==len(prior) and
            metadata["calls"]==(1 if mode=="fresh" else len(prior)+1) and
            target["model_calls_before"]==(0 if mode=="fresh" else len(prior)) and
            target["model_calls_after"]==(1 if mode=="fresh" else len(prior)+1) and
            c["platform"]["pythonhashseed"]=="0" and
            c["platform"]["omp_num_threads"]=="4"):
            return False
        for j,(e,name) in enumerate(zip(prior+[target],expected_names)):
            ref=source[name]
            expected_kind=("warmup" if j<len(warm) else
                           "block_prefix" if j<len(prior) else "target")
            if not (
                e["anchor_id"]==name and e["kind"]==expected_kind and e["index"]==j and
                e["prompt_sha256"]==ref["prompt_sha256"] and
                e["system_sha256"]==ref["system_sha256"] and
                e["raw_sha256"]==sha(e["raw"]) and
                e["action"]==frozen.previous.parse_action(e["raw"]) and
                e["action"] in frozen.previous.ACTIONS and
                e["model_calls_after"]==e["model_calls_before"]+1 and
                e["cache_before"]["kv_cache_hits_available"] is False and
                e["cache_after"]["kv_cache_hits_available"] is False and
                e["cache_before"]["kv_cache_hits"] is None and
                e["cache_after"]["kv_cache_hits"] is None):
                return False
    except (KeyError,TypeError,ValueError,IndexError):
        return False
    return True

def aggregate(workers,out=None):
    if len(workers)!=2 or {w["worker"] for w in workers}!={0,1}:
        raise ValueError("Two independent worker reports required")
    ref=anchors_by_id()
    cells=[c for w in workers for c in w["cells"]]
    ids=[(c["worker"],c["cell_index"]) for c in cells]
    expected={(worker,i) for worker in WORKERS for i in range(12)}
    complete=(
        len(cells)==24 and len(set(ids))==24 and set(ids)==expected and
        all(len(w["cells"])==12 and
            w["cell_order"]==(list(range(12)) if w["worker"]==0 else
                               list(reversed(range(12))))
            for w in workers)
    )
    pids=all(len({c["child_pid"] for c in w["cells"]})==12 for w in workers)
    valid=all(validate_cell(c,ref) for c in cells)
    count=sum(len(c["prior_calls"])+1 for c in cells)
    prior_count=sum(len(c["prior_calls"]) for c in cells)
    R0=complete and pids and valid and count==264 and prior_count==240
    R1=valid and all(
        c["target_call"]["prompt_sha256"]==SOURCE_SHA
        for c in cells)
    by_cell={i:[] for i in range(12)}
    for c in cells:
        by_cell[c["cell_index"]].append(c["target_call"]["raw"])
    matched={i:len(values)==2 and len(set(values))==1 for i,values in by_cell.items()}
    R2=all(matched.values())
    main_index=CELLS.index(("HP",2,"reuse"))
    R3=matched[main_index] and by_cell[main_index][0]=='{"action":"q2"}'
    contrasts={
        "warm_history":[],
        "history_order":[],
        "position_and_extra_history":[],
        "instance":[],
    }
    if R2:
        for i in range(12):
            for j in range(i+1,12):
                a,b=CELLS[i],CELLS[j]
                differences=sum(x!=y for x,y in zip(a,b))
                if differences!=1 or by_cell[i][0]==by_cell[j][0]:continue
                if a[0]!=b[0]:
                    key="history_order" if set((a[0],b[0]))=={"HP","HR"} else "warm_history"
                elif a[1]!=b[1]:key="position_and_extra_history"
                else:key="instance"
                contrasts[key].append({
                    "cell_a":i,"cell_b":j,"factors_a":a,"factors_b":b,
                    "raw_a":by_cell[i][0],"raw_b":by_cell[j][0],
                })
    gates={
        "R0_24_independent_cells_264_real_inferences":R0,
        "R1_frozen_model_and_target_prompt":R1,
        "R2_replicates_literal_per_cell":R2,
        "R3_original_12call_then_P2_q2":R3,
        "R5_annotated_cache_no_transfer_or_product":valid,
    }
    if not (R0 and R1 and valid):status="INVALID_R_PROTOCOL"
    elif not R2:status="INCONCLUSIVE_REPLICATE_VARIANCE"
    elif R3:status="REPRODUCED_WITH_CONTROLS"
    else:status="NO_REPRODUCTION"
    report={
        "study":STUDY,"status":status,"gates":gates,
        "cell_responses":{str(i):{
            "history":CELLS[i][0],"target_position":CELLS[i][1],
            "instance":CELLS[i][2],"responses":by_cell[i],
        } for i in range(12)},
        "contrasts":contrasts,
        "original_sequence_cell":main_index,
        "unique_target_raws":len({raw for raws in by_cell.values() for raw in raws}),
        "total_llm_calls":count,"prior_calls":prior_count,"target_calls":24,
        "kv_cache_hit_counts_available":False,
        "cache_measurement_warning":"n_tokens is only a state proxy; not hit/miss of KV cache",
        "transfer_worlds_evaluated":False,"product_integrated":False,
        "interpretation_limit":"P2 vs P4 also alters number and content of prior prompts; no autonomous learning claims",
        "models":[{
            "worker":w["worker"],
            "cells":[{"cell_index":c["cell_index"],
                      "prior_model":c["prefix_model"],
                      "target_model":c["target_model"],
                      "platform":c["platform"]} for c in w["cells"]]
        } for w in workers],
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12r_summary.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        (out/"f12r_full_audit.json").write_text(
            json.dumps({**report,"workers":workers},
                       indent=2,sort_keys=True,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,choices=(0,1))
    p.add_argument("--trial",type=int,choices=range(12))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12r_trials"))
    a=p.parse_args()
    if a.trial is not None:
        if a.worker is None:p.error("--trial requires --worker")
        a.out.mkdir(parents=True,exist_ok=True)
        cell=run_cell(a.trial,a.worker,FrozenGGUF)
        (a.out/f"f12r_cell_w{a.worker}_c{a.trial}.json").write_text(
            json.dumps(cell,indent=2,sort_keys=True,ensure_ascii=False))
        print(json.dumps({"study":STUDY,"worker":a.worker,"cell":a.trial,
                          "raw":cell["target_call"]["raw"]},sort_keys=True))
        return 0
    if a.aggregate_dir is not None:
        files=sorted(a.aggregate_dir.glob("f12r_worker*.json"))
        if len(files)!=2:raise ValueError("Need both worker audits")
        report=aggregate([json.loads(f.read_text()) for f in files],a.out)
        print(json.dumps({k:report[k] for k in
                          ("study","status","gates","cell_responses","contrasts")},
                         sort_keys=True))
        return 0 if report["status"]!="INVALID_R_PROTOCOL" else 2
    if a.worker is None:p.error("Specify --worker or --aggregate-dir")
    report=run_worker(a.worker,a.out)
    print(json.dumps({"study":STUDY,"worker":a.worker,
                      "cells":len(report["cells"]),"status":report["status"]},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
