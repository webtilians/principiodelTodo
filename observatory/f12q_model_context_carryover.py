"""F12-Q: crossed order/history-length/instance reuse diagnosis on frozen O21.

Preregistered at 0336b419971e00df3703ca55898bb90e813f4da7.
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

STUDY="F12-Q"
SOURCE_SHA="b3a52c881af625e9a56166841a334998ec6a366bbb957cb1eabf3c23f099e105"
SYSTEM=frozen.previous.SYSTEM
WORKERS=(0,1)
ORDERS=("AB","BA")
POSITIONS=(4,6)
INSTANCES=("reuse","fresh")
CELLS=tuple((order,position,instance)
            for order in ORDERS for position in POSITIONS
            for instance in INSTANCES)
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
    if index not in range(8):raise ValueError("Cell must be in 0..7")
    return CELLS[index]

def schedule(order,position):
    if order not in ORDERS or position not in POSITIONS:
        raise ValueError("Unregistered factor values")
    a,b=("O0-start","O7-start") if order=="AB" else ("O7-start","O0-start")
    # Hold immediate predecessor O0-disabled fixed across all cells.
    short=(a,b,"O0-disabled")
    return (short if position==4 else
            ("O7-disabled","O21-disabled")+short)

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
    """One cell only; orchestrator MUST run this in its own OS process."""
    if worker not in WORKERS:raise ValueError("Unregistered worker")
    order,position,mode=cell_attributes(index)
    anchors=anchors_by_id()
    prefix=schedule(order,position)
    first=model_factory()
    prefix_rows=[]
    try:
        for j,name in enumerate(prefix):
            prefix_rows.append(do_call(first,anchors[name],"prior",j))
        prefix_metadata=first.metadata()
        target_is_new=(mode=="fresh")
        if target_is_new:
            # Releasing object also releases its llama context. Fresh GGUF same weights.
            del first
            gc.collect()
            target_model=model_factory()
        else:
            target_model=first
        try:
            target_row=do_call(
                target_model,anchors["O21-start"],"target",len(prefix))
            target_metadata=target_model.metadata()
        finally:
            del target_model
            gc.collect()
    finally:
        # Calling del on a name already released is safe only if it still exists.
        if "first" in locals():
            del first
    result={
        "study":STUDY,"worker":worker,"cell_index":index,
        "order":order,"position":position,"instance_mode":mode,
        "prior_schedule":list(prefix),"target_anchor_id":"O21-start",
        "same_instance_used_for_target":not target_is_new,
        "model_instance_init_count":(2 if target_is_new else 1),
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
        "prefix_model":prefix_metadata,"target_model":target_metadata,
        "prior_calls":prefix_rows,"target_call":target_row,
        "transfer_worlds_evaluated":False,"product_integrated":False,
        "status":"CELL_ONLY_NO_DIAGNOSTIC_CONCLUSION",
    }
    return result

def run_worker(worker,out=None,python_executable=None):
    if worker not in WORKERS:raise ValueError("Unregistered worker")
    if out is None:raise ValueError("Worker needs output dir for subprocess isolation")
    out.mkdir(parents=True,exist_ok=True)
    py=python_executable or sys.executable
    order=list(range(8)) if worker==0 else list(reversed(range(8)))
    results=[]
    for index in order:
        cmd=[
            py,"-m","observatory.f12q_model_context_carryover",
            "--trial",str(index),"--worker",str(worker),"--out",str(out)
        ]
        subprocess.run(cmd,check=True)
        file=out/f"f12q_cell_w{worker}_c{index}.json"
        if not file.exists():raise ValueError("Child failed to produce trial audit")
        results.append(json.loads(file.read_text()))
    if len({r["child_pid"] for r in results})!=8:
        raise ValueError("Every cell requires a new OS subprocess")
    model_meta=[r["target_model"] for r in results]
    report={
        "study":STUDY,"worker":worker,"cell_order":order,
        "cells":results,
        "model_integrity":all(model_is_frozen(x) for x in model_meta),
        "status":"WORKER_ONLY_NO_DIAGNOSTIC_CONCLUSION",
    }
    (out/f"f12q_worker{worker}.json").write_text(
        json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
    return report

def validate_cell(cell,source):
    try:
        index=cell["cell_index"]
        order,position,mode=cell_attributes(index)
        prefix=schedule(order,position)
        prev=cell["prior_calls"]
        target=cell["target_call"]
        target_mod=cell["target_model"]
        pre_mod=cell["prefix_model"]
        expected_sources=list(prefix)+["O21-start"]
        observed=prev+[target]
        expected_count=len(prefix)
        if not (
            cell["order"]==order and cell["position"]==position and
            cell["instance_mode"]==mode and
            cell["prior_schedule"]==list(prefix) and
            cell["target_anchor_id"]=="O21-start" and
            cell["same_instance_used_for_target"]==(mode=="reuse") and
            cell["model_instance_init_count"]==(2 if mode=="fresh" else 1) and
            len(prefix)==position-1 and len(prev)==expected_count and
            len(observed)==position and
            cell["transfer_worlds_evaluated"] is False and
            cell["product_integrated"] is False and
            cell["constructor_parameters"]==EXPECTED_MODEL_PARAMS and
            all(model_is_frozen(m) for m in (pre_mod,target_mod)) and
            pre_mod["calls"]==expected_count and
            target_mod["calls"]==(1 if mode=="fresh" else expected_count+1) and
            target["model_calls_before"]==(0 if mode=="fresh" else expected_count) and
            target["model_calls_after"]==(1 if mode=="fresh" else expected_count+1) and
            cell["platform"]["pythonhashseed"]=="0" and
            cell["platform"]["omp_num_threads"]=="4"):
            return False
        for j,(entry,name) in enumerate(zip(observed,expected_sources)):
            known=source[name]
            if not (
                entry["anchor_id"]==name and
                entry["kind"]==("prior" if j<len(prefix) else "target") and
                entry["index"]==j and
                entry["prompt_sha256"]==known["prompt_sha256"] and
                entry["system_sha256"]==known["system_sha256"] and
                entry["raw_sha256"]==sha(entry["raw"]) and
                entry["action"]==frozen.previous.parse_action(entry["raw"]) and
                entry["action"] in frozen.previous.ACTIONS and
                entry["model_calls_after"]==entry["model_calls_before"]+1 and
                entry["cache_before"]["kv_cache_hits_available"] is False and
                entry["cache_after"]["kv_cache_hits_available"] is False and
                entry["cache_before"]["kv_cache_hits"] is None and
                entry["cache_after"]["kv_cache_hits"] is None):
                return False
    except (KeyError,TypeError,ValueError,IndexError):
        return False
    return True

def aggregate(workers,out=None):
    if len(workers)!=2 or {w["worker"] for w in workers}!={0,1}:
        raise ValueError("Two unique workers required")
    source=anchors_by_id()
    cells=[c for w in workers for c in w["cells"]]
    ids=[(c["worker"],c["cell_index"]) for c in cells]
    complete=(
        len(cells)==16 and len(set(ids))==16 and
        set(ids)=={(w,i) for w in WORKERS for i in range(8)} and
        all(len(w["cells"])==8 and
            w["cell_order"]==(list(range(8)) if w["worker"]==0 else
                               list(reversed(range(8))))
            for w in workers)
    )
    pids_valid=all(
        len({c["child_pid"] for c in w["cells"]})==8
        for w in workers
    )
    all_valid=all(validate_cell(c,source) for c in cells)
    Q0=complete and pids_valid and all_valid and sum(
        len(c["prior_calls"])+1 for c in cells)==80
    Q1=all_valid and all(
        c["target_call"]["prompt_sha256"]==SOURCE_SHA for c in cells
    )
    by_cell={i:[] for i in range(8)}
    for c in cells:
        by_cell[c["cell_index"]].append(c["target_call"]["raw"])
    per_cell_equal={i:len(xs)==2 and len(set(xs))==1
                    for i,xs in by_cell.items()}
    Q2=all(per_cell_equal.values())
    cold=[c["target_call"]["raw"] for c in cells
          if c["instance_mode"]=="fresh"]
    Q3=len(cold)==8 and len(set(cold))==1
    effects={"order":[], "position_and_history_length":[], "instance": []}
    if Q2 and Q3:
        for a in range(8):
            for b in range(a+1,8):
                ca=CELLS[a];cb=CELLS[b]
                count=sum(x!=y for x,y in zip(ca,cb))
                if count!=1:continue
                if by_cell[a][0]==by_cell[b][0]:continue
                key=("order" if ca[0]!=cb[0] else
                     "position_and_history_length" if ca[1]!=cb[1]
                     else "instance")
                effects[key].append({
                    "cell_a":a,"cell_b":b,
                    "factors_a":ca,"factors_b":cb,
                    "raw_a":by_cell[a][0],"raw_b":by_cell[b][0],
                })
    gates={
        "Q0_complete_16_independent_cells_80_real_calls":Q0,
        "Q1_pinned_model_and_fresh_instantiation":Q1,
        "Q2_identical_target_raw_between_workers_per_cell":Q2,
        "Q3_fresh_target_is_invariant_across_predecessors":Q3,
        "Q5_cache_proxy_and_no_transfer_or_product":all_valid,
    }
    if not Q0 or not Q1 or not gates["Q5_cache_proxy_and_no_transfer_or_product"]:
        status="INVALID_Q_PROTOCOL"
    elif not Q2 or not Q3:status="INCONCLUSIVE_VARIANCE"
    elif any(effects.values()):status="DIAGNOSTIC_SIGNAL"
    else:status="DIAGNOSTIC_NO_SIGNAL"
    report={
        "study":STUDY,"status":status,"gates":gates,
        "factor_cells":{str(i):{
            "order":CELLS[i][0],"position":CELLS[i][1],"instance_mode":CELLS[i][2],
            "responses":by_cell[i],
        } for i in range(8)},
        "contrasts":effects,
        "unique_target_raws":len({r for xs in by_cell.values() for r in xs}),
        "total_llm_calls":sum(len(c["prior_calls"])+1 for c in cells),
        "reused_instance_tested":True,
        "fresh_instance_tested":True,
        "kv_cache_hit_counts_available":False,
        "cache_measurement_warning":"n_tokens is a state proxy, not a cache-hit/miss counter",
        "transfer_worlds_evaluated":False,
        "product_integrated":False,
        "models":[{
            "worker":w["worker"],
            "cells":[{"cell_index":c["cell_index"],
                      "prefix_model":c["prefix_model"],
                      "target_model":c["target_model"],
                      "platform":c["platform"]} for c in w["cells"]]
        } for w in workers],
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12q_summary.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        (out/"f12q_full_audit.json").write_text(
            json.dumps({**report,"workers":workers},
                       indent=2,sort_keys=True,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,choices=(0,1))
    p.add_argument("--trial",type=int,choices=range(8))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12q_trials"))
    a=p.parse_args()
    if a.trial is not None:
        if a.worker is None:p.error("--trial requires --worker")
        a.out.mkdir(parents=True,exist_ok=True)
        c=run_cell(a.trial,a.worker,FrozenGGUF)
        (a.out/f"f12q_cell_w{a.worker}_c{a.trial}.json").write_text(
            json.dumps(c,indent=2,sort_keys=True,ensure_ascii=False))
        print(json.dumps({"study":STUDY,"worker":a.worker,
                          "cell":a.trial,"raw":c["target_call"]["raw"],
                          "target_action":c["target_call"]["action"]},sort_keys=True))
        return 0
    if a.aggregate_dir is not None:
        files=sorted(a.aggregate_dir.glob("f12q_worker*.json"))
        if len(files)!=2:raise ValueError("Two worker audit files required")
        report=aggregate([json.loads(f.read_text()) for f in files],a.out)
        print(json.dumps({k:report[k] for k in
                          ("study","status","gates","factor_cells","contrasts")},
                         sort_keys=True))
        return 0 if report["status"]!="INVALID_Q_PROTOCOL" else 2
    if a.worker is None:p.error("Specify --worker or --aggregate-dir")
    worker=run_worker(a.worker,a.out)
    print(json.dumps({"study":STUDY,"worker":a.worker,
                      "cells":len(worker["cells"]),"status":worker["status"]},
                     sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
