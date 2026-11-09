"""F12-T: internals of llama.cpp prefix-match context reuse after O21-disabled.

Prospective preregistration frozen at dcc02503ce26aa210d0a4f68d4f9209a7563ef00.
This code must never evaluate F12-P's new transfer worlds or modify Product.
The llama-cpp-python backend is pinned to 0.3.36 and all instrumentation
is scoped to exactly one target model.chat call per cell.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from observatory import f12s_predecessor_instance_factorial as previous
from observatory import f12o_repeatability_gate as frozen
from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256

STUDY="F12-T"
MODEL_VERSION="0.3.36"
SOURCE_SHA=previous.SOURCE_SHA
SYSTEM=frozen.previous.SYSTEM
WARMUP=previous.WARMUP
PREDECESSORS=("O21-disabled","O0-disabled")
INTERVENTIONS=("NATIVE","OBSERVE","BLOCK_PREFIX","RESET","CLEAR_RESET")
CELLS=tuple((p,intervention) for p in PREDECESSORS for intervention in INTERVENTIONS)
WORKERS=(0,1,2)
WORKER_ORDERS=(tuple(range(10)),tuple(reversed(range(10))),
               (3,4,5,6,7,8,9,0,1,2))
TARGET="O21-start"
ALLOWED_ACTIONS=frozen.previous.ACTIONS

def sha(s):
    return previous.prior.sha(s)

def sources():
    records=frozen.build_anchors()
    by_name={a["id"]:a for a in records}
    if len(records)!=6 or len(by_name)!=6:
        raise ValueError("Six exact F12-N anchors required")
    if by_name[TARGET]["prompt_sha256"]!=SOURCE_SHA:
        raise ValueError("O21-start SHA has changed")
    if any(a["prompt_sha256"]!=sha(a["prompt"]) or
           a["system_sha256"]!=sha(SYSTEM) for a in records):
        raise ValueError("Frozen prompt or SYSTEM differs")
    return by_name

def cell_attributes(index):
    if index not in range(10):
        raise ValueError("Only ten preregistered treatments")
    return CELLS[index]

def state_proxy(model):
    gen=model.generator
    # Only observable attributes in llama-cpp-python v0.3.36.
    cache=previous.prior.cache_observables(model)
    cache["requires_eval"]=getattr(gen,"_requires_eval",None)
    return cache

def intervene_and_chat(model,anchor,intervention):
    if intervention not in INTERVENTIONS:
        raise ValueError("Unknown intervention")
    gen=model.generator
    state_pre=state_proxy(model)
    model_meta_before=model.metadata()
    actions=[]
    reset_applied=False
    clear_applied=False
    patch_installed=False
    original_rm=None
    blocked=0
    seen_positive=0
    # Synchronously intervene; do not patch during any source experience call.
    if intervention=="RESET":
        gen.reset()
        reset_applied=True
    elif intervention=="CLEAR_RESET":
        gen._ctx.kv_cache_clear()
        clear_applied=True
        gen.reset()
        reset_applied=True
    if intervention in ("OBSERVE","BLOCK_PREFIX"):
        original_rm=gen._ctx.kv_cache_seq_rm
        def interceptor(seq_id,p0,p1):
            nonlocal blocked,seen_positive
            seq_id,p0,p1=int(seq_id),int(p0),int(p1)
            positive=p0>0 and p1==-1
            if positive:
                seen_positive+=1
            skip=(intervention=="BLOCK_PREFIX" and positive and blocked==0)
            if skip:
                blocked+=1
                result=False
            else:
                result=bool(original_rm(seq_id,p0,p1))
            actions.append({
                "seq_id":seq_id,"p0":p0,"p1":p1,
                "positive_prefix_removal":positive,
                "blocked":skip,
                "returned":result,
                "delegated":not skip,
            })
            return result
        gen._ctx.kv_cache_seq_rm=interceptor
        patch_installed=True
    state_intervention=state_proxy(model)
    started=time.perf_counter()
    try:
        raw=model.chat(SYSTEM,anchor["prompt"])
    finally:
        if patch_installed:
            gen._ctx.kv_cache_seq_rm=original_rm
    elapsed=time.perf_counter()-started
    state_post=state_proxy(model)
    model_meta_after=model.metadata()
    action=frozen.previous.parse_action(raw)
    return {
        "kind":"target","anchor_id":TARGET,"index":13,
        "prompt_sha256":anchor["prompt_sha256"],
        "system_sha256":anchor["system_sha256"],
        "raw":raw,"raw_sha256":sha(raw),"action":action,
        "intervention":intervention,
        "reset_applied":reset_applied,
        "clear_applied":clear_applied,
        "patch_installed":patch_installed,
        "patch_restored":not patch_installed or gen._ctx.kv_cache_seq_rm==original_rm,
        "kv_seq_rm_calls":actions,
        "attempted_positive_requests":seen_positive,
        "blocked_positive_request_count":blocked,
        "state_before":state_pre,
        "state_after_intervention_before_chat":state_intervention,
        "state_after":state_post,
        "model_calls_before":model_meta_before["calls"],
        "model_calls_after":model_meta_after["calls"],
        "prompt_tokens":model_meta_after["prompt_tokens"]-model_meta_before["prompt_tokens"],
        "completion_tokens":model_meta_after["completion_tokens"]-model_meta_before["completion_tokens"],
        "inference_seconds":model_meta_after["inference_seconds"]-model_meta_before["inference_seconds"],
        "elapsed_wall_seconds":elapsed,
    }

def run_cell(index,worker,model_factory=FrozenGGUF):
    if worker not in WORKERS:raise ValueError("Unknown worker")
    predecessor,intervention=cell_attributes(index)
    ref=sources()
    model=model_factory()
    rows=[]
    try:
        for i,name in enumerate(WARMUP):
            rows.append(previous.prior.do_call(model,ref[name],"warmup",i))
        rows.append(previous.prior.do_call(
            model,ref[predecessor],"predecessor",len(rows)))
        model_before=model.metadata()
        target=intervene_and_chat(model,ref[TARGET],intervention)
        model_after=model.metadata()
    finally:
        del model
    return {
        "study":STUDY,"worker":worker,"cell_index":index,
        "predecessor":predecessor,"intervention":intervention,
        "history":list(WARMUP)+[predecessor],
        "target_anchor_id":TARGET,
        "model_before_target":model_before,
        "model_after_target":model_after,
        "prior_calls":rows,"target_call":target,
        "child_pid":os.getpid(),"parent_pid":os.getppid(),
        "platform":{
            "system":platform.system(),"platform":platform.platform(),
            "machine":platform.machine(),"processor":platform.processor(),
            "python":platform.python_version(),
            "pythonhashseed":os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads":os.environ.get("OMP_NUM_THREADS"),
            "cpu_count":os.cpu_count(),
        },
        "constructor_parameters":previous.EXPECTED_MODEL_PARAMS,
        "transfer_worlds_evaluated":False,"product_integrated":False,
        "status":"CELL_ONLY_NO_SCIENTIFIC_CONCLUSION",
    }

def run_worker(worker,out,python_executable=None):
    if worker not in WORKERS:raise ValueError("Unregistered worker")
    out.mkdir(parents=True,exist_ok=True)
    cmd_python=python_executable or sys.executable
    cells=[]
    for cell_index in WORKER_ORDERS[worker]:
        subprocess.run([
            cmd_python,"-m","observatory.f12t_llamacpp_prefix_internals",
            "--worker",str(worker),"--cell",str(cell_index),
            "--out",str(out),
        ],check=True)
        path=out/f"f12t_cell_w{worker}_c{cell_index}.json"
        if not path.exists():
            raise ValueError("Cell child did not produce audit")
        cells.append(json.loads(path.read_text()))
    if len(cells)!=10 or len({c["child_pid"] for c in cells})!=10:
        raise ValueError("Ten isolated OS subprocesses per worker required")
    report={
        "study":STUDY,"worker":worker,"cell_order":list(WORKER_ORDERS[worker]),
        "cells":cells,
        "status":"WORKER_ONLY_NO_SCIENTIFIC_CONCLUSION",
    }
    (out/f"f12t_worker{worker}.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True))
    return report

def model_pinned(m):
    return (
        m.get("real_model") is True
        and m.get("weights_sha256")==FIXED_MODEL_SHA256
        and m.get("revision")=="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
        and m.get("backend_version")==MODEL_VERSION
        and m.get("n_threads")==4
        and m.get("n_ctx")==4096
        and m.get("temperature")==0.0
        and m.get("max_tokens")==96
    )

def validate_cell(cell,ref):
    try:
        predecessor,intervention=cell_attributes(cell["cell_index"])
        prior=cell["prior_calls"]
        target=cell["target_call"]
        before=cell["model_before_target"]
        after=cell["model_after_target"]
        if not (
            cell["study"]==STUDY and cell["worker"] in WORKERS
            and cell["predecessor"]==predecessor
            and cell["intervention"]==intervention
            and cell["history"]==list(WARMUP)+[predecessor]
            and cell["target_anchor_id"]==TARGET
            and cell["constructor_parameters"]==previous.EXPECTED_MODEL_PARAMS
            and cell["platform"]["pythonhashseed"]=="0"
            and cell["platform"]["omp_num_threads"]=="4"
            and cell["transfer_worlds_evaluated"] is False
            and cell["product_integrated"] is False
            and len(prior)==13
            and model_pinned(before)
            and model_pinned(after)
            and before["calls"]==13 and after["calls"]==14
            and target["model_calls_before"]==13
            and target["model_calls_after"]==14
            and target["anchor_id"]==TARGET and target["kind"]=="target"
            and target["index"]==13
            and target["prompt_sha256"]==SOURCE_SHA
            and target["system_sha256"]==ref[TARGET]["system_sha256"]
            and target["raw_sha256"]==sha(target["raw"])
            and target["action"]==frozen.previous.parse_action(target["raw"])
            and target["action"] in ALLOWED_ACTIONS
            and target["patch_restored"] is True
            and target["intervention"]==intervention
            and target["state_before"]["kv_cache_hits_available"] is False
            and target["state_after"]["kv_cache_hits_available"] is False
            and target["state_before"]["kv_cache_hits"] is None):
            return False
        names=list(WARMUP)+[predecessor]
        for i,(row,name) in enumerate(zip(prior,names)):
            a=ref[name]
            if not (
                row["anchor_id"]==name and
                row["kind"]==("warmup" if i<12 else "predecessor")
                and row["index"]==i
                and row["prompt_sha256"]==a["prompt_sha256"]
                and row["system_sha256"]==a["system_sha256"]
                and row["raw_sha256"]==sha(row["raw"])
                and row["action"]==frozen.previous.parse_action(row["raw"])
                and row["action"] in ALLOWED_ACTIONS
                and row["model_calls_after"]==row["model_calls_before"]+1):
                return False
        if intervention in ("RESET","CLEAR_RESET"):
            state=target["state_after_intervention_before_chat"]
            if not (
                target["reset_applied"] and
                state["n_tokens_observed"]==0 and
                state["requires_eval"] is True):
                return False
            if target["clear_applied"]!=(intervention=="CLEAR_RESET"):
                return False
        elif target["reset_applied"] or target["clear_applied"]:
            return False
        patched=intervention in ("OBSERVE","BLOCK_PREFIX")
        if target["patch_installed"]!=patched:
            return False
        if not patched and (
            target["kv_seq_rm_calls"] or
            target["blocked_positive_request_count"]!=0 or
            target["attempted_positive_requests"]!=0):
            return False
        if patched:
            events=target["kv_seq_rm_calls"]
            positive=sum(
                x["positive_prefix_removal"] and x["p0"]>0 and x["p1"]==-1
                for x in events)
            if not events or positive!=target["attempted_positive_requests"]:
                return False
            if any(x["blocked"] == x["delegated"] for x in events):
                return False
            if intervention=="OBSERVE" and (
                target["blocked_positive_request_count"]!=0 or
                any(x["blocked"] for x in events)
                or positive<1
            ):
                return False
            if intervention=="BLOCK_PREFIX" and (
                target["blocked_positive_request_count"]!=1 or positive<1 or
                not any(x["p0"]==0 and x["delegated"] for x in events) or
                not any(x["blocked"] and x["returned"] is False for x in events)
            ):
                return False
    except (ValueError,KeyError,TypeError,IndexError):
        return False
    return True

def aggregate(workers,out=None):
    if len(workers)!=3 or {w["worker"] for w in workers}!=set(WORKERS):
        raise ValueError("Require three independent workers")
    refs=sources()
    cells=[c for w in workers for c in w["cells"]]
    ids=[(c["worker"],c["cell_index"]) for c in cells]
    complete=(
        len(cells)==30 and len(set(ids))==30
        and set(ids)=={(w,i) for w in WORKERS for i in range(10)}
        and all(
            w["cell_order"]==list(WORKER_ORDERS[w["worker"]])
            and [c["cell_index"] for c in w["cells"]]==w["cell_order"]
            and len(w["cells"])==10
            and all(c["worker"]==w["worker"] for c in w["cells"])
            and len({c["child_pid"] for c in w["cells"]})==10
            for w in workers
        )
    )
    good_cells={ (c["worker"],c["cell_index"]):validate_cell(c,refs)
                for c in cells }
    model_valid=all(
        model_pinned(c["model_after_target"]) for c in cells
    )
    count=sum(len(c["prior_calls"])+1 for c in cells)
    prior_count=sum(len(c["prior_calls"]) for c in cells)
    T0=complete and model_valid and count==420 and prior_count==390
    T1=T0 and all(
        c["model_before_target"]["calls"]==13 and
        c["model_after_target"]["calls"]==14 and
        c["target_call"]["prompt_sha256"]==SOURCE_SHA and
        c["constructor_parameters"]==previous.EXPECTED_MODEL_PARAMS
        for c in cells)
    # Separate intervention-specific audit from general completeness, so the
    # status can name the actual failure mode.
    all_integrity=all(good_cells.values())
    T2=all_integrity and all(
        c["target_call"]["attempted_positive_requests"]>=1
        if c["intervention"]=="OBSERVE"
        else c["target_call"]["blocked_positive_request_count"]==1
        if c["intervention"]=="BLOCK_PREFIX"
        else True
        for c in cells
    )
    by_cell={i:[] for i in range(10)}
    for c in cells:
        by_cell[c["cell_index"]].append(c["target_call"]["raw"])
    equal={i:len(v)==3 and len(set(v))==1 for i,v in by_cell.items()}
    T3=all(equal.values()) and all(
        by_cell[CELLS.index((p,"NATIVE"))]==by_cell[CELLS.index((p,"OBSERVE"))]
        for p in PREDECESSORS
    )
    expected={
        ("O21-disabled","NATIVE"):'{"action":"q2"}',
        ("O21-disabled","OBSERVE"):'{"action":"q2"}',
        ("O21-disabled","BLOCK_PREFIX"):'{"action":"q0"}',
        ("O21-disabled","RESET"):'{"action":"q0"}',
        ("O21-disabled","CLEAR_RESET"):'{"action":"q0"}',
        **{("O0-disabled",intervention):'{"action":"q0"}' for intervention in INTERVENTIONS},
    }
    T4=T3 and all(
        len(by_cell[i])==3 and all(raw==expected[CELLS[i]] for raw in by_cell[i])
        for i in range(10)
    )
    T5=all(
        c["product_integrated"] is False and
        c["transfer_worlds_evaluated"] is False for c in cells
    )
    if not T0 or not T1 or not T5:
        status="INVALID_T_PROTOCOL"
    elif not T2:
        status="INVALID_INTERVENTION"
    elif not T3:
        status="REPLICATE_VARIANCE"
    elif T4:
        status="PREFIX_PATH_CAUSAL_SIGNAL"
    elif len(set(raw for values in by_cell.values() for raw in values))==1:
        status="NO_EFFECT_OBSERVED"
    else:
        status="MIXED_MECHANISM"
    report={
        "study":STUDY,"status":status,
        "gates":{
            "T0_30_cells_420_real_model_calls":T0,
            "T1_pinned_model_same_target":T1,
            "T2_untouched_observer_and_effective_prefix_block":T2,
            "T3_3_literal_replicates_and_observer_noop":T3,
            "T4_exact_q2_to_q0_prefix_path_pattern":T4,
            "T5_no_transfer_or_product":T5,
        },
        "cells":{
            str(i):{
                "predecessor":CELLS[i][0],
                "intervention":CELLS[i][1],
                "raw_outputs":by_cell[i],
                "first_positive_reuse_requests":[
                    next((x["p0"] for x in c["target_call"]["kv_seq_rm_calls"]
                          if x["positive_prefix_removal"]),None)
                    for c in cells if c["cell_index"]==i
                ],
                "blocked_positive_request_counts":[
                    c["target_call"]["blocked_positive_request_count"]
                    for c in cells if c["cell_index"]==i
                ],
            } for i in range(10)
        },
        "total_llm_calls":count,
        "prior_calls":prior_count,
        "target_calls":30,
        "kv_cache_hit_counts_available":False,
        "interpretation_limit":"Prefix branch intervention, not proof of cache corruption or personal memory",
        "transfer_worlds_evaluated":False,
        "product_integrated":False,
        "backend_source":"llama-cpp-python v0.3.36 Llama.generate and LlamaContext.kv_cache_seq_rm",
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12t_summary.json").write_text(
            json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True))
        (out/"f12t_full_audit.json").write_text(
            json.dumps({**report,"workers":workers,"cell_integrity":good_cells},
                       indent=2,ensure_ascii=False,sort_keys=True))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,choices=WORKERS)
    p.add_argument("--cell",type=int,choices=range(10))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12t_cells"))
    args=p.parse_args()
    if args.cell is not None:
        if args.worker is None:p.error("--cell requires --worker")
        args.out.mkdir(parents=True,exist_ok=True)
        result=run_cell(args.cell,args.worker,FrozenGGUF)
        (args.out/f"f12t_cell_w{args.worker}_c{args.cell}.json").write_text(
            json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True))
        print(json.dumps({"study":STUDY,"worker":args.worker,"cell":args.cell,
                          "raw":result["target_call"]["raw"],
                          "intervention":result["intervention"]},
                         sort_keys=True))
        return 0
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12t_worker*.json"))
        if len(files)!=3:raise ValueError("Need three F12-T worker JSONs")
        report=aggregate([json.loads(f.read_text()) for f in files],args.out)
        print(json.dumps({k:report[k] for k in ("study","status","gates","cells")},sort_keys=True))
        return 0 if report["status"] not in ("INVALID_T_PROTOCOL","INVALID_INTERVENTION") else 2
    if args.worker is None:p.error("Supply --worker or --aggregate-dir")
    report=run_worker(args.worker,args.out)
    print(json.dumps({"study":STUDY,"worker":args.worker,"cases":len(report["cells"])}))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
