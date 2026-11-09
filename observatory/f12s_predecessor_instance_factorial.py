"""F12-S: frozen 2x2 predecessor identity x model-instance state.

Pre-registered at 65e4a386eb51d7d13147222795d32e315d8288b3.
Uses ONLY F12-N frozen diagnostic prompts via F12-R; no F12-P transfer worlds.
"""
from __future__ import annotations
import argparse
import gc
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from observatory import f12r_o21_position2_reproduction as prior
from observatory.f12f_evidence_validator import FrozenGGUF

STUDY="F12-S"
PREDECESSORS=("O21-disabled","O0-disabled")
INSTANCES=("reuse","fresh")
CELLS=tuple((predecessor,instance)
            for predecessor in PREDECESSORS for instance in INSTANCES)
WORKER_ORDERS=(
    (0,1,2,3),
    (1,3,0,2),
    (2,0,3,1),
    (3,2,1,0),
)
WORKERS=tuple(range(4))
WARMUP=prior.NORMAL*2
TARGET="O21-start"
SOURCE_SHA=prior.SOURCE_SHA
EXPECTED_MODEL_PARAMS=prior.EXPECTED_MODEL_PARAMS

def sources():
    anchors=prior.anchors_by_id()
    if len(anchors)!=6 or anchors[TARGET]["prompt_sha256"]!=SOURCE_SHA:
        raise ValueError("Frozen F12-N sources modified")
    return anchors

def cell_attributes(index):
    if index not in range(len(CELLS)):
        raise ValueError("F12-S cell index outside preregistration")
    return CELLS[index]

def run_cell(index,worker,model_factory=FrozenGGUF):
    if worker not in WORKERS:raise ValueError("Invalid independent worker")
    predecessor,instance=cell_attributes(index)
    anchors=sources()
    model=model_factory()
    preceding=[]
    try:
        for k,name in enumerate(WARMUP):
            preceding.append(prior.do_call(model,anchors[name],"warmup",k))
        preceding.append(prior.do_call(
            model,anchors[predecessor],"predecessor",len(preceding)))
        prefix_metadata=model.metadata()
        new_instance=instance=="fresh"
        if new_instance:
            del model
            gc.collect()
            target_model=model_factory()
        else:
            target_model=model
        try:
            target=prior.do_call(
                target_model,anchors[TARGET],"target",len(preceding))
            target_metadata=target_model.metadata()
        finally:
            del target_model
            gc.collect()
    finally:
        if "model" in locals():
            del model
    return {
        "study":STUDY,"worker":worker,"cell_index":index,
        "predecessor":predecessor,"instance":instance,
        "history":list(WARMUP)+( [predecessor] ),
        "target_anchor_id":TARGET,
        "same_instance_for_target":not new_instance,
        "model_instance_init_count":(2 if new_instance else 1),
        "child_pid":os.getpid(),"parent_pid":os.getppid(),
        "platform":{
            "system":platform.system(),
            "platform":platform.platform(),
            "machine":platform.machine(),
            "processor":platform.processor(),
            "python":platform.python_version(),
            "cpu_count":os.cpu_count(),
            "pythonhashseed":os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads":os.environ.get("OMP_NUM_THREADS"),
        },
        "constructor_parameters":EXPECTED_MODEL_PARAMS,
        "prefix_model":prefix_metadata,"target_model":target_metadata,
        "prior_calls":preceding,"target_call":target,
        "transfer_worlds_evaluated":False,
        "product_integrated":False,
        "status":"CELL_ONLY_NO_SCIENTIFIC_CONCLUSION",
    }

def run_worker(worker,out,python_executable=None):
    if worker not in WORKERS:raise ValueError("Unregistered worker")
    out.mkdir(parents=True,exist_ok=True)
    python_executable=python_executable or sys.executable
    cells=[]
    for index in WORKER_ORDERS[worker]:
        subprocess.run([
            python_executable,
            "-m","observatory.f12s_predecessor_instance_factorial",
            "--worker",str(worker),"--cell",str(index),"--out",str(out),
        ],check=True)
        path=out/f"f12s_cell_w{worker}_c{index}.json"
        if not path.exists():
            raise ValueError("Missing subprocess cell audit")
        cells.append(json.loads(path.read_text()))
    if len(cells)!=4 or len({c["child_pid"] for c in cells})!=4:
        raise ValueError("Four independent OS subprocesses required per worker")
    result={
        "study":STUDY,"worker":worker,
        "cell_order":list(WORKER_ORDERS[worker]),
        "cells":cells,
        "status":"WORKER_ONLY_NO_SCIENTIFIC_CONCLUSION",
    }
    (out/f"f12s_worker{worker}.json").write_text(
        json.dumps(result,indent=2,ensure_ascii=False,sort_keys=True))
    return result

def validate_cell(cell,source):
    try:
        predecessor,instance=cell_attributes(cell["cell_index"])
        prior_calls=cell["prior_calls"]
        target=cell["target_call"]
        target_metadata=cell["target_model"]
        prefix_metadata=cell["prefix_model"]
        names=list(WARMUP)+[predecessor,TARGET]
        rows=prior_calls+[target]
        if not (
            cell["study"]==STUDY and
            cell["worker"] in WORKERS and
            cell["predecessor"]==predecessor and
            cell["instance"]==instance and
            cell["history"]==names[:-1] and
            cell["target_anchor_id"]==TARGET and
            cell["same_instance_for_target"]==(instance=="reuse") and
            cell["model_instance_init_count"]==(2 if instance=="fresh" else 1) and
            cell["transfer_worlds_evaluated"] is False and
            cell["product_integrated"] is False and
            cell["constructor_parameters"]==EXPECTED_MODEL_PARAMS and
            cell["platform"]["pythonhashseed"]=="0" and
            cell["platform"]["omp_num_threads"]=="4" and
            len(prior_calls)==13 and len(rows)==14 and
            prior.model_is_frozen(prefix_metadata) and
            prior.model_is_frozen(target_metadata) and
            prefix_metadata["calls"]==13 and
            target_metadata["calls"]==(14 if instance=="reuse" else 1) and
            target["model_calls_before"]==(13 if instance=="reuse" else 0) and
            target["model_calls_after"]==(14 if instance=="reuse" else 1)):
            return False
        for k,(row,name) in enumerate(zip(rows,names)):
            anchor=source[name]
            if not (
                row["kind"]==(
                    "warmup" if k<12 else
                    "predecessor" if k==12 else "target"
                ) and
                row["index"]==k and
                row["anchor_id"]==name and
                row["prompt_sha256"]==anchor["prompt_sha256"] and
                row["system_sha256"]==anchor["system_sha256"] and
                row["raw_sha256"]==prior.sha(row["raw"]) and
                row["action"]==prior.frozen.previous.parse_action(row["raw"]) and
                row["action"] in prior.frozen.previous.ACTIONS and
                row["model_calls_after"]==row["model_calls_before"]+1 and
                row["cache_before"]["kv_cache_hits_available"] is False and
                row["cache_after"]["kv_cache_hits_available"] is False and
                row["cache_before"]["kv_cache_hits"] is None and
                row["cache_after"]["kv_cache_hits"] is None):
                return False
    except (KeyError,IndexError,TypeError,ValueError):
        return False
    return True

def aggregate(workers,out=None):
    if len(workers)!=4 or {w["worker"] for w in workers}!=set(WORKERS):
        raise ValueError("All four unique workers required")
    source=sources()
    cells=[c for w in workers for c in w["cells"]]
    expected={(w,i) for w in WORKERS for i in range(4)}
    ids=[(c["worker"],c["cell_index"]) for c in cells]
    complete=(
        len(cells)==16 and len(set(ids))==16 and set(ids)==expected and
        all(len(w["cells"])==4
            and w["cell_order"]==list(WORKER_ORDERS[w["worker"]])
            and [c["cell_index"] for c in w["cells"]]==w["cell_order"]
            and all(c["worker"]==w["worker"] for c in w["cells"])
            for w in workers)
    )
    distinct_pid_by_worker=all(
        len({c["child_pid"] for c in w["cells"]})==4
        for w in workers
    )
    balanced_order=all(
        set(WORKER_ORDERS[w][position] for w in WORKERS)==set(range(4))
        for position in range(4)
    )
    verified=all(validate_cell(c,source) for c in cells)
    total=sum(len(c["prior_calls"])+1 for c in cells)
    prior_count=sum(len(c["prior_calls"]) for c in cells)
    S0=(complete and distinct_pid_by_worker and balanced_order and
        verified and total==224 and prior_count==208)
    S1=(verified and all(
        c["target_call"]["prompt_sha256"]==SOURCE_SHA
        for c in cells))
    by_cell={index:[] for index in range(4)}
    for c in cells:by_cell[c["cell_index"]].append(c["target_call"]["raw"])
    raw_equal={i:len(outputs)==4 and len(set(outputs))==1
               for i,outputs in by_cell.items()}
    S2=all(raw_equal.values())
    expected_raw=(
        '{"action":"q2"}',
        '{"action":"q0"}',
        '{"action":"q0"}',
        '{"action":"q0"}',
    )
    S3=S2 and all(by_cell[i][0]==expected_raw[i] for i in range(4))
    q2_probability={
        i:sum(x=='{"action":"q2"}' for x in by_cell[i])/4
        for i in range(4)
    }
    interaction=((q2_probability[0]-q2_probability[1])-
                 (q2_probability[2]-q2_probability[3]))
    S5=verified and all(
        c["transfer_worlds_evaluated"] is False
        and c["product_integrated"] is False for c in cells
    )
    if not S0 or not S1 or not S5:
        status="INVALID_S_PROTOCOL"
    elif not S2:
        status="REPLICATE_VARIANCE"
    elif S3:
        status="INTERACTION_REPLICATED"
    elif by_cell[0][0]==by_cell[2][0] and by_cell[1][0]==by_cell[3][0] and by_cell[0][0]!=by_cell[1][0]:
        status="INSTANCE_GENERAL_SIGNAL"
    elif by_cell[0][0]==by_cell[1][0] and by_cell[2][0]==by_cell[3][0] and by_cell[0][0]!=by_cell[2][0]:
        status="PREDECESSOR_SIGNAL"
    elif len({by_cell[i][0] for i in range(4)})==1:
        status="NO_SIGNAL"
    else:
        status="MIXED_CONTRAST"
    report={
        "study":STUDY,"status":status,
        "gates":{
            "S0_complete_16_cells_224_real_inferences":S0,
            "S1_pinned_model_instance_boundary_and_source":S1,
            "S2_four_literal_replicates_per_cell":S2,
            "S3_preregistered_interaction_q2_q0_q0_q0":S3,
            "S5_no_transfer_or_product_or_fake_cache_hits":S5,
        },
        "cells":{
            str(i):{
                "predecessor":CELLS[i][0],"instance":CELLS[i][1],
                "raw_outputs":by_cell[i],
                "q2_fraction":q2_probability[i],
            } for i in range(4)
        },
        "interaction_q2_fraction":interaction,
        "total_llm_calls":total,"prior_calls":prior_count,"target_calls":16,
        "unique_target_raws":len({value for values in by_cell.values() for value in values}),
        "kv_cache_hit_counts_available":False,
        "cache_interpretation":"n_tokens proxy is not an observed KV hit/miss counter",
        "treatment_interpretation":"Previous prompt identity includes its token length and full text, not pure semantic meaning",
        "product_integrated":False,"transfer_worlds_evaluated":False,
        "scientific_claim_scope":"diagnostic on consumed synthetic prompts only; model context, not autonomous agent learning",
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12s_summary.json").write_text(
            json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True))
        (out/"f12s_full_audit.json").write_text(
            json.dumps({**report,"workers":workers},
                       indent=2,ensure_ascii=False,sort_keys=True))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,choices=WORKERS)
    p.add_argument("--cell",type=int,choices=range(4))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12s_cells"))
    args=p.parse_args()
    if args.cell is not None:
        if args.worker is None:p.error("--cell requires --worker")
        args.out.mkdir(parents=True,exist_ok=True)
        record=run_cell(args.cell,args.worker,FrozenGGUF)
        (args.out/f"f12s_cell_w{args.worker}_c{args.cell}.json").write_text(
            json.dumps(record,indent=2,ensure_ascii=False,sort_keys=True))
        print(json.dumps({"study":STUDY,"worker":args.worker,"cell":args.cell,
                          "target_raw":record["target_call"]["raw"]},
                         sort_keys=True))
        return 0
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12s_worker*.json"))
        if len(files)!=4:raise ValueError("Require four F12-S worker artifacts")
        report=aggregate([json.loads(file.read_text()) for file in files],args.out)
        print(json.dumps({k:report[k] for k in
                          ("study","status","gates","cells","interaction_q2_fraction")},
                         sort_keys=True))
        return 0 if report["status"]!="INVALID_S_PROTOCOL" else 2
    if args.worker is None:p.error("Specify --worker or --aggregate-dir")
    report=run_worker(args.worker,args.out)
    print(json.dumps({"study":STUDY,"worker":args.worker,
                      "cells":len(report["cells"]),"status":report["status"]},
                     sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
