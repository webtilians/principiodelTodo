"""F12-U independent prefix path confirmation by cached-token sentinel.

Preregistered BEFORE CODE at a800edcd898fb3ea2d96bfa939ee1d2cdab54ccb.
No modification of frozen F12-T scientific gates. Only F12-N consumed prompts.
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
from observatory import f12t_llamacpp_prefix_internals as old
from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256

STUDY="F12-U"
PREDECESSORS=("O21-disabled","O0-disabled")
INTERVENTIONS=("TRACE","PREFIX_ZERO","RESET","FRESH")
CELLS=tuple((p,intervention) for p in PREDECESSORS for intervention in INTERVENTIONS)
WORKER_ORDERS=(tuple(range(8)),tuple(reversed(range(8))),
               (3,4,5,6,7,0,1,2))
WORKERS=(0,1,2)
SOURCE_SHA=old.SOURCE_SHA
EXPECTED_MODEL_PARAMS=old.previous.EXPECTED_MODEL_PARAMS
WARMUP=old.WARMUP
TARGET="O21-start"
EXPECTED_REAL_BACKEND="0.3.36"

def sha(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def sources():
    return old.sources()

def cell_attributes(index):
    if index not in range(8):raise ValueError("Cell outside prereg")
    return CELLS[index]

def trace_target(model, anchor, intervention):
    gen=model.generator
    before=model.metadata()
    state_before=old.state_proxy(model)
    token_before=int(gen.input_ids[0]) if gen.n_tokens>0 else None
    mutation=None
    if intervention=="PREFIX_ZERO":
        if gen.n_tokens<=0 or token_before is None or token_before<0:
            raise ValueError("No valid cached tokens for sentinel intervention")
        gen.input_ids[0]=-1
        mutation={"old_id":token_before,"sentinel":-1,
                  "seen_before_target":int(gen.input_ids[0])}
    elif intervention=="RESET":
        gen.reset()
        if gen.n_tokens!=0 or gen._requires_eval is not True:
            raise ValueError("reset() ineffective")
    elif intervention not in ("TRACE","FRESH"):
        raise ValueError("Unknown target intervention")

    state_at_target=old.state_proxy(model)
    generate_orig=gen.generate
    sample_orig=gen.sample
    kv_orig=gen._ctx.kv_cache_seq_rm
    gen_snapshots=[]
    samples=[]
    kv_events=[]

    def generate_hook(tokens,*args,**kwargs):
        token_list=[int(t) for t in tokens]
        cached=[int(v) for v in gen._input_ids]
        prefix=0
        for a,b in zip(cached,token_list):
            if a!=b:break
            prefix+=1
        gen_snapshots.append({
            "target_token_count":len(token_list),
            "target_token_ids_sha256":sha(json.dumps(token_list,separators=(",",":"))),
            "initial_cached_token_count":len(cached),
            "initial_cached_token_sha256":sha(json.dumps(cached,separators=(",",":"))),
            "computed_longest_prefix":prefix,
            "cached_first_id":cached[0] if cached else None,
            "target_first_id":token_list[0] if token_list else None,
            "reset_argument":kwargs.get("reset",True),
            "requires_eval_at_generate":gen._requires_eval,
        })
        return generate_orig(tokens,*args,**kwargs)

    def sample_hook(*args,**kwargs):
        token=sample_orig(*args,**kwargs)
        samples.append(int(token))
        return token

    def kv_hook(seq_id,p0,p1):
        outcome=bool(kv_orig(seq_id,p0,p1))
        kv_events.append({
            "seq_id":int(seq_id),"p0":int(p0),"p1":int(p1),
            "returned":outcome,"delegated":True,
        })
        return outcome

    gen.generate=generate_hook
    gen.sample=sample_hook
    gen._ctx.kv_cache_seq_rm=kv_hook
    try:
        raw=model.chat(old.SYSTEM,anchor["prompt"])
    finally:
        gen.generate=generate_orig
        gen.sample=sample_orig
        gen._ctx.kv_cache_seq_rm=kv_orig
    after=model.metadata()
    state_after=old.state_proxy(model)
    if mutation is not None:
        mutation["after_target_first_id"]=int(gen.input_ids[0])
        mutation["replaced_by_decode"]=mutation["after_target_first_id"]==token_before

    return {
        "kind":"target","anchor_id":TARGET,"index":13,
        "prompt_sha256":anchor["prompt_sha256"],
        "system_sha256":anchor["system_sha256"],
        "raw":raw,"raw_sha256":sha(raw),"action":old.frozen.previous.parse_action(raw),
        "intervention":intervention,"mutation":mutation,
        "token_id_before_intervention":token_before,
        "state_before":state_before,"state_at_target":state_at_target,
        "state_after":state_after,
        "generate_snapshots":gen_snapshots,
        "chosen_sampled_token_ids":samples,
        "kv_seq_rm_events":kv_events,
        "hooks_restored":(
            gen.generate==generate_orig and
            gen.sample==sample_orig and
            gen._ctx.kv_cache_seq_rm==kv_orig),
        "model_calls_before":before["calls"],"model_calls_after":after["calls"],
        "prompt_tokens":after["prompt_tokens"]-before["prompt_tokens"],
        "completion_tokens":after["completion_tokens"]-before["completion_tokens"],
    }

def run_cell(index,worker,model_factory=FrozenGGUF):
    if worker not in WORKERS:raise ValueError("Unknown worker")
    predecessor,intervention=cell_attributes(index)
    ref=sources()
    model=model_factory()
    prior_rows=[]
    try:
        for i,name in enumerate(WARMUP):
            prior_rows.append(old.previous.prior.do_call(
                model,ref[name],"warmup",i))
        prior_rows.append(old.previous.prior.do_call(
            model,ref[predecessor],"predecessor",12))
        model_prior=model.metadata()
        if intervention=="FRESH":
            del model
            gc.collect()
            model=model_factory()
        model_target_before=model.metadata()
        target=trace_target(model,ref[TARGET],intervention)
        model_target_after=model.metadata()
    finally:
        if "model" in locals():
            del model

    return {
        "study":STUDY,"worker":worker,"cell_index":index,
        "predecessor":predecessor,"intervention":intervention,
        "history":list(WARMUP)+[predecessor],
        "source_model_after_prior":model_prior,
        "target_model_before":model_target_before,
        "target_model_after":model_target_after,
        "prior_calls":prior_rows,"target_call":target,
        "instance_reloaded":intervention=="FRESH",
        "child_pid":os.getpid(),"parent_pid":os.getppid(),
        "platform":{
            "system":platform.system(),"machine":platform.machine(),
            "processor":platform.processor(),
            "python":platform.python_version(),
            "pythonhashseed":os.environ.get("PYTHONHASHSEED"),
            "omp_num_threads":os.environ.get("OMP_NUM_THREADS"),
        },
        "constructor_parameters":EXPECTED_MODEL_PARAMS,
        "transfer_worlds_evaluated":False,
        "product_integrated":False,
        "status":"CELL_ONLY_NO_SCIENTIFIC_RESULT",
    }

def run_worker(worker,out,python_executable=None):
    if worker not in WORKERS:raise ValueError("Unknown worker")
    out.mkdir(parents=True,exist_ok=True)
    py=python_executable or sys.executable
    results=[]
    for index in WORKER_ORDERS[worker]:
        subprocess.run([
            py,"-m","observatory.f12u_prefix_path_independent",
            "--worker",str(worker),"--cell",str(index),"--out",str(out)
        ],check=True)
        file=out/f"f12u_cell_w{worker}_c{index}.json"
        if not file.is_file():raise ValueError("Missing OS child audit")
        results.append(json.loads(file.read_text()))
    if len(results)!=8 or len({r["child_pid"] for r in results})!=8:
        raise ValueError("Worker requires eight independent subprocesses")
    worker_report={
        "study":STUDY,"worker":worker,
        "cell_order":list(WORKER_ORDERS[worker]),
        "cells":results,"status":"WORKER_ONLY",
    }
    (out/f"f12u_worker{worker}.json").write_text(
        json.dumps(worker_report,indent=2,ensure_ascii=False,sort_keys=True))
    return worker_report

def model_pinned(m):
    return (
        m.get("real_model") is True
        and m.get("weights_sha256")==FIXED_MODEL_SHA256
        and m.get("revision")=="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
        and m.get("backend_version")==EXPECTED_REAL_BACKEND
        and m.get("n_threads")==4
        and m.get("n_ctx")==4096
        and m.get("temperature")==0.0
        and m.get("max_tokens")==96
    )

def validate_cell(cell,ref):
    try:
        p,intervention=cell_attributes(cell["cell_index"])
        t=cell["target_call"]
        rows=cell["prior_calls"]
        g=t["generate_snapshots"]
        before=cell["target_model_before"]
        after=cell["target_model_after"]
        expected_calls=0 if intervention=="FRESH" else 13
        if not (
            cell["study"]==STUDY and
            cell["worker"] in WORKERS and
            cell["predecessor"]==p and cell["intervention"]==intervention and
            cell["history"]==list(WARMUP)+[p] and
            cell["constructor_parameters"]==EXPECTED_MODEL_PARAMS and
            cell["transfer_worlds_evaluated"] is False and
            cell["product_integrated"] is False and
            cell["instance_reloaded"]==(intervention=="FRESH") and
            cell["platform"]["pythonhashseed"]=="0" and
            cell["platform"]["omp_num_threads"]=="4" and
            len(rows)==13 and
            all(model_pinned(m) for m in
                (cell["source_model_after_prior"],before,after)) and
            cell["source_model_after_prior"]["calls"]==13 and
            before["calls"]==expected_calls and
            after["calls"]==expected_calls+1 and
            t["model_calls_before"]==expected_calls and
            t["model_calls_after"]==expected_calls+1 and
            t["anchor_id"]==TARGET and t["index"]==13 and
            t["intervention"]==intervention and
            t["prompt_sha256"]==SOURCE_SHA and
            t["system_sha256"]==ref[TARGET]["system_sha256"] and
            t["raw_sha256"]==sha(t["raw"]) and
            t["action"]==old.frozen.previous.parse_action(t["raw"]) and
            t["action"] in old.ALLOWED_ACTIONS and
            t["hooks_restored"] is True and len(g)==1 and
            len(t["chosen_sampled_token_ids"])>0 and
            g[0]["target_token_count"]>0 and
            g[0]["reset_argument"] is True):
            return False
        for i,(row,name) in enumerate(zip(rows,list(WARMUP)+[p])):
            a=ref[name]
            if not (
                row["anchor_id"]==name and row["index"]==i
                and row["raw_sha256"]==sha(row["raw"])
                and row["prompt_sha256"]==a["prompt_sha256"]
                and row["system_sha256"]==a["system_sha256"]
                and row["action"]==old.frozen.previous.parse_action(row["raw"])
                and row["action"] in old.ALLOWED_ACTIONS
                and row["kind"]==("warmup" if i<12 else "predecessor")):
                return False
        if intervention=="PREFIX_ZERO":
            m=t["mutation"]
            if not (m and m["old_id"]>=0 and m["sentinel"]==-1 and
                    m["seen_before_target"]==-1 and m["replaced_by_decode"] is True):
                return False
            if g[0]["computed_longest_prefix"]!=0 or g[0]["cached_first_id"]!=-1:
                return False
        elif t["mutation"] is not None:
            return False
        if intervention in ("TRACE","PREFIX_ZERO") and (
            t["state_before"]["n_tokens_observed"] is None or
            t["state_before"]["n_tokens_observed"]<=0
        ):return False
        if intervention in ("RESET","FRESH") and (
            t["state_at_target"]["n_tokens_observed"]!=0 or
            g[0]["computed_longest_prefix"]!=0
        ):return False
        if intervention=="TRACE" and g[0]["computed_longest_prefix"]<=0:
            return False
        ev=t["kv_seq_rm_events"]
        if not ev or any(not x["delegated"] for x in ev):
            return False
        if intervention=="TRACE" and not any(x["p0"]>0 for x in ev):
            return False
        if intervention!="TRACE" and not any(x["p0"]==0 for x in ev):
            return False
    except (KeyError,IndexError,TypeError,ValueError):
        return False
    return True

def aggregate(workers,recovery_manifest,out=None):
    if len(workers)!=3 or {w["worker"] for w in workers}!=set(WORKERS):
        raise ValueError("Need three unique workers")
    recovered=(
        recovery_manifest.get("original_run_id")==42675902103
        and recovery_manifest.get("reexecuted_model_inferences")==0
        and recovery_manifest.get("strict_saved_and_reevaluated_summary_identical") is True
        and recovery_manifest.get("cells_revalidated")==30
        and all(recovery_manifest.get("original_scientific_gates",{}).values())
    )
    cells=[c for w in workers for c in w["cells"]]
    pairs=[(c["worker"],c["cell_index"]) for c in cells]
    complete=(
        len(cells)==24 and len(set(pairs))==24 and
        set(pairs)=={(w,i) for w in WORKERS for i in range(8)} and
        all(
            w["cell_order"]==list(WORKER_ORDERS[w["worker"]])
            and [c["cell_index"] for c in w["cells"]]==w["cell_order"]
            and len(w["cells"])==8
            and all(c["worker"]==w["worker"] for c in w["cells"])
            and len({c["child_pid"] for c in w["cells"]})==8
            for w in workers
        )
    )
    ref=sources()
    integrity={f'{c["worker"]}:{c["cell_index"]}':validate_cell(c,ref)
               for c in cells}
    counts=sum(len(c["prior_calls"])+1 for c in cells)
    priors=sum(len(c["prior_calls"]) for c in cells)
    U0=recovered and complete and counts==336 and priors==312
    U1=U0 and all(
        c["target_call"]["prompt_sha256"]==SOURCE_SHA and
        c["constructor_parameters"]==EXPECTED_MODEL_PARAMS
        for c in cells
    ) and all(integrity.values()) and len(integrity)==24
    token_shas={c["target_call"]["generate_snapshots"][0]["target_token_ids_sha256"]
                for c in cells if c["target_call"]["generate_snapshots"]}
    U1=U1 and len(token_shas)==1
    U2=U1 and all(integrity.values())
    bycell={i:[] for i in range(8)}
    for c in cells:bycell[c["cell_index"]].append(c["target_call"]["raw"])
    equal={i:len(v)==3 and len(set(v))==1 for i,v in bycell.items()}
    U3=all(equal.values()) and (
        bycell[CELLS.index(("O21-disabled","TRACE"))]==['{"action":"q2"}']*3
        and bycell[CELLS.index(("O0-disabled","TRACE"))]==['{"action":"q0"}']*3
    )
    expected={
        ("O21-disabled","TRACE"):'{"action":"q2"}',
        ("O21-disabled","PREFIX_ZERO"):'{"action":"q0"}',
        **{(p,intervention):'{"action":"q0"}'
            for p in PREDECESSORS for intervention in INTERVENTIONS
            if not (p=="O21-disabled" and intervention=="TRACE")}
    }
    U4=U3 and all(
        bycell[i]==[expected[CELLS[i]]]*3
        for i in range(8)
    )
    token_divergence={}
    for p in PREDECESSORS:
        base=next(c for c in cells if
                  c["predecessor"]==p and c["intervention"]=="TRACE")
        alt=next(c for c in cells if
                 c["predecessor"]==p and c["intervention"]=="PREFIX_ZERO")
        seq1=base["target_call"]["chosen_sampled_token_ids"]
        seq2=alt["target_call"]["chosen_sampled_token_ids"]
        mismatch=next((j for j in range(min(len(seq1),len(seq2)))
                       if seq1[j]!=seq2[j]),None)
        if mismatch is None and len(seq1)!=len(seq2):
            mismatch=min(len(seq1),len(seq2))
        token_divergence[p]=mismatch
    U4=U4 and token_divergence["O21-disabled"] is not None
    U5=all(c["transfer_worlds_evaluated"] is False and
           c["product_integrated"] is False for c in cells)
    gates={
        "U0_original_f12t_recovered_24_cells_336_calls":U0,
        "U1_model_and_identical_target_token_hash":U1,
        "U2_real_prefix_invalidation_traces":U2,
        "U3_replicated_and_passive_control":U3,
        "U4_independent_intervention_and_sample_divergence":U4,
        "U5_no_transfer_or_product":U5,
    }
    if not (U0 and U1 and U2 and U5):
        status="INVALID_U_PROTOCOL"
    elif not U3:
        status="REPLICATE_VARIANCE"
    elif U4:
        status="INDEPENDENT_PREFIX_PATH_CONFIRMED"
    elif len({raw for values in bycell.values() for raw in values})==1:
        status="NO_EFFECT_OBSERVED"
    else:
        status="MIXED"
    report={
        "study":STUDY,"status":status,"gates":gates,
        "cells":{str(i):{"predecessor":CELLS[i][0],
                         "intervention":CELLS[i][1],
                         "raw_outputs":bycell[i]} for i in range(8)},
        "target_token_ids_sha256":next(iter(token_shas)) if len(token_shas)==1 else None,
        "first_sampled_token_divergence":token_divergence,
        "original_f12t_run_id":42675902103,
        "total_llm_calls":counts,"prior_llm_calls":priors,"target_llm_calls":24,
        "real_kv_cache_hits_measured":False,
        "interpretation_limit":"Causal influence of prefix path on this runtime; no evidence of KV corruption, model learning or transfer to other families",
        "product_integrated":False,"transfer_worlds_evaluated":False,
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12u_summary.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        (out/"f12u_full_audit.json").write_text(
            json.dumps({**report,"workers":workers,"cell_integrity":integrity,
                        "retrospective_recovery":recovery_manifest},
                       indent=2,sort_keys=True,ensure_ascii=False))
        if json.loads((out/"f12u_full_audit.json").read_text())["status"]!=status:
            raise ValueError("Full audit failed roundtrip")
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--worker",type=int,choices=WORKERS)
    p.add_argument("--cell",type=int,choices=range(8))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--recovery-manifest",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12u_cells"))
    a=p.parse_args()
    if a.cell is not None:
        if a.worker is None:p.error("--cell needs --worker")
        a.out.mkdir(parents=True,exist_ok=True)
        cell=run_cell(a.cell,a.worker,FrozenGGUF)
        (a.out/f"f12u_cell_w{a.worker}_c{a.cell}.json").write_text(
            json.dumps(cell,indent=2,sort_keys=True,ensure_ascii=False))
        print(json.dumps({"study":STUDY,"worker":a.worker,"cell":a.cell,
                          "raw":cell["target_call"]["raw"]},sort_keys=True))
        return 0
    if a.aggregate_dir is not None:
        if a.recovery_manifest is None:
            p.error("--recovery-manifest required")
        files=sorted(a.aggregate_dir.glob("f12u_worker*.json"))
        if len(files)!=3:raise ValueError("Need three worker reports")
        workers=[json.loads(f.read_text()) for f in files]
        recovery=json.loads(a.recovery_manifest.read_text())
        report=aggregate(workers,recovery,a.out)
        print(json.dumps({k:report[k] for k in
                          ("study","status","gates","cells","first_sampled_token_divergence")},
                         sort_keys=True))
        return 0 if report["status"]!="INVALID_U_PROTOCOL" else 2
    if a.worker is None:p.error("Specify --worker or --aggregate-dir")
    result=run_worker(a.worker,a.out)
    print(json.dumps({"study":STUDY,"worker":a.worker,
                      "cells":len(result["cells"])},sort_keys=True))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
