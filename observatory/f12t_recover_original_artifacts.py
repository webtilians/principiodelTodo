"""F12-T retrospective recovery ONLY: no GGUF instance or new inference.

Recompute frozen gates from original run 42675902103
(worker artifact IDs 13703537243, 13703931660, 13703303855),
compare to original f12t_summary.json (artifact 13704051652), then
write full audit using JSON-safe worker:cell integrity labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from observatory import f12t_llamacpp_prefix_internals as original

RUN_ID=42675902103
WORKERS=(0,1,2)

def sha_bytes(blob):
    return hashlib.sha256(blob).hexdigest()

def recover(original_workers_dir,original_summary_dir,out):
    files=[original_workers_dir/f"f12t_worker{worker}.json"
           for worker in WORKERS]
    if any(not file.is_file() for file in files):
        raise ValueError("Missing original F12-T worker JSON artifact")
    summary_file=original_summary_dir/"f12t_summary.json"
    if not summary_file.is_file():
        raise ValueError("Missing original F12-T scientific summary artifact")
    worker_bytes=[f.read_bytes() for f in files]
    summary_bytes=summary_file.read_bytes()
    workers=[json.loads(blob) for blob in worker_bytes]
    saved=json.loads(summary_bytes)

    # No new LLM inference: avoid historic tuple-key serialization bug.
    reevaluated=original.aggregate(workers,out=None)
    if reevaluated!=saved:
        raise ValueError("Frozen F12-T audit and preserved original summary differ")
    if saved["status"]!="PREFIX_PATH_CAUSAL_SIGNAL" or not all(saved["gates"].values()):
        raise ValueError("Unexpected frozen F12-T scientific gates")
    if saved["total_llm_calls"]!=420 or saved["prior_calls"]!=390 or saved["target_calls"]!=30:
        raise ValueError("Frozen counts invalid")
    refs=original.sources()
    integrity={}
    for worker_index,worker in enumerate(workers):
        if worker["worker"]!=worker_index or len(worker["cells"])!=10:
            raise ValueError("Original worker identity/count mismatch")
        for cell in worker["cells"]:
            key=f'{worker_index}:{cell["cell_index"]}'
            if key in integrity:
                raise ValueError("Duplicate (worker,cell) pair")
            valid=original.validate_cell(cell,refs)
            if not valid:raise ValueError(f"Invalid original raw cell {key}")
            integrity[key]=valid
    if len(integrity)!=30 or not all(integrity.values()):
        raise ValueError("Missing/invalid original cells")
    manifest={
        "original_run_id":RUN_ID,
        "reexecuted_model_inferences":0,
        "source_branch_scientific_prereg_unchanged":True,
        "original_summary_sha256":sha_bytes(summary_bytes),
        "original_worker_sha256":{
            str(index):sha_bytes(blob) for index,blob in enumerate(worker_bytes)
        },
        "strict_saved_and_reevaluated_summary_identical":True,
        "original_scientific_gates":saved["gates"],
        "original_status":saved["status"],
        "cells_revalidated":30,
        "original_total_inferences":420,
        "reconstructed_full_audit_valid":True,
    }
    full={**saved,"original_workers":workers,"cell_integrity":integrity,
          "retrospective_recovery":manifest}
    out.mkdir(parents=True,exist_ok=True)
    (out/"f12t_recovered_full_audit.json").write_text(
        json.dumps(full,indent=2,ensure_ascii=False,sort_keys=True))
    (out/"f12t_recovery_manifest.json").write_text(
        json.dumps(manifest,indent=2,ensure_ascii=False,sort_keys=True))
    reread=json.loads((out/"f12t_recovered_full_audit.json").read_text())
    if reread!=full:
        raise ValueError("Recovered full audit failed JSON roundtrip")
    return manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--workers",type=Path,required=True)
    p.add_argument("--summary",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=recover(args.workers,args.summary,args.out)
    print(json.dumps(result,sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
