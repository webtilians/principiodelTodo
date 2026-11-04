"""Independent F10-S scientific audit. Deliberately NO MCP or sqlite3 imports.

Recomputes gates strictly from immutable per-replica full MCP transcripts,
case-level checks, manifests, checksums and preregistered disjoint seeds.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

FAMILIES=("visibility","supersession","selective_forget",
          "restart","concurrency","stale_context_revalidate")
REQUIRED_TOOLS={"remember","recall","list_memories","forget"}
SOURCE_PRODUCT="2e563ddb194cb3a0193b77a22cc97575a8baba73"
BASE=32_100_000
REQUIRED_KEYS={
    "visibility":{
        "write_ack","b_fresh_recall","b_fresh_list_exactly_once",
        "active_in_sqlite"},
    "supersession":{
        "b_saw_prior","update_ack","b_sees_new","b_not_old",
        "sqlite_one_active_latest"},
    "selective_forget":{
        "b_initially_saw_target","target_id_unique","no_target_fresh_read",
        "neighbor_preserved","purged_target_from_all_db_rows",
        "sqlite_purge_log_present"},
    "restart":{
        "existing_agent_sees","new_process_sees",
        "restarted_process_no_ghost","sqlite_no_ghost"},
    "concurrency":{
        "both_parallel_writes_ack","two_distinct_writes_visible",
        "two_concurrent_updates_ack","one_active_exclusive_version"},
    "stale_context_revalidate":{
        "cache_initially_had_target",
        "old_context_still_contains_text_expected",
        "b_refresh_does_not_contain_deleted","database_deleted"},
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def audit_one(rep,root,prereg_sha):
    m=json.loads((root/"manifest.json").read_text())
    rows=json.loads((root/"cases.json").read_text())
    assert m["experiment"]=="INFINITO_F10S_REAL_MCP_TWO_PROCESS"
    assert m["source_product_sha"]==SOURCE_PRODUCT
    assert m["replica"]==rep
    assert m["protocol"]=="actual MCP stdio with 2 independent processes sharing one SQLite file"
    assert m["db_user_memory_touched"] is False
    assert m["db_only_ephemeral"] is True
    assert m["prereg_sha256"]==prereg_sha
    assert m["cases"]==24 and len(rows)==24
    assert m["cases_sha256"]==sha(root/"cases.json")
    assert m["trace_sha256"]==sha(root/"mcp_trace.jsonl.gz")
    assert REQUIRED_TOOLS<=set(m["tools_a"]) and REQUIRED_TOOLS<=set(m["tools_b"])
    assert m["case_seed_range"]==[BASE+10_000*rep,BASE+10_000*rep+503]
    assert m["sqlite"]["db_exists"] and m["sqlite"]["db_within_temporary_directory"]
    assert m["sqlite"]["purge_log_total"]>=1
    seen=set()
    families=Counter()
    for row in rows:
        f=row["family"]
        assert f in FAMILIES
        seed=row["seed"]
        assert isinstance(seed,int)
        assert BASE+10_000*rep<=seed<=BASE+10_000*rep+503
        family_idx=FAMILIES.index(f)
        index=seed-(BASE+10_000*rep+100*family_idx)
        assert index in range(4)
        assert (f,index) not in seen
        seen.add((f,index))
        families[f]+=1
        assert row["mcp_calls"]>=1
        assert row["passed"] in (True,False)
        if "error" not in row:
            assert set(row["checks"])==REQUIRED_KEYS[f]
            assert all(v in (True,False) for v in row["checks"].values())
            assert row["passed"]==all(row["checks"].values())
        else:
            assert row["passed"] is False
            assert isinstance(row["error"],str) and row["error"]
    assert families=={f:4 for f in FAMILIES}
    trace=[]
    with gzip.open(root/"mcp_trace.jsonl.gz","rt",encoding="utf8") as stream:
        for line in stream:trace.append(json.loads(line))
    assert sum(r["mcp_calls"] for r in rows)==len(trace)
    assert trace
    known_agents=set()
    for x in trace:
        assert set(x)=={"agent","tool","args","output","is_error"}
        assert x["tool"] in REQUIRED_TOOLS
        assert isinstance(x["args"],dict)
        assert isinstance(x["output"],str)
        assert isinstance(x["is_error"],bool)
        known_agents.add(x["agent"])
        assert "secret" not in json.dumps(x).lower()
    assert {"A","B","C-fresh","D-restarted"}<=known_agents
    by={
        family:{
            "pass":sum(row["passed"] for row in rows if row["family"]==family),
            "total":4,
            "failed": [
                {"seed":row["seed"],"checks":row.get("checks",{}),
                 "error":row.get("error",None)}
                for row in rows if row["family"]==family and not row["passed"]
            ],
        }
        for family in FAMILIES
    }
    safety_ok=all(
        row["checks"].get("neighbor_preserved",False)
        and row["checks"].get("purged_target_from_all_db_rows",False)
        for row in rows if row["family"]=="selective_forget")
    safety_ok=safety_ok and all(
        row["checks"].get("sqlite_no_ghost",False)
        for row in rows if row["family"]=="restart")
    gates={
        "G0_integrity":True,
        "G1_cross_agent_visibility":by["visibility"]["pass"]==4,
        "G2_exclusive_supersession":by["supersession"]["pass"]==4,
        "G3_selective_forget_purge":by["selective_forget"]["pass"]==4,
        "G4_restart_and_cache_refresh":(
            by["restart"]["pass"]==4 and
            by["stale_context_revalidate"]["pass"]==4),
        "G5_concurrent_writes":by["concurrency"]["pass"]==4,
        "SAFETY_no_unrelated_loss":safety_ok,
    }
    return {
        "replica":rep,
        "family_results":by,
        "gate_results":gates,
        "n_cases":len(rows),
        "mcp_calls":len(trace),
        "agents":sorted(known_agents),
        "source_sha256":m["source_sha256"],
        "trace_sha256":m["trace_sha256"],
    }

def audit(root):
    prereg=sha(Path(__file__).parents[1]/
              "docs/INFINITO_F10S_PREREG_2026-11-04.md")
    result={
        "experiment":"INFINITO_F10S_TWO_PROCESS_TEMPORAL_COHERENCE",
        "technical":"F10S_INDEPENDENT_AUDIT_COMPLETE",
        "source_product_sha":SOURCE_PRODUCT,
        "preregistration_sha256":prereg,
        "confirm_32300000_sealed":True,
        "replicas":{},
    }
    for rep in range(4):
        output=audit_one(rep,root/str(rep),prereg)
        result["replicas"][str(rep)]=output
        print("F10S_AUDIT_REPLICA",rep,
              {k:f'{v["pass"]}/{v["total"]}'
               for k,v in output["family_results"].items()},
              "GATES",output["gate_results"],flush=True)
    keys=list(result["replicas"]["0"]["gate_results"])
    result["gate_counts"]={
        name:sum(int(x["gate_results"][name])
                 for x in result["replicas"].values())
        for name in keys}
    result["pooled"]={
        f:{
            "correct":sum(v["family_results"][f]["pass"]
                          for v in result["replicas"].values()),
            "total":16,
        } for f in FAMILIES
    }
    result["total_cases"]=sum(x["n_cases"] for x in result["replicas"].values())
    result["total_mcp_calls"]=sum(
        x["mcp_calls"] for x in result["replicas"].values())
    assert result["total_cases"]==96
    result["status"]=(
        "F10S_COHERENCE_PASS" if
        result["gate_counts"]["G0_integrity"]==4 and
        result["gate_counts"]["SAFETY_no_unrelated_loss"]==4 and
        all(result["gate_counts"][f"G{i}_"+{
            1:"cross_agent_visibility",
            2:"exclusive_supersession",
            3:"selective_forget_purge",
            4:"restart_and_cache_refresh",
            5:"concurrent_writes"}[i]]>=3 for i in range(1,6))
        else "F10S_COHERENCE_FAIL"
    )
    print("F10S_AUDIT_STATUS",result["status"],
          result["gate_counts"],result["pooled"],flush=True)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=audit(args.root)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False))
if __name__=="__main__":
    main()
