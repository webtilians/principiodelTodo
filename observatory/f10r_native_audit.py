"""F10-R independent data-only auditor; no MCP/SQLite/learner imports.

Checks preregistered prospective integrity and paired workflows. Scientific
incremental-gain gate compares LEARNED with a STRONG documented MCP policy,
rather than a deliberately disabled/incorrect policy.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np

GOALS=("store","update","retrieve","erase","skip")
ACTIONS=("noop","remember","recall","forget","list_memories")
WORKFLOWS=(
    ("store","retrieve"),("update","retrieve"),("erase","retrieve"),
    ("store","erase"),("update","erase"),("skip","retrieve"))
ARMS=("DOCS","LEARNED","SHUFFLED")
SOURCE_COMMIT="f6bd6239bfc1734c04c233a5cf8cef050c6d54c2"

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def ci_gap(left,right,r,offset):
    assert len(left)==len(right)==48
    a={x["j"]:x for x in left}
    b={x["j"]:x for x in right}
    assert set(a)==set(b)==set(range(48))
    values=np.array([
        int(a[i]["success"])-int(b[i]["success"]) for i in range(48)
    ],float)
    rng=np.random.default_rng(315000+100*r+offset)
    ci=np.quantile(np.mean(values[
        rng.integers(0,48,size=(2000,48))],axis=1),[.025,.975])
    return {"delta":float(values.mean()),"ci95":[float(y) for y in ci]}

def audit_one(r,root):
    manifest=json.loads((root/"frozen.json").read_text())
    done=json.loads((root/"complete.json").read_text())
    observed=json.loads((root/"train_calls.json").read_text())
    assert manifest["replica"]==r
    assert manifest["product_commit"]==SOURCE_COMMIT
    assert manifest["temp_db_only"] and done["no_production_mcp_used"]
    assert manifest["train_complete_before_ood"]
    assert manifest["model_locked_pre_ood"]
    assert manifest["train_cases"]==100
    assert manifest["training_db_name"]=="TRAIN-mcp.db"
    assert set(["remember","recall","list_memories","forget"])<=set(
        manifest["training_protocol_tools"])
    assert done["state"]=="F10R_ORIGINAL_OOD_COMPLETED_FROZEN"
    assert sha(root/"frozen.json")==done["frozen_sha"]
    assert sha(root/"train_calls.json")==manifest["train_sha"]
    assert sha(root/"LEARNED.json")==manifest["learned_sha"]
    assert sha(root/"SHUFFLED.json")==manifest["shuffled_sha"]
    assert len(observed)==100
    assert len({x["seed"] for x in observed})==100
    assert all(x["success"] in (True,False) and
               x["action"] in ACTIONS and x["intent"] in GOALS and
               x["agent_calls"]>=0 for x in observed)
    assert Counter((x["intent"],x["action"]) for x in observed)=={
        (g,a):4 for g in GOALS for a in ACTIONS}
    assert all(31_100_000+10_000*r<=x["seed"]<
               31_100_000+10_000*(r+1) for x in observed)
    stats=json.loads((root/"LEARNED.json").read_text())
    shuffle=json.loads((root/"SHUFFLED.json").read_text())
    assert len(stats)==len(shuffle)==25
    assert all(s["trials"]==4 for s in stats.values())
    assert sum(s["success"] for s in stats.values())==sum(
        int(x["success"]) for x in observed)
    assert sum(s["success"] for s in shuffle.values())==sum(
        int(x["success"]) for x in observed), "Shuffle must preserve labels total"
    def choose(data,intent):
        order=("noop","recall","remember","forget","list_memories")
        def score(a):
            v=data[f"{intent}|{a}"]
            return ((v["success"]+1)/(v["trials"]+2)
                    -.02*(v["calls"]/v["trials"]))
        return max(order,key=lambda a:(score(a),-order.index(a)))
    assert manifest["learner_choices"]=={
        g:choose(stats,g) for g in GOALS}
    assert manifest["shuffled_choices"]=={
        g:choose(shuffle,g) for g in GOALS}
    rows={}
    for arm in ARMS:
        d=json.loads((root/(arm+".json")).read_text())
        x=d["rows"]
        assert len(x)==48
        assert d["db_file"]=="OOD-"+arm+".db"
        assert d["total_agent_calls"]==sum(z["agent_calls"] for z in x)
        assert d["total_tool_calls_including_evaluator"]>=d["total_agent_calls"]
        assert sum(int(y["success"]) for y in x)==done["results"][arm]["successes"]
        assert d["total_agent_calls"]==done["results"][arm]["agent_calls"]
        assert d["total_purges"]==done["results"][arm]["purges"]
        assert {z["j"] for z in x}==set(range(48))
        for z in x:
            assert z["replica"]==r and z["arm"]==arm
            assert z["seed"]==31_400_000+10_000*r+z["j"]
            assert z["kind"]==z["j"]%6
            assert tuple(z["workflow"])==WORKFLOWS[z["kind"]]
            assert len(z["actions"])==2
            assert z["success"]==all(z["step_success"])
            assert z["success"] in (True,False)
            assert z["agent_calls"]==sum(
                act["agent_calls"] for act in z["actions"])
            assert all(act["agent_calls"]>=0 for act in z["actions"])
            assert z["preserve_all"] in (True,False)
            assert z["deleted_absent_active"] in (True,False)
            assert z["deleted_absent_database"] in (True,False)
            assert z["old_lineage_absent_database"] in (True,False)
            assert z["question_unstored"] in (True,False)
            for act in z["actions"]:
                assert act["intent"] in GOALS
                if arm=="LEARNED":
                    assert act["action"]==manifest["learner_choices"][act["intent"]]
                elif arm=="SHUFFLED":
                    assert act["action"]==manifest["shuffled_choices"][act["intent"]]
                else:
                    expected={
                        "store":"remember","update":"remember",
                        "retrieve":"recall","erase":"forget","skip":"noop"}
                    assert act["action"]==expected[act["intent"]]
        rows[arm]=x
    for j in range(48):
        values=[next(x for x in rows[arm] if x["j"]==j) for arm in ARMS]
        assert len({z["seed"] for z in values})==1
        assert len({z["scenario_sha"] for z in values})==1
        assert len({tuple(z["workflow"]) for z in values})==1
    counts={
        arm:{"correct":sum(int(z["success"]) for z in x),
             "total":48,"accuracy":sum(int(z["success"]) for z in x)/48,
             "agent_calls":sum(z["agent_calls"] for z in x),
             "calls_per_workflow":sum(z["agent_calls"] for z in x)/48,
             "preserve_violations":sum(not z["preserve_all"] for z in x),
             "deleted_active_violations":sum(
                 not z["deleted_absent_active"] for z in x),
             "deleted_sqlite_violations":sum(
                 not z["deleted_absent_database"] for z in x),
             "lineage_violations":sum(
                 not z["old_lineage_absent_database"] for z in x),
             "question_stored_violations":sum(
                 not z["question_unstored"] for z in x),
            } for arm,x in rows.items()}
    gap=ci_gap(rows["LEARNED"],rows["DOCS"],r,0)
    shuffle_gap=ci_gap(rows["LEARNED"],rows["SHUFFLED"],r,1)
    learned=counts["LEARNED"]
    baseline=counts["DOCS"]
    relative_reduction=(baseline["agent_calls"]-learned["agent_calls"])/max(
        1,baseline["agent_calls"])
    strict_safety=all(
        counts[arm][k]==0 for arm in ("LEARNED","DOCS")
        for k in ("preserve_violations","deleted_active_violations",
                  "deleted_sqlite_violations","lineage_violations",
                  "question_stored_violations"))
    gates={
        "G0_integrity":True,
        "G1_learner_safe_90pct":(
            learned["accuracy"]>=.90 and
            all(learned[k]==0 for k in (
                "preserve_violations","deleted_active_violations",
                "deleted_sqlite_violations","lineage_violations"))),
        "G2_incremental_vs_docs":strict_safety and (
            (gap["delta"]>=.05 and gap["ci95"][0]>0 and
             learned["agent_calls"]<=baseline["agent_calls"])
            or (relative_reduction>=.15 and gap["delta"]>=-.02)),
        "G3_shuffled_diagnostic":(
            shuffle_gap["delta"]>=.10 and shuffle_gap["ci95"][0]>0),
        "G4_no_collateral_forgetting":strict_safety,
    }
    return {
        "original_source_commit":SOURCE_COMMIT,
        "model_choices":{
            "learned":manifest["learner_choices"],
            "shuffled":manifest["shuffled_choices"]},
        "scores":counts,
        "paired_learned_minus_docs":gap,
        "paired_learned_minus_shuffled":shuffle_gap,
        "relative_calls_saved":relative_reduction,
        "gates":gates,
        "snapshots_sha":{
            k:manifest[k] for k in (
                "train_sha","learned_sha","shuffled_sha","frozen_at")
            if k in manifest},
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data={"technical":"F10R_INDEPENDENT_AUDIT_COMPLETE",
          "original_product_commit":SOURCE_COMMIT,
          "replicas":{}}
    for rep in range(4):
        d=audit_one(rep,a.root/str(rep))
        data["replicas"][str(rep)]=d
        print("F10R_REPLICA",rep,
              {k:v["accuracy"] for k,v in d["scores"].items()},
              "GATES",d["gates"],flush=True)
    keys=tuple(data["replicas"]["0"]["gates"])
    data["gate_counts"]={
        key:sum(int(x["gates"][key]) for x in data["replicas"].values())
        for key in keys}
    data["pooled"]={
        arm:{
            "correct":sum(r["scores"][arm]["correct"] for r in data["replicas"].values()),
            "total":192,
            "agent_calls":sum(r["scores"][arm]["agent_calls"] for r in data["replicas"].values())}
        for arm in ARMS}
    data["status"]=(
        "F10R_INCREMENTAL_GAIN_PASS" if
        data["gate_counts"]["G0_integrity"]==4 and
        data["gate_counts"]["G4_no_collateral_forgetting"]==4 and
        data["gate_counts"]["G1_learner_safe_90pct"]>=3 and
        data["gate_counts"]["G2_incremental_vs_docs"]>=3
        else "F10R_NO_INCREMENTAL_GAIN")
    a.out.write_text(json.dumps(data,indent=2,ensure_ascii=False))
    print("F10R_FINAL_SCIENCE",data["status"],
          data["gate_counts"],data["pooled"],flush=True)

if __name__=="__main__":
    main()
