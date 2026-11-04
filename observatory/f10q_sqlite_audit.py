"""Independent F10-Q data auditor: no MCP, SQLite, EventEffects, world imports.
Reads executed MCP SQL result logs, manifests and frozen OOD predictions only.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np

ARMS=("E100","E50","E0","E_noX","E_shuffleX","Frozen_F10P","Zero")
EVENT_ARMS=ARMS[:5]
VARIANTS=("U_before","U_after","S_before","S_after")
MODES=("iid","source0","source1","alternating")
TRAIN_PROGRAMS={"P","PX","SR","SRP","PXP","SPX","SRR","X"}
GATE_REQUIRED=("G0_real_MCP_readonly","G1_new_shadow_transfer",
               "G2_public_output_dependency","G3_SQL_causal_controls",
               "G4_safety_abstention")
def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def paired(left,right,rep,offset,mode_b="normal"):
    a={(x["seed"],x["variant"]):x for x in left}
    b={(x["seed"],x["variant"]):x for x in right}
    assert a.keys()==b.keys() and len(a)==384
    worlds=sorted({k[0] for k in a})
    assert len(worlds)==96
    differences=np.array([sum(
        int(a[(s,v)]["outcomes"]["normal"]["success"]==1)-
        int(b[(s,v)]["outcomes"][mode_b]["success"]==1)
        for v in VARIANTS)/4 for s in worlds],float)
    rng=np.random.default_rng(295000+100*rep+offset)
    choices=rng.integers(0,96,size=(2000,96))
    boot=np.mean(differences[choices],axis=1)
    return {"difference":float(np.mean(differences)),
            "ci95":[float(x) for x in np.quantile(boot,[.025,.975])],
            "n_worlds":96,"samples":2000}

def accuracy_ci(rows,rep):
    grouped=defaultdict(list)
    for r in rows:
        grouped[r["seed"]].append(int(
            r["outcomes"]["normal"]["success"]==1))
    assert len(grouped)==96 and all(len(x)==4 for x in grouped.values())
    values=np.array([np.mean(grouped[s]) for s in sorted(grouped)])
    rng=np.random.default_rng(295000+100*rep)
    mean=values.mean()
    boot=np.mean(values[rng.integers(0,96,size=(2000,96))],axis=1)
    return {"mean":float(mean),
            "ci95":[float(v) for v in np.quantile(boot,[.025,.975])]}

def inspect_calls(calls):
    for c in calls:
        assert set(c)=={"tool","sql","params","raw"}
        assert c["tool"]=="sql_select"
        assert c["sql"].lower().startswith("select ")
        assert ";" not in c["sql"]
        assert isinstance(c["params"],list)
        payload=json.loads(c["raw"])
        assert set(payload)=={"columns","rows"}
        assert isinstance(payload["columns"],list)
        assert isinstance(payload["rows"],list)
        assert all(len(row)==len(payload["columns"]) for row in payload["rows"])

def original_training(folder,meta,rep):
    rows=[]
    train=folder/"train_mcp.jsonl.gz"
    assert sha(train)==meta["train_log_sha"]
    observed_per_arm={a:0 for a in EVENT_ARMS}
    counts=Counter()
    for line in gzip.open(train,"rt"):
        r=json.loads(line)
        assert set(r)=={
            "trial","seed","variant","case_sha","source","flip",
            "public_bit","program","answer","tool_calls","events","projected"}
        assert r["trial"]==len(rows)+1
        assert r["program"] in TRAIN_PROGRAMS
        assert r["program"]!="SRX"
        assert 29100000+10000*rep<=r["seed"]<29100000+10000*rep+128
        assert r["variant"] in VARIANTS
        assert (r["source"],r["flip"]) in (
            ("PUBLIC",0),("PUBLIC",1),("MEMORY",0))
        assert r["public_bit"] in (0,1)
        assert set(r["projected"])==set(EVENT_ARMS)
        assert len(r["events"])==len(r["program"])
        inspect_calls(r["tool_calls"])
        for arm,ev in r["projected"].items():
            assert len(ev)==len(r["events"])
            for original,item in zip(r["events"],ev):
                assert set(original)==set(item)=={"op","output"}
                assert original["op"]==item["op"]
                assert set(original["output"]) in (
                    {"value"},{"ack"},{"error"})
                if arm=="E100":
                    assert original==item
                if arm=="E0":
                    assert item["output"] is None
                if arm=="E_noX" and item["op"]=="X":
                    assert item["output"] is None
                if arm=="E_shuffleX" and item["op"]!="X":
                    assert original==item
                if arm=="E50" and item["output"] is not None:
                    assert original==item
                if item["output"] is not None:
                    observed_per_arm[arm]+=1
        rows.append(r)
        counts[r["program"]]+=1
    assert len(rows)==1024
    assert len({(x["seed"],x["variant"]) for x in rows})==512
    assert len({x["case_sha"] for x in rows})==512
    assert sum(len(r["tool_calls"]) for r in rows)==meta["train_tool_calls"]
    assert meta["train_tool_calls"]>=1000
    assert meta["never_executed_SRX"]
    for arm in EVENT_ARMS:
        state=json.loads((folder/(arm+".json")).read_text())
        assert state["schema"]=="F10P_TOOL_OUTPUT_EVENTS_V1"
        assert state["episodes"]==1024
        assert state["events_seen"]==observed_per_arm[arm]
        assert sha(folder/(arm+".json"))==meta["snapshot_sha"][arm]
    return {
        "episodes":1024,
        "unique_train_cases":512,
        "train_tool_calls":meta["train_tool_calls"],
        "train_program_counts":dict(counts),
        "visible_event_counts":observed_per_arm,
        "train_case_hashes":{r["case_sha"] for r in rows}
    }

def analyze_replica(rep,folder):
    meta=json.loads((folder/"frozen.json").read_text())
    complete=json.loads((folder/"complete.json").read_text())
    assert meta["state"]=="F10Q_ALL_MEMORIES_FROZEN_BEFORE_OOD_GENERATED"
    assert complete["state"]=="F10Q_OOD_EVAL_COMPLETE_NO_MODEL_UPDATE"
    assert meta["replica"]==rep
    assert meta["mcp_protocol"]=="real_stdio_client_and_server"
    assert "sql_select" in meta["mcp_tools"]
    assert meta["readonly_insert_rejected"] and complete["readonly_insert_rejected"]
    assert meta["training_trials"]==1024
    assert meta["train_unique_cases"]==512
    assert len(meta["snapshot_sha"])==5
    assert sha(folder/"frozen.json")==complete["frozen_sha"]
    assert sha(folder/"ood_rows.json")==complete["ood_sha"]
    assert complete["model_sha_still_immutable"]
    assert complete["ood_cases"]==512 and complete["ood_prediction_rows"]==3584
    training=original_training(folder,meta,rep)
    assert complete["total_mcp_select_calls"]>=meta["train_tool_calls"]
    predictions=json.loads((folder/"ood_rows.json").read_text())
    assert len(predictions)==3584
    assert len({(r["arm"],r["case_sha"]) for r in predictions})==len(predictions)
    assert all(r["arm"] in ARMS for r in predictions)
    assert all(29400000+10000*rep<=r["seed"]<
               29400000+10000*rep+128 for r in predictions)
    origin={}
    baseline=[r for r in predictions if r["arm"]=="E100"]
    assert len(baseline)==512
    invalids=Counter()
    for row in baseline:
        assert row["replica"]==rep
        assert row["base_world"]==row["seed"]-29400000-10000*rep
        assert row["variant"] in VARIANTS
        assert row["source_mode"]==MODES[row["base_world"]%4]
        assert row["n"]==(9 if row["base_world"]%2==0 else 11)
        assert row["task"]=="MEMORY" and row["flip"]==1
        assert row["invalid"]==(
            "missing" if row["base_world"]%8==0 else
            ("null" if row["base_world"]%8==1 else None))
        if row["invalid"] is None:
            assert row["truth"] in (0,1)
        else:
            assert row["truth"] is None
        invalids[row["invalid"] or "valid"]+=1
        origin[(row["seed"],row["variant"])]=row
        for outcome in row["outcomes"].values():
            inspect_calls(outcome["calls"])
    assert invalids=={"valid":384,"missing":64,"null":64}
    assert not set(r["case_sha"] for r in baseline)&training["train_case_hashes"]
    assert len({r["case_sha"] for r in baseline})==512
    for row in predictions:
        b=origin[(row["seed"],row["variant"])]
        assert all(row[k]==b[k] for k in (
            "case_sha","truth","invalid","public_bit","n","source_mode",
            "task","flip","base_world"))
        assert row["program"]=="EMPTY" or (
            len(row["program"])<=3 and set(row["program"])<={"P","S","R","X"})
        assert set(row["outcomes"])==(
            {"normal","x_disabled","no_accredit"} if row["arm"]=="E100"
            else {"normal"})
        for outcome in row["outcomes"].values():
            assert (outcome["answer"] in (0,1,None))
            assert outcome["abstain"]==(outcome["answer"] is None)
            if row["truth"] is None or outcome["answer"] is None:
                assert outcome["success"] is None
            else:
                assert outcome["success"]==int(
                    outcome["answer"]==row["truth"])
            inspect_calls(outcome["calls"])
    by={}
    for arm in ARMS:
        rr=[r for r in predictions if r["arm"]==arm]
        assert len(rr)==512
        valid=[x for x in rr if x["invalid"] is None]
        invalid=[x for x in rr if x["invalid"] is not None]
        by[arm]={
            "valid_correct":sum(x["outcomes"]["normal"]["success"]==1
                                for x in valid),
            "valid_total":384,
            "accuracy":sum(x["outcomes"]["normal"]["success"]==1
                           for x in valid)/384,
            "abstained_invalid":sum(x["outcomes"]["normal"]["abstain"]
                                    for x in invalid),
            "invalid_total":128,
            "programs":dict(Counter(x["program"] for x in valid)),
            "by_variant":{
                v:float(np.mean([r["outcomes"]["normal"]["success"]==1
                                  for r in valid if r["variant"]==v]))
                for v in VARIANTS},
            "by_source":{
                v:float(np.mean([r["outcomes"]["normal"]["success"]==1
                                  for r in valid if r["source_mode"]==v]))
                for v in MODES},
            "by_n":{
                str(n):float(np.mean([r["outcomes"]["normal"]["success"]==1
                                      for r in valid if r["n"]==n]))
                for n in (9,11)},
        }
    valid={arm:[r for r in predictions if r["arm"]==arm and
                r["invalid"] is None] for arm in ARMS}
    e100=valid["E100"]
    accuracy=accuracy_ci(e100,rep)
    contrasts={
        "E100_minus_E0":paired(e100,valid["E0"],rep,1),
        "E100_minus_noX":paired(e100,valid["E_noX"],rep,2),
        "E100_minus_Xdisabled":paired(e100,e100,rep,3,"x_disabled"),
        "E100_minus_noAccreditation":paired(
            e100,e100,rep,4,"no_accredit"),
    }
    def gate_contrast(name,threshold):
        return contrasts[name]["difference"]>=threshold and (
            contrasts[name]["ci95"][0]>0)
    em=by["E100"]
    gates={
        "G0_real_MCP_readonly":(
            training["episodes"]==1024 and
            training["train_tool_calls"]>=1000 and
            meta["readonly_insert_rejected"] and
            meta["never_executed_SRX"]
        ),
        "G1_new_shadow_transfer":(
            em["accuracy"]>=.85 and accuracy["ci95"][0]>.75 and
            min(em["by_variant"].values())>=.75 and
            min(em["by_source"].values())>=.75 and
            min(em["by_n"].values())>=.75 and
            em["programs"].get("SRX",0)>=.9*384
        ),
        "G2_public_output_dependency":(
            gate_contrast("E100_minus_E0",.20) and
            gate_contrast("E100_minus_noX",.20)),
        "G3_SQL_causal_controls":(
            gate_contrast("E100_minus_Xdisabled",.30) and
            gate_contrast("E100_minus_noAccreditation",.10)),
        "G4_safety_abstention":(
            em["abstained_invalid"]/128>=.99 and
            meta["readonly_insert_rejected"]),
        "G5_E50_Frozen_diagnostic":(
            by["E50"]["accuracy"]>=.85 and
            by["Frozen_F10P"]["accuracy"]>=.85),
    }
    return {
        "original_preflight":{
            k:v for k,v in meta.items() if k in (
                "prereg_sha","original_f10p_E100_sha","original_f10p_frozen_sha",
                "snapshot_sha","train_log_sha","mcp_tools",
                "mcp_protocol","readonly_insert_rejected","source_sha")},
        "training":{k:v for k,v in training.items() if k!="train_case_hashes"},
        "arms":by,"E100_ci95_world":accuracy,
        "causal_contrasts":contrasts,
        "gates":gates,
        "total_sql_tool_calls":complete["total_mcp_select_calls"],
        "elapsed_seconds":complete["elapsed_seconds"],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args()
    output={"technical":"F10Q_INDEPENDENT_AUDIT_COMPLETE",
            "status":None,"replicas":{}}
    for rep in range(4):
        record=analyze_replica(rep,a.root/str(rep))
        output["replicas"][str(rep)]=record
        print("F10Q_REPLICA",rep,
              {k:round(v["accuracy"],4) for k,v in record["arms"].items()},
              "GATES",record["gates"],flush=True)
    names=tuple(output["replicas"]["0"]["gates"])
    output["gate_counts"]={
        k:sum(int(v["gates"][k]) for v in output["replicas"].values())
        for k in names}
    output["status"]=(
        "F10Q_MCP_SHADOW_PASS" if
        output["gate_counts"]["G0_real_MCP_readonly"]==4 and
        output["gate_counts"]["G4_safety_abstention"]==4 and
        all(output["gate_counts"][k]>=3 for k in (
            "G1_new_shadow_transfer","G2_public_output_dependency",
            "G3_SQL_causal_controls"))
        else "F10Q_MCP_SHADOW_FAIL")
    output["pooled_descriptive"]={
        arm:{"correct":sum(x["arms"][arm]["valid_correct"]
                 for x in output["replicas"].values()),
             "valid_total":1536}
        for arm in ARMS}
    a.out.write_text(json.dumps(output,indent=2,ensure_ascii=False))
    print("F10Q_SCIENTIFIC_RESULT",output["status"],
          output["gate_counts"],output["pooled_descriptive"],flush=True)

if __name__=="__main__":
    main()
