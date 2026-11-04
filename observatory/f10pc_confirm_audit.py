"""F10-P-C independent JSON/manifest/scientific auditor.

No import of Torch, the world, selector, model, or learner. All 7 original
frozen memories were already authenticated and old 8960 cases fully replayed
BEFORE new 28300000 worlds were unsealed by the confirm evaluator.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

ARMS=("E100","E50","E0","E_noX","E_shuffleX",
      "B_reward","I_privileged")
MODES=("normal","shuffle","x_disabled","public_randomized")
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCES=("iid","signed_source0","signed_source1","alternating")
GATES_REQUIRED=("G0_original_preflight","G1_new_SRX_transfer",
                "G2_causal_F4_X","G3_vs_reward",
                "G4_observable_feedback")
RUN_ORIGINAL=41888234405
SOURCE_COMMIT="9de6adb6ff0299b361d72258b9d988f832256201"
G0_SKILLS_RUN=41685499939

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def mean_success(rows,mode="normal"):
    assert rows
    return float(np.mean([r["outcomes"][mode]["success"] for r in rows]))

def ci_accuracy(rows,rep):
    groups=defaultdict(list)
    for r in rows:
        groups[r["seed"]].append(r["outcomes"]["normal"]["success"])
    assert len(groups)==128 and all(len(v)==4 for v in groups.values())
    v=np.asarray([np.mean(groups[k]) for k in sorted(groups)],float)
    rng=np.random.default_rng(294000+100*rep)
    indices=rng.integers(0,128,size=(2000,128))
    bounds=np.quantile(v[indices].mean(axis=1),[.025,.975])
    return [float(x) for x in bounds]

def paired_diff(a,b,rep,offset,mode_b="normal"):
    left={(x["seed"],x["variant"]):x for x in a}
    right={(x["seed"],x["variant"]):x for x in b}
    assert left.keys()==right.keys() and len(left)==512
    seeds=sorted({s for s,v in left})
    assert len(seeds)==128
    v=np.asarray([sum(
        left[(s,variant)]["outcomes"]["normal"]["success"]-
        right[(s,variant)]["outcomes"][mode_b]["success"]
        for variant in VARIANTS)/4 for s in seeds],float)
    rng=np.random.default_rng(294000+100*rep+offset)
    ci=np.quantile(v[rng.integers(0,128,size=(2000,128))].mean(axis=1),
                   [.025,.975])
    return {"difference":float(np.mean(v)),
            "ci95":[float(x) for x in ci],
            "bootstrap_samples":2000,
            "unit":"128 worlds x four paired variants"}

def record_status(rows,arm):
    x=[r for r in rows if r["arm"]==arm]
    assert len(x)==512
    return {
        "correct":sum(r["outcomes"]["normal"]["success"] for r in x),
        "total":512,
        "accuracy":mean_success(x),
        "programs":dict(Counter(r["program"] for r in x)),
        "by_variant":{
            z:mean_success([r for r in x if r["variant"]==z])
            for z in VARIANTS},
        "by_source":{
            z:mean_success([r for r in x if r["source_mode"]==z])
            for z in SOURCES},
        "by_n":{
            str(z):mean_success([r for r in x if r["n"]==z])
            for z in (9,11)},
    }

def check_replica(rep,folder,original_science):
    original_meta=json.loads((folder/"original_replay_preconfirm.json").read_text())
    complete=json.loads((folder/"confirm_complete.json").read_text())
    rows=json.loads((folder/"confirm_rows.json").read_text())
    assert complete["state"]=="F10PC_CONFIRM_FROZEN_INPUTS_AND_PREDICTIONS_COMPLETE"
    assert original_meta["state"]=="ORIGINAL_F10P_REPLAY_PASS_BEFORE_ANY_CONFIRM_WORLD"
    assert original_meta["original_run"]==RUN_ORIGINAL
    assert complete["original_source_commit"]==SOURCE_COMMIT
    assert original_meta["original_source_commit"]==SOURCE_COMMIT
    assert original_meta["original_skill_run"]==G0_SKILLS_RUN
    assert original_meta["replica"]==rep
    assert original_meta["original_dev_ood_rows_replayed"]==8960
    assert original_meta["original_interventions_replayed"]==8960*4
    assert original_meta["original_unique_case_shas"]==3072+768+512
    assert len(original_meta["frozen_memories_sha256"])==7
    assert len(original_meta["original_checkpoint_sha256"])==5
    assert original_meta["original_dev_E100"]==(
        original_science["replicas"][str(rep)]["dev_seen"])
    assert original_meta["original_G0_pass"]==all(
        x>=.9 for x in original_meta["original_dev_E100"].values())
    assert complete["original_preflight_sha"]==sha(
        folder/"original_replay_preconfirm.json")
    assert complete["new_eval_rows_sha"]==sha(folder/"confirm_rows.json")
    assert complete["new_case_count"]==512
    assert complete["new_world_count"]==128
    assert complete["total_new_prediction_rows"]==3584
    assert complete["confirm_seed_base"]==28_300_000
    assert complete["original_snapshot_sha_verified_after_confirm"]
    assert complete["original_skill_sha_verified_after_confirm"]
    assert complete["prereg_sha"]==sha(
        Path(__file__).parents[1]/
        "docs/INFINITO_F10PC_PREREG_2026-11-04.md")
    assert len(rows)==7*512
    assert all(r["replica"]==rep and r["split"]=="confirm" for r in rows)
    assert all(r["arm"] in ARMS for r in rows)
    expected_seeds={28_300_000+10_000*rep+j for j in range(128)}
    assert {r["seed"] for r in rows}==expected_seeds
    assert len({(r["arm"],r["seed"],r["variant"]) for r in rows})==3584
    baseline={
        (r["seed"],r["variant"]):r for r in rows if r["arm"]=="E100"}
    assert len(baseline)==512
    world_groups=defaultdict(list)
    for x in baseline.values():
        assert x["base_world"]==x["seed"]-(28_300_000+10_000*rep)
        assert x["n"]==(9 if x["base_world"]%2==0 else 11)
        assert x["source_mode"]==SOURCES[(x["base_world"]//2)%4]
        assert x["task"]=="MEMORY" and x["flip"]==1
        assert x["truth"] in (0,1) and x["public_bit"] in (0,1)
        assert x["variant"] in VARIANTS
        assert isinstance(x["case_sha"],str) and len(x["case_sha"])==64
        world_groups[x["seed"]].append(x)
    assert len(world_groups)==128
    for w in world_groups.values():
        assert len(w)==4 and {r["variant"] for r in w}==set(VARIANTS)
        z={r["variant"]:r for r in w}
        assert (z["S_after"]["truth"]!=z["U_after"]["truth"] and
                z["U_after"]["truth"]==z["U_before"]["truth"]==
                z["S_before"]["truth"])
    fingerprints={}
    for r in rows:
        assert set(r["outcomes"])==set(MODES)
        assert r["program_id"] in range(85)
        assert r["program"] in (
            "EMPTY","P","S","R","X") or len(r["program"]) in (2,3)
        assert r["case_sha"]==baseline[
            (r["seed"],r["variant"])]["case_sha"]
        assert all(r[k]==baseline[(r["seed"],r["variant"])][k] for k in
                   ("task","flip","truth","winner","selected","public_bit",
                    "source_mode","n","base_world"))
        for mode in MODES:
            outcome=r["outcomes"][mode]
            assert outcome["answer"] in (0,1)
            assert outcome["success"]==int(outcome["answer"]==r["truth"])
            assert outcome["operations"] in (0,1,2,3)
            assert outcome["invalid"] in (0,1,2,3)
        assert r["outcomes"]["normal"]["operations"]==(
            0 if r["program"]=="EMPTY" else len(r["program"]))
    new_shas={r["case_sha"] for r in baseline.values()}
    assert len(new_shas)==512
    assert hashlib.sha256(
        json.dumps(sorted(new_shas)).encode()).hexdigest()==complete["new_case_shas_sha"]
    for arm in ARMS:
        arr=[x for x in rows if x["arm"]==arm]
        assert len(arr)==512
        assert {r["case_sha"] for r in arr}==new_shas
    by={arm:record_status(rows,arm) for arm in ARMS}
    e100=[r for r in rows if r["arm"]=="E100"]
    arms={arm:[r for r in rows if r["arm"]==arm] for arm in ARMS}
    contrasts={
        "E100_minus_F4shuffle":paired_diff(e100,e100,rep,1,"shuffle"),
        "E100_minus_Xdisabled":paired_diff(e100,e100,rep,2,"x_disabled"),
        "E100_minus_B":paired_diff(e100,arms["B_reward"],rep,3),
        "E100_minus_E0":paired_diff(e100,arms["E0"],rep,4),
        "E100_minus_noX":paired_diff(e100,arms["E_noX"],rep,5),
        "E100_minus_shuffleX":paired_diff(e100,arms["E_shuffleX"],rep,6),
        "E100_minus_E50":paired_diff(e100,arms["E50"],rep,7),
    }
    ci=ci_accuracy(e100,rep)
    def contrast_pass(key,threshold):
        s=contrasts[key]
        return s["difference"]>=threshold and s["ci95"][0]>0
    x=by["E100"]
    gates={
        "G0_original_preflight":original_meta["original_G0_pass"],
        "G1_new_SRX_transfer":(
            x["accuracy"]>=.8 and ci[0]>.7 and
            min(x["by_n"].values())>=.7 and
            min(x["by_source"].values())>=.7 and
            min(x["by_variant"].values())>=.7 and
            x["programs"].get("SRX",0)>=.8*512
        ),
        "G2_causal_F4_X":(
            contrast_pass("E100_minus_F4shuffle",.2) and
            contrast_pass("E100_minus_Xdisabled",.2)
        ),
        "G3_vs_reward":contrast_pass("E100_minus_B",.15),
        "G4_observable_feedback":all(
            contrast_pass(k,.15) for k in
            ("E100_minus_E0","E100_minus_noX","E100_minus_shuffleX")
        ),
        "G5_E50_diagnostic":by["E50"]["accuracy"]>=.8,
        "G6_privileged_diagnostic":by["I_privileged"]["accuracy"]>=.8,
    }
    return {
        "original":original_meta,
        "confirm_integrity":complete,
        "confirm_case_sha256":sorted(new_shas),
        "confirm_metrics":by,
        "E100_ci95_world":ci,
        "paired_contrasts":contrasts,
        "gates":gates,
    }

def audit(root,orig_science_path):
    orig_science=json.loads(orig_science_path.read_text())
    assert orig_science["status"]=="F10P_OBSERVABLE_EFFECTS_PASS"
    assert orig_science["gate_counts"]["G0_seen"]==3
    assert orig_science["unique_case_hashes"]==4*(3072+768+512)
    record={
        "experiment":"F10-P-C 28300000+ original-frozen-memory confirmation",
        "technical":"F10PC_TECHNICAL_COMPLETE",
        "original_run":RUN_ORIGINAL,
        "original_skill_run":G0_SKILLS_RUN,
        "original_source_commit":SOURCE_COMMIT,
        "prereg_sha256":sha(
            Path(__file__).parents[1]/
            "docs/INFINITO_F10PC_PREREG_2026-11-04.md"),
        "replicas":{},
    }
    all_cases=set()
    for rep in range(4):
        v=check_replica(rep,root/str(rep),orig_science)
        assert not all_cases&set(v["confirm_case_sha256"])
        all_cases.update(v["confirm_case_sha256"])
        del v["confirm_case_sha256"]
        record["replicas"][str(rep)]=v
        print("F10PC_REPLICA",rep,
              {arm:round(stats["accuracy"]*100,3)
               for arm,stats in v["confirm_metrics"].items()},
              "E100_PROGRAMS",v["confirm_metrics"]["E100"]["programs"],
              "GATES",v["gates"],flush=True)
    assert len(all_cases)==4*512==2048
    record["unique_confirm_cases"]=len(all_cases)
    keys=tuple(record["replicas"]["0"]["gates"])
    record["gate_counts"]={
        key:sum(int(r["gates"][key]) for r in record["replicas"].values())
        for key in keys}
    record["pooled_descriptive"]={
        arm:{
            "correct":sum(r["confirm_metrics"][arm]["correct"]
                for r in record["replicas"].values()),
            "total":2048,
        }
        for arm in ARMS}
    record["status"]=(
        "F10PC_CONFIRM_PASS" if all(record["gate_counts"][key]>=3
           for key in GATES_REQUIRED)
        else "F10PC_CONFIRM_FAIL"
    )
    print("F10PC_CONFIRM_ORIGINAL_AUDIT",record["status"],
          record["gate_counts"],record["pooled_descriptive"],flush=True)
    return record

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--original-science",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    data=audit(a.root,a.original_science)
    a.out.write_text(json.dumps(data,indent=2,ensure_ascii=False))
