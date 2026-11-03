"""F10-N-C INDEPENDENT confirmation audit. NEVER imports generator, actor or torch.

All scientific gates, bootstrap unit and seed settings were prospectively
registered before new 26300000+ worlds. Four variants from one base world
are resampled as a BLOCK. Previous G0 is an immutable prerequisite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

ARMS=("reward_table","additive_reward_table",
      "learned_effects_uniform","learned_effects_directed")
MODES=("normal","shuffle","x_disabled","public_randomized")
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCES=("iid","signed_source0","signed_source1","alternating")
ORIGINAL_RUN_N=41714113859
ORIGINAL_RUN_L=41685499939
ORIGINAL_CODE="6d3e7bd8753f3ad5d1d4a6e44208866bac036298"

def score(r,mode="normal"):
    return int(r["outcomes"][mode]["success"])

def rate(rows,mode="normal"):
    assert rows
    return sum(score(r,mode) for r in rows)/len(rows)

def grouped_ci(rows,rep):
    worlds=defaultdict(list)
    for r in rows:
        worlds[r["seed"]].append(score(r))
    assert len(worlds)==128 and all(len(v)==4 for v in worlds.values())
    values=np.array([np.mean(v) for _,v in sorted(worlds.items())],np.float64)
    rng=np.random.default_rng(268000+100*rep)
    bootstrap=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
    return [float(z) for z in np.quantile(bootstrap,(.025,.975))]

def grouped_contrast(left,right,rep,right_mode="normal",offset=1):
    a={(r["seed"],r["variant"]):r for r in left}
    b={(r["seed"],r["variant"]):r for r in right}
    assert len(a)==len(b)==512 and a.keys()==b.keys()
    seeds=sorted({seed for seed,_ in a})
    assert len(seeds)==128
    deltas=np.array([
        sum(score(a[(seed,v)])-score(b[(seed,v)],right_mode)
            for v in VARIANTS)/4
        for seed in seeds],np.float64)
    rng=np.random.default_rng(268000+100*rep+offset)
    bootstrap=deltas[rng.integers(0,len(deltas),
                                 size=(2000,len(deltas)))].mean(1)
    return {
        "estimate":float(deltas.mean()),
        "ci95":[float(z) for z in np.quantile(bootstrap,(.025,.975))],
        "worlds":128,
        "resamples":2000,
        "unit":"base_world_with_four_variants",
    }

def stats(rows):
    programs=Counter(r["program"] for r in rows)
    return {
        "n_cases":len(rows),
        "n_base_worlds":len({r["seed"] for r in rows}),
        "accuracy":rate(rows),
        "correct":sum(score(r) for r in rows),
        "srx_fraction":programs["SRX"]/len(rows),
        "program_counts":dict(programs.most_common()),
        "average_operations":sum(r["outcomes"]["normal"]["operations"]
                                 for r in rows)/len(rows),
        "average_return":sum(r["outcomes"]["normal"]["reward"]
                             for r in rows)/len(rows),
        "selection_accuracy":sum(r["selected"]==r["winner"]
                                  for r in rows)/len(rows),
        "oracle_srx_accuracy":sum(r["oracle_srx_accuracy_diagnostic"]
                                    for r in rows)/len(rows),
        "shuffle_accuracy":rate(rows,"shuffle"),
        "x_disabled_accuracy":rate(rows,"x_disabled"),
        "by_n":{str(n):rate([r for r in rows if r["n"]==n])
                for n in (9,11)},
        "by_source":{s:rate([r for r in rows if r["source_mode"]==s])
                     for s in SOURCES},
        "by_variant":{v:rate([r for r in rows if r["variant"]==v])
                      for v in VARIANTS},
        "constant_zero_baseline":sum(r["truth"]==0 for r in rows)/len(rows),
        "constant_one_baseline":sum(r["truth"]==1 for r in rows)/len(rows),
    }

def audit(root,original_f10n_audit_path,original_f10l_g0_path):
    archive=json.loads(original_f10n_audit_path.read_text())
    g0=json.loads(original_f10l_g0_path.read_text())
    assert archive["technical"]=="F10N_TECHNICAL_COMPLETE"
    assert archive["status"]=="F10N_EFFECT_MEMORY_COMPOSITION_PASS"
    assert archive["confirm_26300000_unopened"]
    assert archive["gate_counts"]["G0_seen_tasks"]>=3
    assert archive["original_f10l_run"]==ORIGINAL_RUN_L
    assert g0["technical"]=="F10L_PHASE1_TECHNICAL_COMPLETE"
    assert g0["eligible_for_composition"]
    report={
        "experiment":"F10-N-C: independent new-seed confirmation of frozen effect-memory",
        "technical":"F10NC_TECHNICAL_COMPLETE",
        "original_f10n_run_id":ORIGINAL_RUN_N,
        "original_f10n_code_commit":ORIGINAL_CODE,
        "original_f10l_run_id":ORIGINAL_RUN_L,
        "original_g0_prerequisite":archive["gate_counts"]["G0_seen_tasks"],
        "confirm_world_namespace":26300000,
        "no_training_or_memory_update":True,
        "prespecified_bootstrap":"2000 resamples per 128 base worlds; grouped quartets",
        "replicas":{},
    }
    all_case_hashes=set()
    for rep in range(4):
        data=json.loads((root/(str(rep)+".json")).read_text())
        assert data["experiment"]=="F10-N-C original immutable 26300000+ confirmation"
        assert data["status"]=="CONFIRM_GENERATED_ONCE_AFTER_SHA_AND_ORIGINAL_OLD_OOD_REGRESSION"
        assert data["replica"]==rep
        assert data["original_f10n_run_id"]==ORIGINAL_RUN_N
        assert data["original_f10n_code_commit"]==ORIGINAL_CODE
        assert data["original_f10l_run_id"]==ORIGINAL_RUN_L
        assert data["namespace"]==26300000
        assert data["n_base_worlds"]==128 and data["n_cases"]==512
        assert data["n_rows"]==2048
        assert data["no_training_or_memory_update"]
        assert data["original_ood_worlds_bit_exact"]
        assert data["original_dev_ood_predictions_bit_exact"]
        assert data["uses_unmodified_original_f10n_evaluate"]
        assert data["only_input_data_changed"]
        proof=data["original_sha_proof"]
        assert proof["f10l_original_sha256"]==g0["replicas"][str(rep)]["model_sha256"]
        assert data["f10l_original_sha"]==proof["f10l_original_sha256"]
        original=archive["replicas"][str(rep)]
        assert original["original_f10l_sha"]==proof["f10l_original_sha256"]
        for name,actual in proof["memory_sha256"].items():
            assert original["frozen_memory_sha"][name]["sha256"]==actual
        assert set(proof["memory_sha256"])=={
            "reward_table","learned_effects_uniform","learned_effects_directed"}
        assert set(proof["trial_log_sha256"])=={"uniform","directed"}
        assert all(len(v)==64 for v in proof["memory_sha256"].values())
        assert all(len(v)==64 for v in proof["trial_log_sha256"].values())
        assert len(proof["original_training_json_sha256"])==64
        assert len(proof["original_rows_json_sha256"])==64
        rows=data["rows"]
        assert len(rows)==2048
        assert {r["arm"] for r in rows}==set(ARMS)
        fingerprints={}
        for arm in ARMS:
            sub=[r for r in rows if r["arm"]==arm]
            assert len(sub)==512
            d={(r["seed"],r["variant"]):(
                r["case_sha"],r["truth"],r["task"],r["flip"],
                r["public_bit"],r["n"],r["source_mode"],
                r["winner"],r["selected"]
            ) for r in sub}
            assert len(d)==512
            if not fingerprints:
                fingerprints=d
            else:
                assert fingerprints==d, "arms do not see identical worlds"
            assert {r["seed"] for r in sub}=={
                26_300_000+10000*rep+j for j in range(128)}
        hashes={x[0] for x in fingerprints.values()}
        assert len(hashes)==512 and not hashes&all_case_hashes
        all_case_hashes.update(hashes)
        for j in range(128):
            rows_world=[r for r in rows
                        if r["arm"]==ARMS[0] and r["base_world"]==j]
            assert len(rows_world)==4
            v={r["variant"]:r for r in rows_world}
            assert set(v)==set(VARIANTS)
            assert all(r["seed"]==26_300_000+10000*rep+j
                       for r in rows_world)
            assert all(r["n"]==(9 if j%2==0 else 11)
                       for r in rows_world)
            assert all(r["source_mode"]==SOURCES[(j//2)%4]
                       for r in rows_world)
            assert len({r["public_bit"] for r in rows_world})==1
            assert v["U_before"]["truth"]==v["U_after"]["truth"]
            assert v["U_after"]["truth"]==v["S_before"]["truth"]
            assert v["S_after"]["truth"]!=v["U_after"]["truth"]
        for r in rows:
            assert r["split"]=="confirm"
            assert r["replica"]==rep
            assert r["task"]=="MEMORY" and r["flip"]==1
            assert r["source_mode"] in SOURCES
            assert r["variant"] in VARIANTS
            assert r["n"] in (9,11)
            assert r["winner"] in range(12)
            assert r["selected"] in range(12)
            assert r["truth"] in (0,1)
            assert 0<=r["program_id"]<85
            assert isinstance(r["predicted_score"],(int,float))
            assert r["oracle_srx_accuracy_diagnostic"] in (0,1)
            assert set(r["outcomes"])==set(MODES)
            for mode,v in r["outcomes"].items():
                assert v["answer"] in (0,1)
                assert v["success"]==int(v["answer"]==r["truth"])
                assert 0<=v["operations"]<=3 and 0<=v["invalid"]<=3
                assert -1.34<=v["reward"]<=1.0
            if r["arm"] in ("reward_table","additive_reward_table"):
                assert r["program"]!="SRX", "reward-only baseline proposed heldout!"
            if r["arm"]=="learned_effects_uniform" and r["program"]=="SRX":
                assert r["outcomes"]["normal"]["success"]==r[
                    "oracle_srx_accuracy_diagnostic"]
        arms={arm:[r for r in rows if r["arm"]==arm] for arm in ARMS}
        m={arm:stats(sub) for arm,sub in arms.items()}
        c=arms["learned_effects_uniform"]
        b=arms["additive_reward_table"]
        d=arms["learned_effects_directed"]
        contrasts={
            "C_minus_shuffle":grouped_contrast(c,c,rep,"shuffle",1),
            "C_minus_x_disabled":grouped_contrast(c,c,rep,"x_disabled",2),
            "C_minus_B":grouped_contrast(c,b,rep,"normal",3),
            "D_minus_C":grouped_contrast(d,c,rep,"normal",4),
        }
        ci=grouped_ci(c,rep)
        s=m["learned_effects_uniform"]
        cg1=(s["accuracy"]>=.80 and ci[0]>.70 and s["srx_fraction"]>=.80
             and min(s["by_n"].values())>=.70
             and min(s["by_source"].values())>=.70
             and min(s["by_variant"].values())>=.70)
        cg2=all(contrasts[k]["estimate"]>=.20 and
                contrasts[k]["ci95"][0]>0
                for k in ("C_minus_shuffle","C_minus_x_disabled"))
        cg3=(contrasts["C_minus_B"]["estimate"]>=.15 and
             contrasts["C_minus_B"]["ci95"][0]>0)
        cg4=(contrasts["D_minus_C"]["estimate"]>=.05 and
             contrasts["D_minus_C"]["ci95"][0]>0)
        report["replicas"][str(rep)]={
            "accuracy_and_strata":m,
            "composition_accuracy_ci95":ci,
            "contrasts":contrasts,
            "gates":{
                "C_G1_transfer_never_executed_SRX":cg1,
                "C_G2_F4_XOR_causality":cg2,
                "C_G3_effects_vs_reward":cg3,
                "C_G4_directed_vs_uniform":cg4,
            },
            "original_memories_sha256":proof["memory_sha256"],
            "original_f10l_sha256":proof["f10l_original_sha256"],
            "old_ood_regression_verified_before_confirm":True,
        }
        print("F10NC_REPLICA",rep,
              "OOD",{a:round(v["accuracy"],6) for a,v in m.items()},
              "SRX",s["program_counts"],"ci95",ci,
              "GATES",report["replicas"][str(rep)]["gates"],
              flush=True)
    assert len(all_case_hashes)==2048
    report["unique_confirm_case_hashes"]=len(all_case_hashes)
    keys=("C_G1_transfer_never_executed_SRX",
          "C_G2_F4_XOR_causality","C_G3_effects_vs_reward",
          "C_G4_directed_vs_uniform")
    report["gate_counts"]={
        k:sum(int(x["gates"][k]) for x in report["replicas"].values())
        for k in keys
    }
    report["status"]=(
        "F10N_CONFIRM_PASS"
        if (report["original_g0_prerequisite"]>=3 and
            all(report["gate_counts"][k]>=3 for k in keys[:3]))
        else "F10N_CONFIRM_FAIL"
    )
    pooled={
        arm:sum(rep["accuracy_and_strata"][arm]["correct"]
                for rep in report["replicas"].values())
        for arm in ARMS
    }
    report["pooled_descriptive"]={
        arm:{"correct":count,"total":2048,"accuracy":count/2048}
        for arm,count in pooled.items()
    }
    print("F10NC_INDEPENDENT_FROZEN_SCIENCE",report["status"],
          report["gate_counts"],report["pooled_descriptive"],flush=True)
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--original-f10n-audit",required=True,type=Path)
    ap.add_argument("--original-f10l-g0",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    result=audit(a.root,a.original_f10n_audit,a.original_f10l_g0)
    a.out.write_text(json.dumps(result,indent=2))
