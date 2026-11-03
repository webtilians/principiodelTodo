"""F10-M independent frozen-controller scientific auditor: no torch / model imports.

Uses ORIGINAL SHA manifests from training jobs, counts actually executed
program rollouts and groups 4 counterfactual variants per base world.
Gates are copied exactly from prospective preregistration.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=("global_bandit","context_mlp","factorized","paired_factorized")
MODES=("normal","shuffle","x_disabled","public_randomized")
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCES=("iid","signed_source0","signed_source1","alternating")
CONTEXTS=(("PUBLIC",0),("PUBLIC",1),("MEMORY",0))

def success(r,mode="normal"):
    return int(r["outcomes"][mode]["success"])

def rate(rows,mode="normal"):
    assert rows
    return sum(success(r,mode) for r in rows)/len(rows)

def contrast(left,right,rep,kind,offset=0):
    a={(r["seed"],r["variant"]):r for r in left}
    b={(r["seed"],r["variant"]):r for r in right}
    assert a.keys()==b.keys()
    worlds=sorted({s for s,_ in a})
    assert len(worlds)==128
    deltas=np.array([
        sum(success(a[(seed,v)],"normal")-
            success(b[(seed,v)],"normal" if kind=="arm" else kind)
            for v in VARIANTS)/4. for seed in worlds],np.float64)
    rng=np.random.default_rng(257000+100*rep+offset)
    bs=deltas[rng.integers(len(deltas),size=(2000,len(deltas)))].mean(axis=1)
    return {"estimate":float(deltas.mean()),
            "ci95":[float(x) for x in np.quantile(bs,(.025,.975))],
            "unit":"base_world","worlds":128,"bootstrap":2000}

def accuracy_ci(rows,rep):
    groups={}
    for r in rows:
        groups.setdefault(r["seed"],[]).append(success(r))
    assert len(groups)==128 and all(len(v)==4 for v in groups.values())
    vals=np.array([sum(v)/4. for _,v in sorted(groups.items())],np.float64)
    rng=np.random.default_rng(257000+100*rep)
    bs=vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(axis=1)
    return [float(x) for x in np.quantile(bs,(.025,.975))]

def stats(rows):
    if not rows:
        return None
    programs={}
    for r in rows:
        programs[r["program"]]=programs.get(r["program"],0)+1
    return {
        "cases":len(rows),"worlds":len({r["seed"] for r in rows}),
        "accuracy":rate(rows),
        "mean_operational_return":sum(r["outcomes"]["normal"]["return"]
                                      for r in rows)/len(rows),
        "mean_operations":sum(r["outcomes"]["normal"]["operations"]
                              for r in rows)/len(rows),
        "mean_invalid":sum(r["outcomes"]["normal"]["invalid"]
                           for r in rows)/len(rows),
        "read_rate":sum(r["outcomes"]["normal"]["read"] for r in rows)/len(rows),
        "normal_vs_shuffle":rate(rows)-rate(rows,"shuffle"),
        "normal_vs_x_disabled":rate(rows)-rate(rows,"x_disabled"),
        "normal_vs_public_randomized":rate(rows)-rate(rows,"public_randomized"),
        "program_counts":dict(sorted(programs.items(),
                                     key=lambda item:(-item[1],item[0]))),
        "by_n":{str(n):rate([r for r in rows if r["n"]==n])
                for n in sorted({r["n"] for r in rows})},
        "by_source":{source:rate([r for r in rows if r["source_mode"]==source])
                     for source in sorted({r["source_mode"] for r in rows})},
        "by_variant":{v:rate([r for r in rows if r["variant"]==v])
                      for v in VARIANTS},
    }

def audit(root):
    out={
        "experiment":"INFINITO F10-M autonomous choice of operation sequence",
        "technical":"F10M_TECHNICAL_COMPLETE",
        "preregistered_gate_version":"2026-11-03",
        "original_f10l_run":41685499939,
        "original_f10l_original_commit":"b5c32cbfb50a24414607e3a24e30e139ca92fede",
        "confirm_25300000_unopened":True,
        "replicas":{},
    }
    all_ids=set()
    for rep in range(4):
        folder=root/str(rep)
        meta=json.loads((folder/"training.json").read_text())
        rows=json.loads((folder/"rows.json").read_text())
        assert meta["status"]=="FROZEN_BEFORE_DEV_AND_OOD"
        assert meta["replica"]==rep
        assert meta["confirm_25300000_sealed"]
        assert meta["no_pretrained_model_training"]
        assert meta["no_optimal_program_labels"]
        assert meta["no_unexecuted_program_reward_queries"]
        assert meta["unseen_memory_flip1_never_train"]
        assert meta["epochs"]==16 and meta["batch"]==64
        assert meta["lr"]==.01 and meta["epsilon"]==.2
        assert meta["entropy"]==.002 and meta["ema"]==.9
        assert meta["program_count"]==85
        assert len(meta["program_list"])==85
        assert len(set(meta["program_list"]))==85
        assert meta["train_context_counts"]=={
            "PUBLIC_0":1024,"PUBLIC_1":1024,"MEMORY_0":1024}
        assert meta["original_skills"]["original_run_id"]==41685499939
        assert meta["original_skills"]["original_code_sha"]==out["original_f10l_original_commit"]
        assert meta["original_skills"]["pretrained_xor_bit_flip1_lut"]==[1,0]
        assert set(meta["arms"])==set(ARMS)
        assert len(meta["train_case_hashes"])==3072
        train_hashes=set(meta["train_case_hashes"])
        assert len(train_hashes)==3072
        assert not (train_hashes&all_ids)
        all_ids.update(train_hashes)
        assert len(rows)==4*(768+512)
        assert len({(r["arm"],r["case_sha"]) for r in rows})==len(rows)
        parameters={}
        for arm in ARMS:
            a=meta["arms"][arm]
            assert a["checkpoint"]==arm+".pt"
            assert hashlib.sha256((folder/a["checkpoint"]).read_bytes()).hexdigest()==a["sha256"]
            assert a["epochs"]==16 and a["updates"]==768
            assert a["executed_environment_episodes"]==98304
            assert a["train_presentations"]==16*3072
            assert a["no_supervised_labels"]
            assert len(a["curves"])==16
            assert a["curves"][-1]["executed_episodes"]==98304
            parameters[arm]=a["parameters"]
        assert parameters["factorized"]==parameters["paired_factorized"]
        assert meta["arms"]["factorized"]["initial_trainable_sha"]==(
            meta["arms"]["paired_factorized"]["initial_trainable_sha"])
        assert all(len(meta["arms"][a]["curves"])==16 for a in ARMS)
        canonical={}
        for arm in ARMS:
            d={(r["split"],r["seed"],r["variant"]):
                (r["case_sha"],r["truth"],r["task"],r["flip"],
                 r["public_bit"],r["winner"],r["frozen_selected"],
                 r["n"],r["source_mode"])
               for r in rows if r["arm"]==arm}
            assert len(d)==1280
            if not canonical:canonical=d
            else:assert d==canonical
        for split in ("dev","ood"):
            block=[r for r in rows if r["split"]==split and r["arm"]==ARMS[0]]
            assert len(block)==(768 if split=="dev" else 512)
            hashes={r["case_sha"] for r in block}
            assert len(hashes)==len(block)
            assert not hashes&all_ids
            all_ids.update(hashes)
            if split=="dev":
                expected={25_200_000+10000*rep+1000*t+j
                          for t in range(3) for j in range(64)}
                assert {(r["task"],r["flip"]) for r in block}==set(CONTEXTS)
                assert {r["seed"] for r in block}==expected
                assert {r["source_mode"] for r in block}=={"iid"}
                assert {r["n"] for r in block}=={5,7}
            else:
                assert all(r["task"]=="MEMORY" and r["flip"]==1 for r in block)
                assert {r["seed"] for r in block}=={
                    25_400_000+10000*rep+j for j in range(128)}
                assert {r["n"] for r in block}=={9,11}
                for source in SOURCES:
                    for n in (9,11):
                        count=len({r["seed"] for r in block
                                   if r["source_mode"]==source and r["n"]==n})
                        assert count==16,(source,n,count)
            for seed in {r["seed"] for r in block}:
                family=[r for r in block if r["seed"]==seed]
                assert len(family)==4
                variants={r["variant"]:r for r in family}
                assert set(variants)==set(VARIANTS)
                assert len({r["public_bit"] for r in family})==1
                assert len({r["task"] for r in family})==1
                assert len({r["flip"] for r in family})==1
                if family[0]["task"]=="MEMORY":
                    assert variants["U_before"]["truth"]==variants["U_after"]["truth"]
                    assert variants["U_after"]["truth"]==variants["S_before"]["truth"]
                    assert variants["S_after"]["truth"]!=variants["U_after"]["truth"]
                else:
                    assert len({r["truth"] for r in family})==1
        for r in rows:
            assert r["replica"]==rep
            assert r["program_id"] in range(85)
            assert r["program"]==meta["program_list"][r["program_id"]]
            assert set(r["outcomes"])==set(MODES)
            assert r["truth"] in (0,1)
            assert r["winner"] in range(12)
            assert r["frozen_selected"] in range(12)
            for mode,x in r["outcomes"].items():
                assert x["answer"] in (0,1) and x["success"] in (0,1)
                assert x["success"]==int(x["answer"]==r["truth"])
                assert x["operations"] in range(4)
                assert x["invalid"] in range(4)
                assert x["return"]>=-1.34 and x["return"]<=1
                assert x["read"] in (0,1)
        split_stats={}
        for split in ("dev","ood"):
            split_stats[split]={}
            for arm in ARMS:
                r=[x for x in rows if x["split"]==split and x["arm"]==arm]
                if split=="dev":
                    split_stats[split][arm]={
                        f"{kind}_{flip}":stats([x for x in r
                                               if x["task"]==kind and x["flip"]==flip])
                        for kind,flip in CONTEXTS}
                else:
                    split_stats[split][arm]=stats(r)
        c_ood=[r for r in rows if r["arm"]=="factorized" and r["split"]=="ood"]
        a_ood=[r for r in rows if r["arm"]=="global_bandit" and r["split"]=="ood"]
        d_ood=[r for r in rows if r["arm"]=="paired_factorized" and r["split"]=="ood"]
        contrasts={
            "C_vs_global_A":contrast(c_ood,a_ood,rep,"arm",offset=3),
            "D_vs_C":contrast(d_ood,c_ood,rep,"arm",offset=4),
            "C_normal_vs_shuffle":contrast(c_ood,c_ood,rep,"shuffle",offset=1),
            "C_normal_vs_x_disabled":contrast(c_ood,c_ood,rep,"x_disabled",offset=2),
        }
        ci=accuracy_ci(c_ood,rep)
        cv=split_stats["ood"]["factorized"]
        dev=split_stats["dev"]["factorized"]
        g0=all(dev[f"{kind}_{flip}"]["accuracy"]>=.8
               for kind,flip in CONTEXTS)
        g1=(cv["accuracy"]>=.80 and ci[0]>.70 and
            min(cv["by_n"].values())>=.70 and
            min(cv["by_source"].values())>=.70 and
            min(cv["by_variant"].values())>=.70)
        g2=all(contrasts[k]["estimate"]>=.20 and
               contrasts[k]["ci95"][0]>0
               for k in ("C_normal_vs_shuffle","C_normal_vs_x_disabled"))
        majorities=[next(iter(dev[f"{kind}_{flip}"]["program_counts"]))
                    for kind,flip in CONTEXTS]
        adapt=len(set(majorities))>=2
        diff=contrasts["C_vs_global_A"]
        g3=(diff["estimate"]>=.10 and diff["ci95"][0]>0 and adapt)
        delta=contrasts["D_vs_C"]
        g4=(delta["estimate"]>=.05 and delta["ci95"][0]>0)
        out["replicas"][str(rep)]={
            "metrics":split_stats,
            "bootstrap":contrasts,
            "composition_accuracy_ci95":ci,
            "program_majority_by_seen_context":majorities,
            "gates":{
                "G0_seen_tasks":g0,
                "G1_unseen_program":g1,
                "G2_memory_xor_causality":g2,
                "G3_contextual_vs_global":g3,
                "G4_paired_advantage":g4,
            },
            "controller_sha256":{a:meta["arms"][a]["sha256"] for a in ARMS},
            "original_frozen_sha256":meta["original_skills"]["all_5_original_checkpoint_sha"],
            "parameters":parameters,
            "learning_curves":{a:meta["arms"][a]["curves"] for a in ARMS},
        }
        print("F10M_REPLICA",rep,
              "DEV",{k:round(v["accuracy"],4) for k,v in dev.items()},
              "OOD",{a:round(split_stats["ood"][a]["accuracy"],4) for a in ARMS},
              "C_CI",ci,
              "C_TOP",next(iter(cv["program_counts"])),
              "GATES",out["replicas"][str(rep)]["gates"],flush=True)
    assert len(all_ids)==4*(3072+768+512)
    out["unique_case_identity_hashes"]=len(all_ids)
    counts={k:sum(bool(r["gates"][k]) for r in out["replicas"].values())
            for k in ("G0_seen_tasks","G1_unseen_program",
                      "G2_memory_xor_causality","G3_contextual_vs_global",
                      "G4_paired_advantage")}
    out["gate_counts"]=counts
    out["scientific_gates"]={k:v>=3 for k,v in counts.items()}
    out["status"]=(
        "F10M_PROGRAM_DISCOVERY_PASS"
        if all(out["scientific_gates"][g] for g in
             ("G0_seen_tasks","G1_unseen_program",
              "G2_memory_xor_causality","G3_contextual_vs_global"))
        else "F10M_PROGRAM_DISCOVERY_FAIL")
    print("F10M_FINAL_ORIGINAL_SCIENCE",out["status"],counts,flush=True)
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.root)
    a.out.write_text(json.dumps(result,indent=2))
