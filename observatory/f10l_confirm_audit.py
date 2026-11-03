"""F10-L-C independent science auditor: NO torch/models/generator imported.

One preregistered confirmation split, immutable original SHA checks,
G3 and G4 copied with *unchanged* thresholds from original F10-L.
World is bootstrap unit; four counterfactuals must never be separated.
"""
from __future__ import annotations

import argparse,json,hashlib
from pathlib import Path
import numpy as np

ARMS=("xor_onpolicy","xor_two_actions",
      "sel_onpolicy","sel_uniform_credit","sel_uniform_pair")
MODES=("composed","shuffle","no_flip","oracle_slot",
       "xor_onpolicy_selector_paired","xor_paired_selector_onpolicy",
       "symbolic_xor_learned_selector","symbolic_oracle")
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCES=("iid","signed_source0","signed_source1","alternating")

def good(row,mode="composed"):
    return int(row["predictions"][mode]==row["truth"])

def accuracy(rows,mode="composed"):
    assert rows
    return sum(good(r,mode) for r in rows)/len(rows)

def bootstrap(rows,rep,left="composed",right=None,offset=0):
    by_world={}
    for r in rows:
        by_world.setdefault(r["base_world"],[]).append(r)
    assert len(by_world)==128
    assert all(len(xs)==4 for xs in by_world.values())
    deltas=np.array([
        sum(good(r,left)-(good(r,right) if right else 0) for r in xs)/4.
        for _,xs in sorted(by_world.items())],dtype=np.float64)
    rng=np.random.default_rng(241000+100*rep+offset)
    means=deltas[rng.integers(0,len(deltas),size=(2000,len(deltas)))].mean(1)
    return {"estimate":float(deltas.mean()),
            "ci95":[float(x) for x in np.quantile(means,[.025,.975])],
            "unit":"base_world","n_worlds":128,"resamples":2000}

def audit(root,g0_path):
    gate=json.loads(g0_path.read_text())
    assert gate["technical"]=="F10L_PHASE1_TECHNICAL_COMPLETE"
    assert gate["eligible_for_composition"]
    assert gate["gate_counts"]["G0_both_primitives"]>=3
    assert gate["composition_23400000_not_opened"]
    output={
        "experiment":"INFINITO F10-L-C independent confirmation on reserved seeds 23300000+",
        "technical":"F10LC_TECHNICAL_COMPLETE",
        "original_run_id":41685499939,
        "original_code_commit":"b5c32cbfb50a24414607e3a24e30e139ca92fede",
        "original_g0_gate":gate["gate_counts"],
        "model_training_during_confirmation":False,
        "no_post_ood_model_selection":True,
        "replicas":{},
    }
    all_hashes=set()
    all_rows=[]
    for rep in range(4):
        meta=json.loads((root/(str(rep)+".json")).read_text())
        assert meta["status"]=="CONFIRMATION_WORLDS_OPENED_ONCE_AND_EVALUATED"
        assert meta["replica"]==rep
        assert meta["original_model_run_id"]==41685499939
        assert meta["original_code_commit"]==output["original_code_commit"]
        assert meta["using_unmodified_original_stage2_run"]
        assert meta["old_ood_regression_passed_bit_exact"]
        assert meta["training_updates_during_confirmation"]==0
        assert meta["confirmation_namespace"]==23_300_000
        assert meta["worlds"]==128 and meta["n_predictions"]==512
        assert set(meta["checkpoint_sha"])==set(ARMS)
        assert meta["checkpoint_sha"]==gate["replicas"][str(rep)]["model_sha256"]
        rows=meta["rows"]
        assert len(rows)==512
        hashes={r["hash"] for r in rows}
        assert len(hashes)==512 and not (hashes&all_hashes)
        all_hashes.update(hashes)
        assert {r["seed"] for r in rows}=={
            23_300_000+10_000*rep+j for j in range(128)}
        for world_id in range(128):
            group=[r for r in rows if r["base_world"]==world_id]
            assert len(group)==4
            variants={r["variant"]:r for r in group}
            assert set(variants)==set(VARIANTS)
            assert all(r["seed"]==23_300_000+10_000*rep+world_id for r in group)
            assert all(r["n"]==(9 if world_id%2==0 else 11) for r in group)
            assert all(r["source_mode"]==SOURCES[(world_id//2)%4] for r in group)
            assert variants["U_before"]["truth"]==variants["U_after"]["truth"]
            assert variants["U_after"]["truth"]==variants["S_before"]["truth"]
            assert variants["S_after"]["truth"]!=variants["U_after"]["truth"]
        for row in rows:
            assert row["replica"]==rep
            assert row["signed_count"]>=4
            assert row["winner"] in range(12)
            assert row["selected"] in range(12)
            assert row["selected_onpolicy"] in range(12)
            assert row["truth"] in (0,1)
            assert set(row["predictions"])==set(MODES)
            assert all(p in (0,1) for p in row["predictions"].values())
            assert good(row,"symbolic_oracle")==1
        stat={k:accuracy(rows,k) for k in MODES}
        stat["selection"]=sum(r["selected"]==r["winner"] for r in rows)/len(rows)
        stat["selection_onpolicy"]=sum(
            r["selected_onpolicy"]==r["winner"] for r in rows)/len(rows)
        stat["by_n"]={str(n):accuracy([r for r in rows if r["n"]==n])
                      for n in (9,11)}
        stat["by_source"]={s:accuracy([r for r in rows if r["source_mode"]==s])
                          for s in SOURCES}
        stat["by_variant"]={v:accuracy([r for r in rows if r["variant"]==v])
                           for v in VARIANTS}
        stat["constant_0_baseline"]=sum(r["truth"]==0 for r in rows)/len(rows)
        stat["constant_1_baseline"]=sum(r["truth"]==1 for r in rows)/len(rows)
        stat["symbolic_oracle"]=accuracy(rows,"symbolic_oracle")
        effects={
            "composition":bootstrap(rows,rep,offset=0),
            "vs_shuffle":bootstrap(rows,rep,right="shuffle",offset=1),
            "vs_no_flip":bootstrap(rows,rep,right="no_flip",offset=2),
        }
        # Identical scientific thresholds and 3/4-replica criteria as F10-L.
        confirm_g3=(stat["composed"]>=.80 and
            effects["composition"]["ci95"][0]>.70 and
            min(stat["by_n"].values())>=.70 and
            min(stat["by_source"].values())>=.70 and
            min(stat["by_variant"].values())>=.70)
        confirm_g4=all(
            effects[k]["estimate"]>=.20 and effects[k]["ci95"][0]>0
            for k in ("vs_shuffle","vs_no_flip"))
        output["replicas"][str(rep)]={
            "metrics":stat,"bootstrap":effects,
            "model_sha256":meta["checkpoint_sha"],
            "gates":{"C_G3_composition":confirm_g3,"C_G4_causal_use":confirm_g4},
            "original_ood_regression_verified":True,
        }
        print("F10LC_CONFIRM_REPLICA",rep,
              "accuracy",round(stat["composed"],6),
              "select",round(stat["selection"],6),
              "delta_shuffle",round(effects["vs_shuffle"]["estimate"],6),
              "delta_noflip",round(effects["vs_no_flip"]["estimate"],6),
              "G3",confirm_g3,"G4",confirm_g4,flush=True)
        all_rows.extend(rows)
    assert len(all_hashes)==2048
    output["unique_confirm_case_hashes"]=len(all_hashes)
    counts={k:sum(int(r["gates"][k]) for r in output["replicas"].values())
            for k in ("C_G3_composition","C_G4_causal_use")}
    output["gate_counts"]=counts
    output["confirm_pass"]=all(n>=3 for n in counts.values())
    output["status"]="F10L_CONFIRM_PASS" if output["confirm_pass"] else "F10L_CONFIRM_FAIL"
    output["pooled_descriptive"]={
        "correct":sum(good(r) for r in all_rows),
        "cases":len(all_rows),
        "accuracy":accuracy(all_rows),
        "NOTE":"4 variants/world; intervals must remain grouped by world and replica",
    }
    print("F10LC_CONFIRM_FINAL_FIXED_GATE",output["status"],counts,flush=True)
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--gate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    result=audit(args.root,args.gate)
    args.out.write_text(json.dumps(result,indent=2))
