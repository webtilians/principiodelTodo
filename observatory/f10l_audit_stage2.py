"""F10-L final independent audit. Reads ONLY frozen DEV-gate and OOD JSON.

The auditor does NOT import training, generator, model, torch or stage2.
Four counterfactual variants from one world are ONE bootstrap unit.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np

MODES=("composed","shuffle","no_flip","oracle_slot",
       "xor_onpolicy_selector_paired","xor_paired_selector_onpolicy",
       "symbolic_xor_learned_selector","symbolic_oracle")
SOURCE_MODES=("iid","signed_source0","signed_source1","alternating")
VARIANTS=("U_before","U_after","S_before","S_after")

def good(r,mode="composed"):
    return int(r["predictions"][mode]==r["truth"])

def score(rows,mode="composed"):
    return sum(good(r,mode) for r in rows)/len(rows)

def sample_worlds(rows,rep,left="composed",right=None,offset=0):
    groups={}
    for r in rows:
        groups.setdefault(r["base_world"],[]).append(r)
    assert len(groups)==128
    assert all(len(group)==4 for group in groups.values())
    vals=np.array([
        sum(good(r,left)-(good(r,right) if right else 0) for r in group)/4.
        for _,group in sorted(groups.items())
    ],np.float64)
    rng=np.random.default_rng(239000+rep*100+offset)
    sampled=vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(1)
    return {
        "estimate":float(vals.mean()),
        "ci95":[float(z) for z in np.quantile(sampled,(.025,.975))],
        "unit":"base_world","n_worlds":128,"resamples":2000,
    }

def audit(dev_gate_path,ood_root):
    stage1=json.loads(dev_gate_path.read_text())
    assert stage1["technical"]=="F10L_PHASE1_TECHNICAL_COMPLETE"
    assert stage1["eligible_for_composition"] is True
    assert stage1["gate_counts"]["G0_both_primitives"]>=3
    assert stage1["composition_23400000_not_opened"]
    out={
        "experiment":"INFINITO F10-L zero-shot composition after independent G0 unlock",
        "technical":"F10L_STAGE2_TECHNICAL_COMPLETE",
        "prerequisite_g0":stage1["gate_counts"],
        "confirm_23300000_unopened":True,
        "replicas":{},
    }
    all_hashes=set()
    for rep in range(4):
        meta=json.loads((ood_root/(str(rep)+".json")).read_text())
        assert meta["replica"]==rep
        assert meta["status"]=="EVALUATED_ONLY_AFTER_G0"
        assert meta["no_optimizer_or_training_in_stage2"]
        assert meta["confirm_23300000_unopened"]
        assert meta["checkpoint_sha"]==stage1["replicas"][str(rep)]["model_sha256"]
        rows=meta["rows"]
        assert len(rows)==512
        hashes={r["hash"] for r in rows}
        assert len(hashes)==512 and not hashes&all_hashes
        all_hashes.update(hashes)
        assert {r["seed"] for r in rows}==set(
            range(23_400_000+10_000*rep,23_400_000+10_000*rep+128))
        for base in range(128):
            group=[r for r in rows if r["base_world"]==base]
            assert len(group)==4
            by={r["variant"]:r for r in group}
            assert set(by)==set(VARIANTS)
            assert len({r["winner"] for r in group})>=1
            assert (by["U_before"]["truth"]==by["U_after"]["truth"]==
                    by["S_before"]["truth"])
            assert by["S_after"]["truth"]!=by["U_after"]["truth"]
        for row in rows:
            assert row["replica"]==rep
            assert row["n"] in (9,11)
            assert row["source_mode"] in SOURCE_MODES
            assert row["variant"] in VARIANTS
            assert row["signed_count"]>=4
            assert row["truth"] in (0,1)
            assert row["winner"] in range(12)
            assert row["selected"] in range(12)
            assert row["selected_onpolicy"] in range(12)
            assert set(row["predictions"])==set(MODES)
            assert all(v in (0,1) for v in row["predictions"].values())
            assert good(row,"symbolic_oracle")==1
        accuracy={k:score(rows,k) for k in MODES}
        accuracy["selection"]=sum(r["selected"]==r["winner"] for r in rows)/len(rows)
        accuracy["selection_onpolicy"]=sum(
            r["selected_onpolicy"]==r["winner"] for r in rows)/len(rows)
        accuracy["by_n"]={str(k):score([r for r in rows if r["n"]==k])
                          for k in (9,11)}
        accuracy["by_source"]={s:score([r for r in rows if r["source_mode"]==s])
                              for s in SOURCE_MODES}
        accuracy["by_variant"]={v:score([r for r in rows if r["variant"]==v])
                               for v in VARIANTS}
        bootstrap={
            "composed":sample_worlds(rows,rep,offset=0),
            "vs_shuffle":sample_worlds(rows,rep,right="shuffle",offset=1),
            "vs_no_flip":sample_worlds(rows,rep,right="no_flip",offset=2),
        }
        g3=(accuracy["composed"]>=.80 and
            bootstrap["composed"]["ci95"][0]>.70 and
            min(accuracy["by_n"].values())>=.70 and
            min(accuracy["by_source"].values())>=.70 and
            min(accuracy["by_variant"].values())>=.70)
        g4=all(bootstrap[k]["estimate"]>=.20 and
               bootstrap[k]["ci95"][0]>0
               for k in ("vs_shuffle","vs_no_flip"))
        out["replicas"][str(rep)]={
            "metrics":accuracy,"bootstrap":bootstrap,
            "gates":{"G3_zero_shot_composition":g3,"G4_causal_reuse":g4},
        }
        print("F10L_REPLICA_OOD",rep,
              "composed",round(accuracy["composed"],5),
              "selection",round(accuracy["selection"],5),
              "vs_shuffle",round(bootstrap["vs_shuffle"]["estimate"],5),
              "vs_no_flip",round(bootstrap["vs_no_flip"]["estimate"],5),
              "g3",g3,"g4",g4,flush=True)
    assert len(all_hashes)==4*512
    out["unique_ood_case_ids"]=len(all_hashes)
    counts={
        k:sum(bool(v["gates"][k]) for v in out["replicas"].values())
        for k in ("G3_zero_shot_composition","G4_causal_reuse")
    }
    out["gate_counts"]=counts
    out["status"]=(
        "F10L_COMPOSITION_PASS" if all(v>=3 for v in counts.values())
        else "F10L_COMPOSITION_FAIL"
    )
    print("F10L_GLOBAL_SCIENCE",out["status"],counts,flush=True)
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--stage1",type=Path,required=True)
    p.add_argument("--ood-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.stage1,a.ood_root)
    a.out.write_text(json.dumps(result,indent=2))
