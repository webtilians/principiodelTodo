"""F10-L independent phase-one audit: TRAIN integrity + DEV primitive gates.

NEVER imports OOD generator, model code, torch or composition routines.
A FAIL of G0 is a scientific outcome, NOT a CI failure and MUST NOT launch
stage two. Keep counterfactual quadruples grouped in all resampling.
"""
from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=("xor_onpolicy","xor_two_actions","sel_onpolicy",
      "sel_uniform_credit","sel_uniform_pair")

def hit(row):
    if row["task"]=="xor":
        return int(row["action"]==row["truth"])
    return int(row["selected"]==row["winner"])

def world_vector(rows):
    groups={}
    for r in rows:
        groups.setdefault(r["base_world"],[]).append(hit(r))
    assert len(groups)==128
    assert all(len(v)==4 for v in groups.values())
    return np.array([np.mean(v) for _,v in sorted(groups.items())],np.float64)

def absolute(rows,rep,offset):
    vals=world_vector(rows)
    rng=np.random.default_rng(237000+rep*100+offset)
    vals_sampled=vals[rng.integers(len(vals),size=(2000,len(vals)))].mean(1)
    return {"accuracy":float(vals.mean()),
            "ci95":[float(x) for x in np.quantile(vals_sampled,[.025,.975])],
            "n_worlds":128,"n_cases":512,"bootstrap":2000}

def contrast(a,b,rep,offset):
    aa={(r["base_world"],r.get("variant",r.get("bit",0)*2+r.get("flip",0))):r for r in a}
    bb={(r["base_world"],r.get("variant",r.get("bit",0)*2+r.get("flip",0))):r for r in b}
    assert set(aa)==set(bb)
    worlds=sorted({k[0] for k in aa})
    assert len(worlds)==128
    vectors=np.array([np.mean([hit(aa[k])-hit(bb[k]) for k in aa if k[0]==w])
                      for w in worlds],np.float64)
    rng=np.random.default_rng(237000+rep*100+offset)
    resampled=vectors[rng.integers(len(vectors),size=(2000,len(vectors)))].mean(1)
    return {"delta":float(vectors.mean()),
            "ci95":[float(x) for x in np.quantile(resampled,[.025,.975])],
            "bootstrap":2000,"unit":"base_world"}

def run(root):
    out={
        "experiment":"F10-L phase one independent prerequisite audit",
        "technical":"F10L_PHASE1_TECHNICAL_COMPLETE",
        "confirm_23300000_not_generated":True,
        "composition_23400000_not_opened":True,
        "replicas":{},
    }
    all_hashes=set()
    for rep in range(4):
        folder=root/str(rep)
        meta=json.loads((folder/"train.json").read_text())
        rows=json.loads((folder/"dev.json").read_text())
        assert meta["status"]=="FROZEN_BEFORE_DEV_AND_ANY_OOD"
        assert meta["replica"]==rep
        assert meta["epochs"]==8 and meta["batch"]==64
        assert meta["lr"]==.003 and meta["entropy"]==.002 and meta["ema"]==.9
        assert meta["train_composition_exposure"]==0
        assert meta["confirm_23300000_unopened"]
        assert meta["train_xor_balance"]=={str(k):1024 for k in range(4)}
        assert set(meta["arms"])==set(ARMS)
        assert len(meta["train_xor_case_hashes"])==4096
        assert len(meta["train_select_case_hashes"])==4096
        train_hashes=set(meta["train_xor_case_hashes"]+meta["train_select_case_hashes"])
        assert len(train_hashes)==8192
        assert not train_hashes&all_hashes
        all_hashes.update(train_hashes)
        initial_x={meta["arms"][a]["initial_weights_sha256"] for a in ARMS[:2]}
        initial_s={meta["arms"][a]["initial_weights_sha256"] for a in ARMS[2:]}
        assert len(initial_x)==len(initial_s)==1
        for arm in ARMS:
            m=meta["arms"][arm]
            assert m["checkpoint"]==arm+".pt"
            assert hashlib.sha256((folder/m["checkpoint"]).read_bytes()).hexdigest()==m["sha256"]
            assert m["executed_environment_episodes"]==65536
            assert m["training_states_seen"]==32768
            assert m["updates"]==512 and m["epochs"]==8
            assert m["loss_supervised_winner_or_answer"] is False
            assert len(m["curves"])==8
            assert m["curves"][-1]["episodes"]==65536
        assert len(rows)==2560
        keys={(r["arm"],r["hash"]) for r in rows}
        assert len(keys)==len(rows)
        hashes={r["hash"] for r in rows}
        assert len(hashes)==1024 and not hashes&all_hashes
        all_hashes.update(hashes)
        expected_sets={
            "xor":set(range(23_200_000+10_000*rep,23_200_000+10_000*rep+512)),
            "select":set(range(23_205_000+10_000*rep,23_205_000+10_000*rep+128)),
        }
        canonical={}
        for arm in ARMS:
            subset=[r for r in rows if r["arm"]==arm]
            assert len(subset)==512
            expected="xor" if arm.startswith("xor") else "select"
            assert all(r["task"]==expected for r in subset)
            assert {r["seed"] for r in subset}==expected_sets[expected]
            d={r["hash"]:(
                r["seed"],r["base_world"],r.get("truth"),r.get("bit"),
                r.get("flip"),r.get("winner"),r.get("variant"),r.get("n")
            ) for r in subset}
            if expected not in canonical:
                canonical[expected]=d
            else:
                assert canonical[expected]==d
            for case in subset:
                if case["task"]=="xor":
                    assert case["truth"]==case["bit"]^case["flip"]
                    assert case["action"] in (0,1)
                else:
                    assert case["selected"] in range(12)
                    assert case["winner"] in range(12)
                    assert case["n"] in (5,7)
                    assert case["variant"] in (
                        "U_before","U_after","S_before","S_after")
        metrics={a:absolute([r for r in rows if r["arm"]==a],rep,i)
                 for i,a in enumerate(ARMS)}
        xor_a=[r for r in rows if r["arm"]=="xor_onpolicy"]
        xor_b=[r for r in rows if r["arm"]=="xor_two_actions"]
        sel_b=[r for r in rows if r["arm"]=="sel_uniform_credit"]
        sel_c=[r for r in rows if r["arm"]=="sel_uniform_pair"]
        contrasts={
            "X_paired_minus_onpolicy":contrast(xor_b,xor_a,rep,10),
            "S_paired_minus_uniform_credit":contrast(sel_c,sel_b,rep,11),
        }
        truth_table={}
        for b in (0,1):
            for flip in (0,1):
                part=[r for r in xor_b if r["bit"]==b and r["flip"]==flip]
                assert len(part)==128
                truth_table[f"{b}_{flip}"]=sum(hit(r) for r in part)/len(part)
        x=metrics["xor_two_actions"]
        s=metrics["sel_uniform_pair"]
        g0=(x["accuracy"]>=.95 and x["ci95"][0]>.90 and
            s["accuracy"]>=.85 and s["ci95"][0]>.75)
        gx=contrasts["X_paired_minus_onpolicy"]
        gs=contrasts["S_paired_minus_uniform_credit"]
        out["replicas"][str(rep)]={
            "metrics":metrics,"contrasts":contrasts,
            "xor_two_actions_truth_table":truth_table,
            "gates":{
                "G0_both_primitives":g0,
                "G1_XOR_pair_advantage":gx["delta"]>=.05 and gx["ci95"][0]>0,
                "G2_SELECTOR_credit_advantage":gs["delta"]>=.05 and gs["ci95"][0]>0,
            },
            "model_sha256":{a:meta["arms"][a]["sha256"] for a in ARMS},
            "parameters":{a:meta["arms"][a]["parameters"] for a in ARMS},
            "train_curves":{a:meta["arms"][a]["curves"] for a in ARMS},
        }
        print("F10L_REPLICA_PRIMITIVES",rep,
              "XOR",{a:round(metrics[a]["accuracy"],4) for a in ARMS[:2]},
              "SELECT",{a:round(metrics[a]["accuracy"],4) for a in ARMS[2:]},
              "XOR_TRUTH_TABLE",truth_table,"G0",g0,flush=True)
    assert len(all_hashes)==4*(8192+1024)
    out["unique_case_identity_hashes"]=len(all_hashes)
    counts={
        k:sum(int(r["gates"][k]) for r in out["replicas"].values())
        for k in ("G0_both_primitives","G1_XOR_pair_advantage",
                  "G2_SELECTOR_credit_advantage")
    }
    out["gate_counts"]=counts
    out["eligible_for_composition"]=counts["G0_both_primitives"]>=3
    out["scientific_status"]=(
        "F10L_PRIMITIVES_PASS_COMPOSITION_AUTHORIZED"
        if out["eligible_for_composition"]
        else "F10L_PRIMITIVE_COMPETENCE_FAIL"
    )
    print("F10L_PHASE1_FINAL",out["scientific_status"],counts,flush=True)
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    report=run(a.root)
    a.out.write_text(json.dumps(report,indent=2))
