"""F10-H independent fixed-gate audit of all saved worlds/rows/checkpoints.

Does not import torch, train or infer, and does not open confirmation worlds.
"""
from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=("joint","selection_only","interpretation_only","both","gru_only")
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCE_MODES=("iid","signed_source_0","signed_source_1","alternating_storage_source")
MODES=("read","skip","shuffle","erase_sign","erase_time","oracle_patch")

def success(row,mode="read"):
    return int(row["predictions"][mode]==row["truth"])

def accuracy(rows,mode="read"):
    return sum(success(r,mode) for r in rows)/len(rows)

def summary(rows,has_selector=True):
    res={
        "cases":len(rows),
        "read":accuracy(rows),
        "skip":accuracy(rows,"skip"),
        "shuffle":accuracy(rows,"shuffle"),
        "erase_sign":accuracy(rows,"erase_sign"),
        "erase_time":accuracy(rows,"erase_time"),
        "oracle_patch":accuracy(rows,"oracle_patch"),
        "by_variant":{v:accuracy([r for r in rows if r["variant"]==v]) for v in VARIANTS},
        "by_k":{str(k):accuracy([r for r in rows if r["k"]==k])
                for k in sorted({r["k"] for r in rows})},
        "by_source":{m:accuracy([r for r in rows if r["source_mode"]==m])
                    for m in sorted({r["source_mode"] for r in rows})},
        "by_signed_count":{str(k):accuracy([r for r in rows if r["signed_count"]==k])
                           for k in sorted({r["signed_count"] for r in rows})},
    }
    seeds={r["seed"] for r in rows}
    assert len(rows)==4*len(seeds)
    res["all_four"]=sum(all(
        success(r) for r in rows if r["seed"]==s) for s in seeds)/len(seeds)
    if has_selector:
        res["selection"]=sum(r["selected"]==r["winner"] for r in rows)/len(rows)
        res["winner_weight"]=sum(r["winner_weight"] for r in rows)/len(rows)
        res["selection_vs_decision"]={
            f"selection_{a}_decision_{b}":sum(
                (r["selected"]==r["winner"])==a and bool(success(r))==b for r in rows)
            for a in (False,True) for b in (False,True)
        }
    else:
        res["selection"]=None
        res["winner_weight"]=None
    res["baselines"]={k:sum(r["baselines"][k]==r["truth"] for r in rows)/len(rows)
                      for k in rows[0]["baselines"]}
    return res

def bootstrap(a,b,metric_a,metric_b,replica):
    """Matched world bootstrap; four counterfactuals per independent seed."""
    key=lambda r:(r["seed"],r["variant"])
    A={key(r):r for r in a};B={key(r):r for r in b}
    assert A.keys()==B.keys()
    seeds=sorted({k[0] for k in A})
    assert len(seeds)==128
    def get(r,metric):
        if metric=="selection":return int(r["selected"]==r["winner"])
        return success(r,metric)
    deltas=np.array([sum(
        get(A[(seed,v)],metric_a)-get(B[(seed,v)],metric_b)
        for v in VARIANTS)/4 for seed in seeds])
    rng=np.random.default_rng(197000+replica)
    samples=deltas[rng.integers(0,len(deltas),size=(2000,len(deltas)))].mean(1)
    return {
        "delta":float(deltas.mean()),
        "ci95":[float(x) for x in np.quantile(samples,[.025,.975])],
        "worlds":len(seeds),"resamples":2000,
    }

def verify(root):
    result={
      "experiment":"INFINITO F10-H fixed four-way supervision factorial",
      "technical":"F10H_TECHNICAL_COMPLETE",
      "confirm_19300000_opened":False,
      "old_training_used":False,
      "replicas":{},
    }
    global_hashes=set()
    aggregated=[]
    for rep in range(4):
        directory=root/str(rep)
        train=json.loads((directory/"training.json").read_text())
        rows=json.loads((directory/"rows.json").read_text())
        assert train["replica"]==rep
        assert train["stage"]=="TRAIN_COMPLETED_BEFORE_DEV_OR_OOD"
        assert train["preregistered_train_worlds"]==512
        assert train["train_cases"]==2048 and train["epochs"]==20
        assert train["batch"]==64 and train["lr"]==.002
        assert train["no_previous_checkpoint_or_training_examples"]
        assert not train["confirm_19300000_opened"]
        assert set(train["arms"])==set(ARMS)
        assert len(train["train_hashes"])==2048
        hashes=set(train["train_hashes"])
        assert len(hashes)==2048
        assert not hashes&global_hashes
        global_hashes|=hashes
        for arm in ARMS:
            meta=train["arms"][arm]
            assert meta["steps"]==640 and meta["examples"]==40960
            assert len(meta["losses"])==20
            assert meta["checkpoint"]==arm+".pt"
            assert hashlib.sha256((directory/meta["checkpoint"]).read_bytes()).hexdigest()==meta["sha256"]
            assert meta["decision_supervision"]
            assert meta["selection_supervision"]==(arm in ("selection_only","both"))
            assert meta["interpretation_supervision"]==(arm in ("interpretation_only","both"))
        params=train["arms"]["joint"]["parameters"]
        assert all(train["arms"][arm]["parameters"]==params for arm in ARMS[:-1])
        assert abs(train["arms"]["gru_only"]["parameters"]-params)/params<=.05
        assert len(rows)==5*2*128*4
        assert len({(r["split"],r["arm"],r["seed"],r["variant"]) for r in rows})==len(rows)
        # Every control sees IDENTICAL public histories/labels/winners and
        # all four counterfactual variants for each world seed.
        reference={}
        for arm in ARMS:
            armrows=[r for r in rows if r["arm"]==arm]
            observed={(r["split"],r["seed"],r["variant"]):
                 (r["hash"],r["truth"],r["winner"],r["k"],
                  r["signed_count"],r["signed_value_changes"],
                  r["source_mode"],r["baselines"]) for r in armrows}
            assert len(observed)==1024
            if arm==ARMS[0]:reference=observed
            else:assert reference==observed
        for split,namespace,n in (("dev",19_200_000,128),("ood",19_400_000,128)):
            control=[r for r in rows if r["split"]==split and r["arm"]=="joint"]
            expected=set(range(namespace+10000*rep,namespace+10000*rep+n))
            assert {r["seed"] for r in control}==expected
            these_hashes={r["hash"] for r in control}
            assert len(these_hashes)==4*n and not these_hashes&global_hashes
            global_hashes|=these_hashes
            for seed in expected:
                wr=[r for r in control if r["seed"]==seed]
                assert {r["variant"] for r in wr}==set(VARIANTS)
                assert len({r["source_mode"] for r in wr})==1
                assert len({r["k"] for r in wr})==1
                assert len({r["signed_count"] for r in wr})==2
                assert all(r["signed_count"]>=4 and r["signed_value_changes"]>=3 for r in wr)
                assert {r["truth"] for r in wr}=={0,1}
            expected_k=({7,9} if split=="dev" else {12,16})
            assert {r["k"] for r in control}==expected_k
            if split=="ood":
                assert set(r["source_mode"] for r in control)==set(SOURCE_MODES)
                cross={(k,m):set() for k in expected_k for m in SOURCE_MODES}
                for r in control:
                    cross[(r["k"],r["source_mode"])].add(r["seed"])
                assert all(len(s)==16 for s in cross.values())
        for r in rows:
            assert r["replica"]==rep and r["truth"] in (0,1)
            assert r["variant"] in VARIANTS
            assert r["signed_count"]>=4 and r["signed_value_changes"]>=3
            assert set(r["predictions"])==set(MODES)
            assert all(x in (0,1) for x in r["predictions"].values())
            assert 0<=r["winner_weight"]<=1
            assert r["baselines"]["last_signed_public"]==r["truth"]
        metrics={
          split:{arm:summary([r for r in rows if r["split"]==split and r["arm"]==arm],
                             has_selector=arm!="gru_only") for arm in ARMS}
          for split in ("dev","ood")
        }
        arms={arm:[r for r in rows if r["split"]=="ood" and r["arm"]==arm]
              for arm in ARMS}
        contrasts={
            "selection_effect":bootstrap(arms["selection_only"],arms["joint"],
                                         "selection","selection",rep),
            "interpretation_effect":bootstrap(arms["interpretation_only"],arms["joint"],
                                              "oracle_patch","oracle_patch",rep),
            "both_vs_joint":bootstrap(arms["both"],arms["joint"],"read","read",rep),
            "both_vs_selection_only":bootstrap(arms["both"],arms["selection_only"],"read","read",rep),
            "both_vs_interpretation_only":bootstrap(arms["both"],arms["interpretation_only"],"read","read",rep),
            "read_skip":bootstrap(arms["both"],arms["both"],"read","skip",rep),
            "read_shuffle":bootstrap(arms["both"],arms["both"],"read","shuffle",rep),
        }
        best=("selection_only" if metrics["ood"]["selection_only"]["read"]>=
                             metrics["ood"]["interpretation_only"]["read"]
              else "interpretation_only")
        contrasts["both_vs_best_single"]=contrasts["both_vs_"+best]
        contrasts["best_single"]=best
        both=metrics["ood"]["both"]
        sg=contrasts["selection_effect"]
        it=contrasts["interpretation_effect"]
        jj=contrasts["both_vs_joint"]
        bs=contrasts["both_vs_best_single"]
        rskip=contrasts["read_skip"]
        rshuffle=contrasts["read_shuffle"]
        g1=(both["selection"]>=.90 and both["read"]>=.90
            and min(both["by_variant"].values())>=.85
            and min(both["by_k"].values())>=.85
            and min(both["by_source"].values())>=.85
            and both["all_four"]>=.80
            and rskip["ci95"][0]>.20 and rshuffle["ci95"][0]>.10)
        g2=(sg["delta"]>=.10 and sg["ci95"][0]>0
            and it["delta"]>=.10 and it["ci95"][0]>0)
        g3=(jj["delta"]>=.10 and jj["ci95"][0]>0
            and bs["delta"]>=.05 and bs["ci95"][0]>0)
        result["replicas"][str(rep)]={
            "metrics":metrics,"contrasts":contrasts,
            "gates":{"G1_capacity_transfer":g1,
                     "G2_separate_contributions":g2,
                     "G3_complementarity":g3},
            "budget":train["arms"],
        }
        print("F10H_REPLICA_GATE",rep,"G1",g1,"G2",g2,"G3",g3,
              "OOD_read",{a:round(metrics["ood"][a]["read"],5) for a in ARMS},
              "selection",{a:metrics["ood"][a]["selection"] for a in ARMS[:-1]},
              "interpretation",{a:metrics["ood"][a]["oracle_patch"] for a in ARMS[:-1]},
              "contrasts",{k:v for k,v in contrasts.items() if k in
                 ("selection_effect","interpretation_effect","both_vs_joint",
                  "both_vs_best_single","read_skip")},flush=True)
        aggregated.extend(rows)
    assert len(global_hashes)==4*(2048+512+512)
    result["unique_state_hashes"]=len(global_hashes)
    counts={
      g:sum(int(r["gates"][g]) for r in result["replicas"].values())
      for g in ("G1_capacity_transfer","G2_separate_contributions","G3_complementarity")
    }
    result["gate_replica_counts"]=counts
    result["gates"]={k:v>=3 for k,v in counts.items()}
    result["status"]=(
        "F10H_FACTORIAL_SUPERVISION_PASS" if all(result["gates"].values())
        else "F10H_FACTORIAL_SUPERVISION_FAIL")
    result["aggregate"]={
      split:{arm:summary([r for r in aggregated if r["split"]==split and r["arm"]==arm],
                         has_selector=arm!="gru_only") for arm in ARMS}
      for split in ("dev","ood")
    }
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    audit=verify(args.root)
    args.output.write_text(json.dumps(audit,indent=2))
    print("F10H_GLOBAL_FIXED_GATES",audit["status"],audit["gate_replica_counts"],flush=True)
