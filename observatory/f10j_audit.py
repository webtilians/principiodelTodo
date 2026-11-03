"""F10-J independent preregistered scientific audit, no torch/model imports.

Each 4-way counterfactual group is ONE bootstrap unit. Only frozen rows are read.
"""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=("on_policy","forced_gate","forced_gate_slot_explore","paired_credit","supervised_ceiling")
VARIANTS=("U_late","U_early","S_late","S_early")
MODES=("read","force_read","force_skip","shuffle","erase_sign","erase_time")
SOURCES=("iid","signed_source_0","signed_source_1","alternating_storage_source")

def success(r,mode="read"):
    return int(r["predictions"][mode]["action"]==r["truth"])

def mean_accuracy(rows,mode="read"):
    assert rows
    return sum(success(r,mode) for r in rows)/len(rows)

def stat(rows):
    if not rows:
        return None
    reads=sum(r["predictions"]["read"]["read"] for r in rows)
    return {
        "n_predictions":len(rows),"n_worlds":len({r["seed"] for r in rows}),
        "accuracy":mean_accuracy(rows),
        "read_fraction":reads/len(rows),
        "skip_fraction":1-reads/len(rows),
        "selection_accuracy":sum(r["predictions"]["read"]["selected"]==r["winner"]
                                 for r in rows)/len(rows),
        "forced_read_accuracy":mean_accuracy(rows,"force_read"),
        "forced_skip_accuracy":mean_accuracy(rows,"force_skip"),
        "shuffle_accuracy":mean_accuracy(rows,"shuffle"),
        "erase_sign_accuracy":mean_accuracy(rows,"erase_sign"),
        "erase_time_accuracy":mean_accuracy(rows,"erase_time"),
        "mean_read_probability":sum(r["predictions"]["read"]["read_probability"]
                                    for r in rows)/len(rows),
        "operational_return":sum(
            (2*success(r)-1)*(.99**r["delay"])-.05*r["predictions"]["read"]["read"]
            for r in rows)/len(rows),
        "all_four":sum(all(success(r) for r in rows if r["seed"]==seed)
                       for seed in {r["seed"] for r in rows})/len({r["seed"] for r in rows}),
        "by_k":{str(k):mean_accuracy([r for r in rows if r["k"]==k])
                for k in sorted({r["k"] for r in rows})},
        "by_source":{s:mean_accuracy([r for r in rows if r["source_mode"]==s])
                     for s in sorted({r["source_mode"] for r in rows})},
        "by_variant":{v:mean_accuracy([r for r in rows if r["variant"]==v])
                      for v in VARIANTS},
    }

def contrast(left,right,rep,mode_left="read",mode_right="read"):
    la={(r["seed"],r["variant"]):r for r in left}
    rb={(r["seed"],r["variant"]):r for r in right}
    assert la.keys()==rb.keys()
    worlds=sorted({s for s,_ in la})
    assert len(worlds)==64,"Informative world grouping should have 64 base worlds"
    deltas=np.array([
        sum(success(la[(seed,v)],mode_left)-success(rb[(seed,v)],mode_right)
            for v in VARIANTS)/4. for seed in worlds
    ],dtype=np.float64)
    rng=np.random.default_rng(217000+rep)
    samples=deltas[rng.integers(0,len(worlds),size=(2000,len(worlds)))].mean(axis=1)
    return {
        "point":float(deltas.mean()),
        "ci95":[float(x) for x in np.quantile(samples,(.025,.975))],
        "worlds":len(worlds),"bootstrap":2000,"unit":"base_world",
    }

def audit(root):
    out={
        "experiment":"INFINITO F10-J prospectively frozen structured information exploration",
        "technical_status":"F10J_TECHNICAL_COMPLETE",
        "confirm_21300000_unopened":True,
        "product_master_not_modified_by_experiment":True,
        "replicas":{},
    }
    all_hashes=set()
    for rep in range(4):
        folder=root/str(rep)
        t=json.loads((folder/"training.json").read_text())
        rows=json.loads((folder/"rows.json").read_text())
        assert t["status"]=="FROZEN_BEFORE_DEV_OOD"
        assert t["replica"]==rep and t["train_worlds"]==512 and t["train_cases"]==2048
        assert t["epochs"]==20 and t["batch"]==64 and t["lr"]==.002
        assert t["gamma"]==.99 and t["read_cost"]==.05
        assert t["entropy"]==.005 and t["ema_momentum"]==.95
        assert t["slot_explore_epsilon"]==.5 and t["train_reward_flip_probability"]==.1
        assert t["two_executed_episodes_per_state"]
        assert not t["truth_used_for_rl_targets"] and not t["winner_used_for_rl_targets"]
        assert t["labels_only_for_privileged_ceiling"] and t["confirm_21300000_unopened"]
        assert set(t["arms"])==set(ARMS)
        assert len(t["train_hashes"])==2048
        assert len(set(t["train_hashes"]))==2048
        assert not set(t["train_hashes"])&all_hashes
        all_hashes.update(t["train_hashes"])
        n_params=None
        for arm in ARMS:
            a=t["arms"][arm]
            assert a["checkpoint"]==arm+".pt"
            assert hashlib.sha256((folder/a["checkpoint"]).read_bytes()).hexdigest()==a["sha256"]
            assert a["optimizer_steps"]==640 and a["train_presentations"]==40960
            assert a["executed_environment_episodes"]==81920
            assert a["privileged_labels"]==(arm=="supervised_ceiling")
            assert len(a["curves"])==20
            assert a["curves"][-1]["environment_rollouts"]==81920
            if n_params is None:
                n_params=a["parameters"]
            else:
                assert n_params==a["parameters"]
        assert len(rows)==5120
        assert len({(r["arm"],r["split"],r["seed"],r["variant"]) for r in rows})==5120
        all_map={}
        for arm in ARMS:
            arm_rows=[r for r in rows if r["arm"]==arm]
            base_map={(r["split"],r["seed"],r["variant"]):(
                r["hash"],r["truth"],r["winner"],r["informative"],r["k"],
                r["signed_count"],r["source_mode"],r["xor"],r["public_bit"],r["delay"])
                for r in arm_rows}
            assert len(base_map)==1024
            if arm==ARMS[0]:
                all_map=base_map
            else:
                assert all_map==base_map
        for split,base,lengths in (
                ("dev",21_200_000,{6,9}),("ood",21_400_000,{12,16})):
            block=[r for r in rows if r["split"]==split and r["arm"]==ARMS[0]]
            assert len(block)==512
            expected=set(range(base+10_000*rep,base+10_000*rep+128))
            assert {r["seed"] for r in block}==expected
            hashes={r["hash"] for r in block}
            assert len(hashes)==512 and not hashes&all_hashes
            all_hashes.update(hashes)
            assert {r["k"] for r in block}==lengths
            for seed in expected:
                variants={r["variant"]:r for r in block if r["seed"]==seed}
                assert set(variants)==set(VARIANTS)
                assert len({r["informative"] for r in variants.values()})==1
                assert len({r["k"] for r in variants.values()})==1
                assert len({r["source_mode"] for r in variants.values()})==1
                if variants["U_late"]["informative"]:
                    assert variants["U_late"]["truth"]!=variants["S_late"]["truth"]
                    assert variants["U_late"]["truth"]==variants["S_early"]["truth"]
                    assert variants["U_late"]["truth"]==variants["U_early"]["truth"]
                else:
                    assert len({r["truth"] for r in variants.values()})==1
            if split=="ood":
                for info in (False,True):
                    for length in lengths:
                        for source in SOURCES:
                            n=len({r["seed"] for r in block if r["informative"]==info
                                   and r["k"]==length and r["source_mode"]==source})
                            assert n==8,(rep,info,length,source,n)
            for r in block:
                assert r["signed_count"]>=3 and r["signed_changes"]>=2
                assert r["delay"] in (2,3,4)
                assert r["truth"] in (0,1)
                assert r["baseline_public"]==r["public_bit"]^r["xor"]
                if not r["informative"]:
                    assert r["baseline_public"]==r["truth"]
                assert set(r["predictions"])==set(MODES)
                for mode in MODES:
                    p=r["predictions"][mode]
                    assert p["action"] in (0,1) and p["read"] in (0,1)
                    assert 0<=p["selected"]<48
                    assert 0<=p["read_probability"]<=1
                    assert 0<=p["action_probability"]<=1
        splits={}
        for split in ("dev","ood"):
            splits[split]={}
            for arm in ARMS:
                ar=[r for r in rows if r["split"]==split and r["arm"]==arm]
                splits[split][arm]={
                    "overall":stat(ar),
                    "informative":stat([r for r in ar if r["informative"]]),
                    "public":stat([r for r in ar if not r["informative"]]),
                }
        grp={a:[r for r in rows if r["split"]=="ood" and
                r["arm"]==a and r["informative"]] for a in ARMS}
        contrasts={
            "B_minus_A":contrast(grp["forced_gate"],grp["on_policy"],rep),
            "C_minus_B":contrast(grp["forced_gate_slot_explore"],grp["forced_gate"],rep),
            "D_minus_C":contrast(grp["paired_credit"],grp["forced_gate_slot_explore"],rep),
            "C_read_minus_skip":contrast(grp["forced_gate_slot_explore"],grp["forced_gate_slot_explore"],rep,
                                          "force_read","force_skip"),
            "D_read_minus_skip":contrast(grp["paired_credit"],grp["paired_credit"],rep,
                                          "force_read","force_skip"),
        }
        autonomous={}
        for name,key in (("forced_gate_slot_explore","C_read_minus_skip"),
                         ("paired_credit","D_read_minus_skip")):
            inf=splits["ood"][name]["informative"]
            pub=splits["ood"][name]["public"]
            c=contrasts[key]
            autonomous[name]=(
                inf["accuracy"]>=.75 and
                inf["selection_accuracy"]>=.65 and
                inf["read_fraction"]>=.80 and
                c["point"]>=.20 and c["ci95"][0]>0 and
                pub["accuracy"]>=.90 and pub["skip_fraction"]>=.80 and
                min(inf["by_k"].values())>=.65 and
                min(inf["by_source"].values())>=.65
            )
        gates={
            "C_autonomous":autonomous["forced_gate_slot_explore"],
            "D_autonomous":autonomous["paired_credit"],
            "G2_B_minus_A":contrasts["B_minus_A"]["point"]>=.10 and
                           contrasts["B_minus_A"]["ci95"][0]>0,
            "G3_C_minus_B":contrasts["C_minus_B"]["point"]>=.08 and
                           contrasts["C_minus_B"]["ci95"][0]>0,
            "G4_D_minus_C":contrasts["D_minus_C"]["point"]>=.05 and
                           contrasts["D_minus_C"]["ci95"][0]>0,
        }
        out["replicas"][str(rep)]={
            "metrics":splits,"contrasts":contrasts,"gates":gates,
            "training":{a:{k:t["arms"][a][k] for k in
                ("sha256","parameters","optimizer_steps","train_presentations",
                 "executed_environment_episodes","privileged_labels","wall_seconds","curves")}
                for a in ARMS}
        }
        print("F10J_REPLICA_FIXED_GATES",rep,
              "OOD_INFO",{a:round(splits["ood"][a]["informative"]["accuracy"],4) for a in ARMS},
              "OOD_PUBLIC",{a:round(splits["ood"][a]["public"]["accuracy"],4) for a in ARMS},
              "READ_INFO",{a:round(splits["ood"][a]["informative"]["read_fraction"],3) for a in ARMS},
              "GATES",gates,flush=True)
    assert len(all_hashes)==12288
    out["unique_history_and_F4_hashes"]=len(all_hashes)
    gcounts={
        "G1_C_or_D":sum(x["gates"]["C_autonomous"] or x["gates"]["D_autonomous"]
                        for x in out["replicas"].values()),
        "G2_B_minus_A":sum(x["gates"]["G2_B_minus_A"] for x in out["replicas"].values()),
        "G3_C_minus_B":sum(x["gates"]["G3_C_minus_B"] for x in out["replicas"].values()),
        "G4_D_minus_C":sum(x["gates"]["G4_D_minus_C"] for x in out["replicas"].values()),
    }
    out["gate_counts"]=gcounts
    out["gates"]={k:v>=3 for k,v in gcounts.items()}
    out["status"]=("F10J_STRUCTURED_EXPLORATION_PASS" if all(out["gates"].values())
                   else "F10J_STRUCTURED_EXPLORATION_FAIL")
    print("F10J_FIXED_GLOBAL",out["status"],out["gate_counts"],flush=True)
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    result=audit(args.root)
    args.output.write_text(json.dumps(result,indent=2))
