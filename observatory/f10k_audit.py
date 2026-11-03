"""Independent F10-K preregistered gate and SHA audit; no torch/model imports."""
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=("joint_shared","staged_shared","staged_unshared","supervised_ceiling")
TYPES=("P","I","S","M")
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCE_MODES=("iid","signed_source0","signed_source1","alternating")
MODES=("read","no_flip","shuffle","oracle_slot")

def correct(r,mode="read"):
    return int(r["predictions"][mode]["action"]==r["truth"])

def rate(rows,mode="read"):
    assert rows
    return sum(correct(r,mode) for r in rows)/len(rows)

def metrics(rows):
    return {
        "n":len(rows),"base_worlds":len({r["seed"] for r in rows}),
        "accuracy":rate(rows),
        "selected_winner":sum(r["predictions"]["read"]["slot"]==r["winner"]
                              for r in rows)/len(rows),
        "no_flip":rate(rows,"no_flip"),
        "shuffle":rate(rows,"shuffle"),
        "oracle_slot":rate(rows,"oracle_slot"),
        "by_variant":{v:rate([r for r in rows if r["variant"]==v]) for v in VARIANTS},
        "by_n":{str(n):rate([r for r in rows if r["n"]==n])
                for n in sorted({r["n"] for r in rows})},
        "by_source":{m:rate([r for r in rows if r["source_mode"]==m])
                     for m in sorted({r["source_mode"] for r in rows})},
    }

def contrast(left,right,left_mode,right_mode,rep):
    a={(r["seed"],r["variant"]):r for r in left}
    b={(r["seed"],r["variant"]):r for r in right}
    assert a.keys()==b.keys()
    seeds=sorted({k[0] for k in a})
    assert len(seeds)==128
    delta=np.array([sum(correct(a[(s,v)],left_mode)-correct(b[(s,v)],right_mode)
                        for v in VARIANTS)/4. for s in seeds],np.float64)
    rng=np.random.default_rng(227000+rep)
    samples=delta[rng.integers(0,len(seeds),size=(2000,len(seeds)))].mean(axis=1)
    return {
        "estimate":float(delta.mean()),
        "ci95":[float(x) for x in np.quantile(samples,(.025,.975))],
        "world_count":128,"bootstrap":2000,"unit":"base_world",
    }

def absolute_ci(rows,rep):
    groups={}
    for r in rows:
        groups.setdefault(r["seed"],[]).append(correct(r))
    assert len(groups)==128
    vals=np.array([sum(x)/len(x) for _,x in sorted(groups.items())],np.float64)
    rng=np.random.default_rng(227000+rep)
    samples=vals[rng.integers(0,len(vals),size=(2000,len(vals)))].mean(axis=1)
    return [float(x) for x in np.quantile(samples,(.025,.975))]

def audit(root):
    out={
        "experiment":"INFINITO F10-K frozen reward-only primitive to novel composition",
        "technical":"F10K_TECHNICAL_COMPLETE",
        "confirm_22300000_never_opened":True,
        "scientific_frozen_no_tuning":True,
        "replicas":{},
    }
    all_hashes=set()
    for rep in range(4):
        folder=root/str(rep)
        meta=json.loads((folder/"train.json").read_text())
        rows=json.loads((folder/"eval.json").read_text())
        assert meta["status"]=="ALL_CHECKPOINTS_FROZEN_BEFORE_DEV_AND_OOD"
        assert meta["replica"]==rep and meta["train_counts"]=={"P":2048,"I":2048,"S":2048}
        assert meta["never_train_memory_plus_flip_one"]
        assert meta["confirm_22300000_unopened"]
        assert meta["batch"]==64 and meta["lr"]==.003
        assert meta["entropy"]==.002 and meta["ema"]==.9
        assert set(meta["arms"])==set(ARMS)
        train_hashes=meta["train_case_sha"]
        assert len(train_hashes)==6144 and len(set(train_hashes))==6144
        assert not set(train_hashes)&all_hashes
        all_hashes.update(train_hashes)
        for arm in ARMS:
            x=meta["arms"][arm]
            assert x["checkpoint"]==arm+".pt"
            assert hashlib.sha256((folder/x["checkpoint"]).read_bytes()).hexdigest()==x["sha256"]
            assert x["episodes"]==24576 and x["updates"]==384
            assert x["episode_counts"]=={"P":8192,"I":8192,"S":8192,"M":0}
            assert x["rl_no_winner_or_truth_target"]==(arm!="supervised_ceiling")
            assert x["used_supervised_labels"]==(arm=="supervised_ceiling")
            assert len(x["curves"])==16
        assert len(rows)==5120
        assert len({(r["arm"],r["case_sha"]) for r in rows})==5120
        canonical={}
        for arm in ARMS:
            sub=[r for r in rows if r["arm"]==arm]
            data={(r["split"],r["seed"],r["task"],r["variant"]):
                  (r["case_sha"],r["truth"],r["winner"],r["source_mode"],
                   r["n"],r["flip"],r["public_bit"])
                  for r in sub}
            assert len(data)==1280
            if arm==ARMS[0]:canonical=data
            else:assert canonical==data,"unequal data across arms"
        for split,task_count in (("dev",64),("ood",128)):
            group=[r for r in rows if r["arm"]==ARMS[0] and r["split"]==split]
            assert len(group)==(768 if split=="dev" else 512)
            tasks=("P","I","S") if split=="dev" else ("M",)
            assert {r["task"] for r in group}==set(tasks)
            hashes={r["case_sha"] for r in group}
            assert len(hashes)==len(group) and not hashes&all_hashes
            all_hashes.update(hashes)
            for task in tasks:
                t=[r for r in group if r["task"]==task]
                base=22_200_000 if split=="dev" else 22_400_000
                offset={"P":0,"I":2000,"S":4000,"M":0}[task]
                expected={base+10000*rep+offset+j for j in range(task_count)}
                assert {r["seed"] for r in t}==expected
                assert len(t)==4*task_count
                for seed in expected:
                    siblings={r["variant"]:r for r in t if r["seed"]==seed}
                    assert set(siblings)==set(VARIANTS)
                    if task in ("M","S"):
                        assert siblings["U_before"]["truth"]==siblings["U_after"]["truth"]
                        assert siblings["U_after"]["truth"]==siblings["S_before"]["truth"]
                        assert siblings["S_after"]["truth"]!=siblings["U_after"]["truth"]
                    if task=="M":
                        assert all(r["flip"]==1 and r["n"] in (9,11) for r in siblings.values())
                    if task=="P":
                        assert {r["flip"] for r in siblings.values()}=={0,1}
                        assert {r["truth"] for r in siblings.values()}=={0,1}
                    if task in ("I","S"):
                        assert all(r["flip"]==0 for r in siblings.values())
        if rep==0:
            assert not (set(train_hashes)&hashes)
        for r in rows:
            assert r["replica"]==rep and r["truth"] in (0,1)
            assert set(r["predictions"])==set(MODES)
            for v in r["predictions"].values():
                assert v["action"] in (0,1)
                assert 0<=v["slot"]<12
                assert 0<=v["p_action"]<=1
        split_stats={}
        for split,tasks in (("dev",("P","I","S")),("ood",("M",))):
            split_stats[split]={
                arm:{task:metrics([r for r in rows if r["split"]==split
                                   and r["arm"]==arm and r["task"]==task])
                     for task in tasks}
                for arm in ARMS
            }
        m={arm:[r for r in rows if r["split"]=="ood" and r["arm"]==arm]
           for arm in ARMS}
        contrasts={
            "B_minus_A":contrast(m["staged_shared"],m["joint_shared"],"read","read",rep),
            "B_minus_C":contrast(m["staged_shared"],m["staged_unshared"],"read","read",rep),
            "B_minus_shuffle":contrast(m["staged_shared"],m["staged_shared"],"read","shuffle",rep),
            "B_minus_no_flip":contrast(m["staged_shared"],m["staged_shared"],"read","no_flip",rep),
        }
        b=split_stats["ood"]["staged_shared"]["M"]
        d=split_stats["dev"]["staged_shared"]
        ci=absolute_ci(m["staged_shared"],rep)
        g0=(d["P"]["accuracy"]>=.90 and d["I"]["accuracy"]>=.90 and
            d["S"]["accuracy"]>=.80 and d["S"]["selected_winner"]>=.80)
        g1=(b["accuracy"]>=.80 and ci[0]>.65 and
            min(b["by_variant"].values())>=.70 and
            min(b["by_n"].values())>=.70 and
            min(b["by_source"].values())>=.70)
        g2=all(contrasts[name]["estimate"]>=.20 and
               contrasts[name]["ci95"][0]>0
               for name in ("B_minus_shuffle","B_minus_no_flip"))
        g3=(contrasts["B_minus_A"]["estimate"]>=.05 and
            contrasts["B_minus_A"]["ci95"][0]>0)
        out["replicas"][str(rep)]={
            "metrics":split_stats,"contrasts":contrasts,
            "composition_accuracy_ci95_world":ci,
            "gates":{"G0_primitive_competence":g0,
                     "G1_zero_shot_composition":g1,
                     "G2_causal_reuse":g2,
                     "G3_curriculum_gain":g3},
            "train":{a:{k:meta["arms"][a][k] for k in
                ("sha256","parameters","episodes","updates","seconds","curves")}
                for a in ARMS},
        }
        print("F10K_REPLICA_GATES",rep,
              "DEV_P_I_S",{
                  k:round(d[k]["accuracy"],4) for k in ("P","I","S")
              },"DEV_SELECTOR",round(d["S"]["selected_winner"],4),
              "OOD_M",{a:round(split_stats["ood"][a]["M"]["accuracy"],4)
                        for a in ARMS},
              "ZERO_SHOT_CI",ci,
              "GATES",out["replicas"][str(rep)]["gates"],flush=True)
    assert len(all_hashes)==29696
    out["unique_case_id_hashes_with_seeds"]=len(all_hashes)
    counts={key:sum(int(r["gates"][key]) for r in out["replicas"].values())
            for key in ("G0_primitive_competence","G1_zero_shot_composition",
                        "G2_causal_reuse","G3_curriculum_gain")}
    out["gate_counts"]=counts
    out["scientific_gates"]={key:value>=3 for key,value in counts.items()}
    out["status"]=(
        "F10K_COMPOSITION_PASS" if all(out["scientific_gates"][g] for g in
            ("G0_primitive_competence","G1_zero_shot_composition","G2_causal_reuse"))
        else "F10K_COMPOSITION_FAIL"
    )
    print("F10K_FINAL_FIXED_STATUS",out["status"],out["gate_counts"],flush=True)
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.root)
    a.out.write_text(json.dumps(result,indent=2))
