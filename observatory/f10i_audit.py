"""F10-I independent stdlib+NumPy audit, fixed scientific gates.

Never loads model code or torch and never changes training. World base is the
resampling unit: 4 causal variants from a seed ALWAYS stay together.
"""
from __future__ import annotations

import argparse,hashlib,json
from pathlib import Path
import numpy as np

ARMS=("rl_standard","rl_pair","rl_pair_predict","supervised_ceiling")
VARIANTS=("U_late","U_early","S_late","S_early")
MODES=("read","force_read","force_skip","shuffle","erase_sign","erase_time")
SOURCE_MODES=("iid","signed_source_0","signed_source_1","alternating_storage_source")

def correct(r,mode="read"):
    return int(r["prediction"][mode]==r["truth"])

def accuracy(rows,mode="read"):
    return sum(correct(x,mode) for x in rows)/len(rows)

def report(rows):
    seeds={r["seed"] for r in rows}
    assert len(rows)==4*len(seeds)
    results={
        "cases":len(rows),"worlds":len(seeds),
        "accuracy":accuracy(rows),
        "forced_read":accuracy(rows,"force_read"),
        "forced_skip":accuracy(rows,"force_skip"),
        "shuffle":accuracy(rows,"shuffle"),
        "erase_sign":accuracy(rows,"erase_sign"),
        "erase_time":accuracy(rows,"erase_time"),
        "read_fraction":sum(r["greedy_read"] for r in rows)/len(rows),
        "selection":sum(r["selected_slot"]==r["winner"] for r in rows)/len(rows),
        "mean_read_probability":sum(r["read_probability"] for r in rows)/len(rows),
        "net_environment_reward":sum(
            (2*correct(r)-1)*.99**r["delay"]-.05*r["greedy_read"] for r in rows)/len(rows),
        "all_four":sum(all(correct(r) for r in rows if r["seed"]==seed)
                       for seed in seeds)/len(seeds),
        "by_variant":{v:accuracy([r for r in rows if r["variant"]==v]) for v in VARIANTS},
        "by_k":{str(k):accuracy([r for r in rows if r["k"]==k])
                for k in sorted({r["k"] for r in rows})},
        "by_source":{s:accuracy([r for r in rows if r["source_mode"]==s])
                     for s in sorted({r["source_mode"] for r in rows})},
        "by_signed_count":{str(s):accuracy([r for r in rows if r["signed_count"]==s])
                      for s in sorted({r["signed_count"] for r in rows})},
        "baseline_accuracy":{
            name:sum(r["baselines"][name]==r["truth"] for r in rows)/len(rows)
            for name in rows[0]["baselines"]
        },
    }
    return results

def paired(a,b,mode_a,mode_b,rep):
    aa={(r["seed"],r["variant"]):r for r in a}
    bb={(r["seed"],r["variant"]):r for r in b}
    assert aa.keys()==bb.keys()
    worlds=sorted({k[0] for k in aa})
    assert len(worlds)==128
    differences=np.array([sum(
         correct(aa[(seed,v)],mode_a)-correct(bb[(seed,v)],mode_b)
         for v in VARIANTS)/4. for seed in worlds],dtype=np.float64)
    rng=np.random.default_rng(207000+rep)
    means=differences[rng.integers(0,len(differences),size=(2000,len(differences)))].mean(1)
    return {
        "estimate":float(differences.mean()),
        "ci95":[float(x) for x in np.quantile(means,[.025,.975])],
        "resamples":2000,"unit":"world","worlds":len(worlds)
    }

def analyze(root):
    result={
        "experiment":"INFINITO F10-I experience-only reward vs paired reward vs future prediction",
        "technical":"F10I_TECHNICAL_COMPLETE",
        "old_worlds_opened":False,
        "confirm_20300000_opened":False,
        "replicas":{},
    }
    all_hashes=set()
    pooled=[]
    for rep in range(4):
        folder=root/str(rep)
        meta=json.loads((folder/"training.json").read_text())
        rows=json.loads((folder/"rows.json").read_text())
        assert meta["stage"]=="FROZEN_BEFORE_DEV_OOD" and meta["replica"]==rep
        assert meta["train_worlds"]==512 and meta["train_cases"]==2048
        assert meta["epochs"]==20 and meta["batch"]==64 and meta["lr"]==.002
        assert meta["episodes_per_example_epoch"]==2
        assert meta["read_cost"]==.05 and meta["gamma"]==.99
        assert meta["entropy"]==.005 and meta["ema_momentum"]==.95
        assert meta["prediction_loss_weight"]==.30 and meta["prediction_intrinsic_weight"]==.10
        assert meta["no_old_training_or_checkpoint"] and meta["no_confirm_20300000"]
        assert meta["train_sensors_from_post_action_environment"]
        assert set(meta["arms"])==set(ARMS)
        assert len(meta["train_hashes"])==2048
        original_hashes=set(meta["train_hashes"])
        assert len(original_hashes)==2048 and not original_hashes&all_hashes
        all_hashes.update(original_hashes)
        params=None
        for arm in ARMS:
            a=meta["arms"][arm]
            assert a["checkpoint"]==arm+".pt"
            digest=hashlib.sha256((folder/a["checkpoint"]).read_bytes()).hexdigest()
            assert digest==a["sha256"]
            assert a["optimizer_steps"]==640
            assert a["train_state_presentations"]==40960
            assert a["environment_rollouts"]==81920
            assert a["environment_verifier_observations"]==81920
            assert a["used_winner_label_in_loss"]==(arm=="supervised_ceiling")
            assert a["used_action_label_in_loss"]==(arm=="supervised_ceiling")
            assert len(a["learning_curve"])==20 and len(a["losses"])==20
            assert a["learning_curve"][-1]["cumulative_rollouts"]==81920
            assert all(0<=x["sampled_action_accuracy"]<=1 and
                       0<=x["read_fraction"]<=1 for x in a["learning_curve"])
            if params is None:params=a["parameters"]
            else:assert params==a["parameters"]
        assert len(rows)==4*2*128*4
        assert len({(r["arm"],r["split"],r["seed"],r["variant"]) for r in rows})==len(rows)
        reference={}
        for arm in ARMS:
            group=[r for r in rows if r["arm"]==arm]
            observed={(r["split"],r["seed"],r["variant"]):
                (r["hash"],r["truth"],r["winner"],r["sensor"],r["delay"],
                 r["k"],r["signed_count"],r["signed_value_changes"],
                 r["source_mode"],r["baselines"])
                for r in group}
            assert len(observed)==1024
            if arm==ARMS[0]:reference=observed
            else:assert observed==reference,"Each arm must see exactly same experiences"
        for split,base,k_allowed in (
             ("dev",20_200_000,{6,9}),("ood",20_400_000,{12,16})):
            grp=[r for r in rows if r["arm"]==ARMS[0] and r["split"]==split]
            assert len(grp)==512
            assert {r["seed"] for r in grp}==set(range(base+rep*10000,base+rep*10000+128))
            hashes={r["hash"] for r in grp}
            assert len(hashes)==512 and not hashes&all_hashes
            all_hashes.update(hashes)
            for seed in {r["seed"] for r in grp}:
                same=[r for r in grp if r["seed"]==seed]
                assert {r["variant"] for r in same}==set(VARIANTS)
                assert len({r["k"] for r in same})==1
                assert len({r["source_mode"] for r in same})==1
                assert sum(r["truth"]!=same[0]["truth"] for r in same)==1
            assert {r["k"] for r in grp}==k_allowed
            if split=="ood":
                assert {r["source_mode"] for r in grp}==set(SOURCE_MODES)
                for k in k_allowed:
                    for mode in SOURCE_MODES:
                        assert len({r["seed"] for r in grp if
                                    r["k"]==k and r["source_mode"]==mode})==16
        for row in rows:
            assert row["replica"]==rep
            assert row["truth"] in (0,1) and row["sensor"] in (0,1)
            assert row["delay"] in (2,3,4)
            assert row["signed_count"]>=3 and row["signed_value_changes"]>=2
            assert set(row["prediction"])==set(MODES)
            assert all(v in (0,1) for v in row["prediction"].values())
            assert row["greedy_read"] in (0,1)
            assert 0<=row["read_probability"]<=1
            assert 0<=row["selected_slot"]<48
            assert row["baselines"]["last_signed_rule"]==row["truth"]
        metrics={
            split:{arm:report([r for r in rows if r["split"]==split and r["arm"]==arm])
                   for arm in ARMS} for split in ("dev","ood")
        }
        groups={a:[r for r in rows if r["split"]=="ood" and r["arm"]==a]
                for a in ARMS}
        contrasts={
            "paired_minus_standard":paired(groups["rl_pair"],groups["rl_standard"],"read","read",rep),
            "predict_minus_paired":paired(groups["rl_pair_predict"],groups["rl_pair"],"read","read",rep),
            "paired_force_read_skip":paired(groups["rl_pair"],groups["rl_pair"],"force_read","force_skip",rep),
            "predict_force_read_skip":paired(groups["rl_pair_predict"],groups["rl_pair_predict"],"force_read","force_skip",rep),
        }
        autonomous={}
        for arm,delta_key in (
             ("rl_pair","paired_force_read_skip"),
             ("rl_pair_predict","predict_force_read_skip")):
            m=metrics["ood"][arm]
            ci=contrasts[delta_key]
            autonomous[arm]=(
                m["accuracy"]>=.75 and
                min(m["by_variant"].values())>=.70 and
                min(m["by_k"].values())>=.70 and
                min(m["by_source"].values())>=.70 and
                m["selection"]>=.65 and
                ci["estimate"]>=.15 and ci["ci95"][0]>0
            )
        pair=contrasts["paired_minus_standard"]
        predictor=contrasts["predict_minus_paired"]
        g2=pair["estimate"]>=.08 and pair["ci95"][0]>0
        g3=predictor["estimate"]>=.05 and predictor["ci95"][0]>0
        result["replicas"][str(rep)]={
            "metrics":metrics,"contrasts":contrasts,
            "gates":{"rl_pair_autonomous":autonomous["rl_pair"],
                     "rl_pair_predict_autonomous":autonomous["rl_pair_predict"],
                     "G2_paired_gain":g2,"G3_prediction_gain":g3},
            "budgets":{a:{k:meta["arms"][a][k] for k in
                        ("parameters","optimizer_steps","train_state_presentations",
                         "environment_rollouts","environment_verifier_observations",
                         "seconds","sha256","supervision")}
                       for a in ARMS},
            "train_learning_curves":{a:meta["arms"][a]["learning_curve"] for a in ARMS},
        }
        print("F10I_REPLICA_FIXED_EVAL",rep,
              "ood",{a:round(metrics["ood"][a]["accuracy"],5) for a in ARMS},
              "selection",{a:round(metrics["ood"][a]["selection"],5) for a in ARMS},
              "autonomous",autonomous,"g2",g2,"g3",g3,
              "paired",pair,"sensor_pred",predictor,flush=True)
        pooled.extend(rows)
    assert len(all_hashes)==4*(2048+512+512)
    result["unique_history_hashes"]=len(all_hashes)
    counts={
        "G1_rl_pair":sum(bool(r["gates"]["rl_pair_autonomous"]) for r in result["replicas"].values()),
        "G1_rl_pair_predict":sum(bool(r["gates"]["rl_pair_predict_autonomous"]) for r in result["replicas"].values()),
        "G2_paired_gain":sum(bool(r["gates"]["G2_paired_gain"]) for r in result["replicas"].values()),
        "G3_prediction_gain":sum(bool(r["gates"]["G3_prediction_gain"]) for r in result["replicas"].values()),
    }
    result["gate_counts"]=counts
    result["gates"]={
        "G1_autonomous":max(counts["G1_rl_pair"],counts["G1_rl_pair_predict"])>=3,
        "G2_paired_gain":counts["G2_paired_gain"]>=3,
        "G3_prediction_gain":counts["G3_prediction_gain"]>=3,
    }
    result["status"]=("F10I_EXPERIENTIAL_LEARNING_PASS" if all(result["gates"].values())
                      else "F10I_EXPERIENTIAL_LEARNING_FAIL")
    result["aggregate"]={
        split:{arm:report([r for r in pooled if r["split"]==split and r["arm"]==arm])
               for arm in ARMS}
        for split in ("dev","ood")
    }
    return result

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    r=analyze(a.root)
    a.output.write_text(json.dumps(r,indent=2))
    print("F10I_FIXED_GLOBAL_STATUS",r["status"],r["gate_counts"],flush=True)
