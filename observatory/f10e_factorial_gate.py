"""F10-E preregistered 2x2 factorial diagnostic gate; stdlib independent.

Audits EACH row, EACH 4-case matched world, each count/timing cell,
and all four causal contrasts. Does not train, tune or open any worlds.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

N=108
VARIANTS=("U_before","U_after","S_before","S_after")
COUNTS=("3","5","8")
TIMINGS=("early","middle","late")
MODES=("rel_read","rel_skip","unary_read","gru_only",
       "f10b_read","f10b_gru_only","last_any","last_signed","first_signed")
CONTRASTS=("credential_useful","precedence_useful",
           "credential_null_stable","precedence_null_stable")

def correct(r,mode):
    return int((r["pred"][mode]>=.5)==r["truth"])

def selected(rows):
    return {mode:sum(correct(r,mode) for r in rows) for mode in MODES}

def audit(reports):
    assert len(reports)==4
    out={}
    g1=g2=g3=0
    for index,r in enumerate(reports):
        assert r["stage"]=="CAUSAL_FACTORIAL_DEV_ONLY" and r["index"]==index
        assert r["n_base_worlds"]==N and r["n_variants_per_world"]==4
        assert r["n_cases"]==4*N
        assert r["seeds"]==[16200000+10000*index+j for j in range(N)]
        assert r["all_models_frozen"] and not r["train_performed"]
        assert not r["reader_head_trained"] and not r["old_worlds_revisited"]
        assert not r["confirmation_16300000_opened"]
        assert r["truth_from_public_signed_evidence_only"]
        assert r["paired_factorial_changes_public_history"]
        assert r["read_skip_same_recurrent_history_within_variant"]
        assert r["source_sha_provenance"]["f10c_original_run"]==41447171697
        assert r["source_sha_provenance"]["f10b_original_run"]==41443064754
        assert len(r["source_sha_provenance"]["frozen_checkpoint_sha256"])==5
        rows=r["rows"]
        assert len(rows)==N*4
        grouped={}
        for x in rows:
            assert x["seed"] in r["seeds"]
            assert x["variant"] in VARIANTS
            assert x["record_count"] in (3,5,8)
            assert x["timing"] in TIMINGS
            assert x["truth"] in (0,1) and x["base_value"] in (0,1)
            assert x["challenge_source"] in (0,1) and x["confirm_source"] in (0,1)
            assert set(x["pred"])==set(MODES)
            assert all(0<=v<=1 for v in x["pred"].values())
            assert correct(x,"last_signed")
            assert int(x["pred"]["first_signed"])==x["base_value"]
            expected=(1-x["base_value"] if x["variant"]=="S_after"
                      else x["base_value"])
            assert x["truth"]==expected
            grouped.setdefault(x["seed"],{})[x["variant"]]=x
        assert len(grouped)==N
        assert all(len(v)==4 and set(v)==set(VARIANTS) for v in grouped.values())
        world_joint={key:0 for key in (
            "all_four_correct",*CONTRASTS,"always_initial_signed_baseline_correct")}
        cross={}
        for seed,group in grouped.items():
            xs=[group[k] for k in VARIANTS]
            assert len({x["key"] for x in xs})==1
            assert len({x["record_count"] for x in xs})==1
            assert len({x["timing"] for x in xs})==1
            assert len({x["base_value"] for x in xs})==1
            assert len({x["challenge_source"] for x in xs})==1
            assert len({x["confirm_source"] for x in xs})==1
            assert len({x["history_sha256"] for x in xs})==4
            k=xs[0]["record_count"];timing=xs[0]["timing"]
            cross[(k,timing)]=cross.get((k,timing),0)+1
            b=xs[0]["base_value"]
            p={key:int(group[key]["pred"]["rel_read"]>=.5) for key in VARIANTS}
            world_joint["all_four_correct"]+=int(
                all(p[key]==group[key]["truth"] for key in VARIANTS))
            world_joint["credential_useful"]+=int(
                p["U_after"]==b and p["S_after"]==1-b)
            world_joint["precedence_useful"]+=int(
                p["S_before"]==b and p["S_after"]==1-b)
            world_joint["credential_null_stable"]+=int(
                p["U_before"]==b and p["S_before"]==b)
            world_joint["precedence_null_stable"]+=int(
                p["U_before"]==b and p["U_after"]==b)
        assert set(cross)=={(k,t) for k in (3,5,8) for t in TIMINGS}
        assert all(v==12 for v in cross.values())
        assert r["overall"]["paired_contrasts"]==world_joint
        overall=selected(rows)
        assert r["overall"]["correct"]==overall
        assert overall["last_signed"]==4*N
        assert overall["first_signed"]==3*N
        for axis,values,attr,expected_size in (
            ("by_variant",VARIANTS,"variant",N),
            ("by_k",COUNTS,"record_count",N//3*4),
            ("by_timing",TIMINGS,"timing",N//3*4)):
            assert set(r[axis])==set(values)
            for v in values:
                subset=[x for x in rows if str(x[attr])==str(v)]
                assert len(subset)==expected_size
                assert r[axis][v]["cases"]==expected_size
                assert r[axis][v]["correct"]==selected(subset)
        effects={str(v):sum(
            correct(x,"rel_read")-correct(x,"rel_skip")==v for x in rows)
            for v in (-1,0,1)}
        assert r["overall"]["D_read_skip"]==effects
        assert sum(effects.values())==N*4
        delta=(overall["rel_read"]-overall["rel_skip"])/(N*4)
        delta_gru=(overall["rel_read"]-overall["gru_only"])/(N*4)
        delta_unary=(overall["rel_read"]-overall["unary_read"])/(N*4)
        ci=r["overall"]["world_cluster_ci"]
        assert ci["unit"]=="world" and ci["draws"]==2000
        assert abs(ci["estimate"]-delta)<1e-7
        pass_g1=(
            all(r["by_variant"][var]["correct"]["rel_read"]>=87
                for var in VARIANTS)
            and world_joint["all_four_correct"]>=76
            and all(r["by_k"][k]["correct"]["rel_read"]>=108
                    for k in COUNTS)
            and all(r["by_timing"][t]["correct"]["rel_read"]>=108
                    for t in TIMINGS))
        pass_g2=all(world_joint[contrast]>=87 for contrast in CONTRASTS)
        pass_g3=delta>=.10 and ci["ci95"][0]>0 and delta_gru>=.05
        g1+=int(pass_g1);g2+=int(pass_g2);g3+=int(pass_g3)
        out[str(index)]={
            "correct":overall,"D_clean":effects,
            "read_skip_delta":delta,"read_skip_ci95":ci["ci95"],
            "read_minus_gru":delta_gru,"read_minus_unary":delta_unary,
            "variant":{v:r["by_variant"][v]["correct"] for v in VARIANTS},
            "count":{k:r["by_k"][k]["correct"] for k in COUNTS},
            "timing":{t:r["by_timing"][t]["correct"] for t in TIMINGS},
            "contrasts":world_joint,
            "g1":pass_g1,"g2":pass_g2,"g3":pass_g3,
            "source_checkpoints":r["source_sha_provenance"]["frozen_checkpoint_sha256"],
        }
    gates={"G1_joint_truth":g1>=3,
           "G2_credential_precedence_contrasts":g2>=3,
           "G3_external_memory_utility":g3>=3}
    return {"experiment":"F10-E frozen actor public evidence causal factorial",
            "phase":"DEV_DIAGNOSTIC_ONLY",
            "status":("F10E_FACTORIAL_DIAGNOSTIC_PASS" if all(gates.values())
                      else "F10E_FACTORIAL_DIAGNOSTIC_FAIL"),
            "gates":gates,"replicas_passing":{"G1":g1,"G2":g2,"G3":g3},
            "no_training":True,"no_confirmation_opened":True,
            "all_original_checkpoints_frozen":True,
            "per_replica":out}

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gathered",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    reports=[json.loads((a.gathered/f"f10e-actor-{i}"/"dev-report.json").read_text())
             for i in range(4)]
    result=audit(reports)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("F10E_PREREG_DIAGNOSTIC_GATE",json.dumps({
        "status":result["status"],"gates":result["gates"],
        "replicas_passing":result["replicas_passing"],
        "replicas":{k:{
            "correct":v["correct"],"contrasts":v["contrasts"],
            "read_skip_delta":v["read_skip_delta"],
            "read_minus_unary":v["read_minus_unary"],
            "ci95":v["read_skip_ci95"],
            "gates":[v["g1"],v["g2"],v["g3"]],
            "by_variant":v["variant"],
            "by_count":v["count"],
            "by_timing":v["timing"],
            } for k,v in result["per_replica"].items()}},ensure_ascii=False),flush=True)
