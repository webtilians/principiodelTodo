"""F10-D external-only frozen transfer gates using stdlib.

Thresholds were committed before creating / evaluating 15200000+ worlds.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

CELLS=("signed|stable","unsigned|stable","signed|revised","unsigned|revised")
COUNTS=("3","4","6","8")
SOURCE_MODES=("iid","signed_A","signed_B","alternating")
MODES=("relational_read","relational_skip","relational_stale",
       "relational_forged","unary_read","unary_skip",
       "recurrent_only","f10b_frozen_read","f10b_frozen_recurrent",
       "public_last_signed_rule","public_last_any","public_majority")
N_WORLDS=64

def is_correct(r,name):
    return (r["scores"][name]>=.5)==r["truth"]

def count(rows,name):
    return sum(is_correct(r,name) for r in rows)

def aggregate(reports):
    assert len(reports)==4
    per={}
    g1=g2=g3=0
    for index,r in enumerate(reports):
        assert r["index"]==index and r["stage"]=="STRUCTURAL_DEV_ONLY"
        assert r["worlds"]==N_WORLDS and r["cases"]==4*N_WORLDS
        assert r["dev_world_seeds"]==[15200000+10000*index+j for j in range(N_WORLDS)]
        assert r["frozen_original_checkpoints"] and not r["training_performed"]
        assert not r["new_reader_trained"] and not r["original_confirmations_opened"]
        assert not r["F10D_15300000_opened"]
        assert r["labels_only_from_public_signed_records"]
        assert r["same_history_for_four_queries"]
        assert r["recurrent_state_fixed_across_f4_interventions"]
        assert r["provenance"]["f10c_original_run"]==41447171697
        assert r["provenance"]["f10b_original_run"]==41443064754
        assert len(r["provenance"]["frozen_checkpoint_sha256"])==5
        rows=r["rows"]
        assert len(rows)==256
        assert set(r["by_cell"])==set(CELLS)
        assert set(r["by_count"])==set(COUNTS)
        assert set(r["by_source"])==set(SOURCE_MODES)
        worlds={}
        for item in rows:
            assert item["seed"] in r["dev_world_seeds"]
            assert item["cell"] in CELLS
            assert str(item["record_count"]) in COUNTS
            assert item["source_mode"] in SOURCE_MODES
            assert item["truth"] in (0,1)
            assert set(item["scores"])==set(MODES)
            assert all(0<=v<=1 for v in item["scores"].values())
            assert is_correct(item,"public_last_signed_rule")
            worlds.setdefault(item["seed"],[]).append(item)
        assert len(worlds)==N_WORLDS
        assert all(len(items)==4 and
                   {x["cell"] for x in items}==set(CELLS) and
                   {x["record_count"] for x in items}=={3,4,6,8} and
                   len({x["history_sha256"] for x in items})==1 and
                   len({x["source_mode"] for x in items})==1
                   for items in worlds.values())
        table={}
        for selector,values in (
            ("by_cell",CELLS),("by_count",COUNTS),("by_source",SOURCE_MODES)):
            assert set(r[selector])==set(values)
            table[selector]={}
            for key in values:
                subset=[x for x in rows if (
                    x["cell"]==key if selector=="by_cell" else
                    str(x["record_count"])==key if selector=="by_count" else
                    x["source_mode"]==key)]
                assert len(subset)==64
                assert r[selector][key]["cases"]==64
                assert r[selector][key]["correct"]=={
                    m:count(subset,m) for m in MODES}
                table[selector][key]={m:count(subset,m) for m in MODES}
        assert set(r["cross_cell_count"])==set(CELLS)
        for cell in CELLS:
            assert set(r["cross_cell_count"][cell])==set(COUNTS)
            for k in COUNTS:
                subset=[x for x in rows if x["cell"]==cell
                        and str(x["record_count"])==k]
                assert len(subset)==16
                assert r["cross_cell_count"][cell][k]["cases"]==16
                assert r["cross_cell_count"][cell][k]["correct"]=={
                    m:count(subset,m) for m in MODES}
        corr={m:count(rows,m) for m in MODES}
        assert r["overall"]["correct"]==corr
        assert corr["public_last_signed_rule"]==256
        causal={str(v):sum(int(is_correct(x,"relational_read"))-
                            int(is_correct(x,"relational_skip"))==v for x in rows)
                for v in (-1,0,1)}
        assert r["overall"]["D_relational"]==causal
        assert sum(causal.values())==256
        read_skip=(corr["relational_read"]-corr["relational_skip"])/256
        rel_unary=(corr["relational_read"]-corr["unary_read"])/256
        rel_only=(corr["relational_read"]-corr["recurrent_only"])/256
        hard=[x for x in rows if x["cell"]=="unsigned|revised"]
        gain_hard=(count(hard,"relational_read")-count(hard,"unary_read"))/64
        bootstrap=r["overall"]["read_skip_world_bootstrap"]
        assert bootstrap["unit"]=="world" and bootstrap["samples"]==2000
        assert abs(bootstrap["estimate"]-read_skip)<1e-7
        assert r["overall"]["relational_vs_unary_world_bootstrap"]["unit"]=="world"
        pass1=(
            all(table["by_cell"][cell]["relational_read"]>=52 for cell in CELLS)
            and all(table["by_count"][k]["relational_read"]>=52 for k in COUNTS)
            and all(table["by_source"][mode]["relational_read"]>=45
                    for mode in SOURCE_MODES))
        pass2=(read_skip>=.10 and bootstrap["ci95"][0]>0 and rel_only>=.05)
        pass3=(rel_unary>=.05 and gain_hard>=.08)
        g1+=int(pass1);g2+=int(pass2);g3+=int(pass3)
        per[str(index)]={
            "correct":corr,"D_relational":causal,
            "read_skip_delta":read_skip,
            "read_skip_ci95_world":bootstrap["ci95"],
            "relational_minus_frozen_unary":rel_unary,
            "relational_minus_recurrent":rel_only,
            "hard_unsigned_revised_minus_unary":gain_hard,
            "g1_transfer_validity":pass1,
            "g2_causal_read":pass2,
            "g3_pairwise_advantage":pass3,
            "by_cell":table["by_cell"],
            "by_count":table["by_count"],
            "by_source":table["by_source"],
            "cross_cell_count":r["cross_cell_count"],
            "frozen_shas":r["provenance"]["frozen_checkpoint_sha256"],
        }
    gates={
        "G1_transfer_validity":g1>=3,
        "G2_causal_memory":g2>=3,
        "G3_relational_advantage":g3>=3,
    }
    return {
        "experiment":"F10-D frozen model structural transfer",
        "phase":"DEVELOPMENT_TRANSFER_ONLY",
        "status":"F10D_STRUCTURAL_TRANSFER_PASS" if all(gates.values())
                 else "F10D_STRUCTURAL_TRANSFER_FAIL",
        "gates":gates,"replica_gate_counts":{"G1":g1,"G2":g2,"G3":g3},
        "four_replicas_complete":True,"training_performed":False,
        "previous_worlds_revisited":False,"future_confirmation_opened":False,
        "read_gate_head_trained":False,"per_replica":per,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gathered",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    reports=[json.loads((a.gathered/f"f10d-frozen-actor-{i}"/"dev-report.json").read_text())
             for i in range(4)]
    result=aggregate(reports)
    a.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("F10D_FIXED_STRUCTURAL_TRANSFER_GATE",json.dumps({
        "status":result["status"],"gates":result["gates"],
        "replica_gate_counts":result["replica_gate_counts"],
        "replicas":{i:{
            "correct":x["correct"],"D":x["D_relational"],
            "delta_read_skip":x["read_skip_delta"],
            "delta_rel_unary":x["relational_minus_frozen_unary"],
            "delta_hard_rel_unary":x["hard_unsigned_revised_minus_unary"],
            "ci95":x["read_skip_ci95_world"],
            "gates":[x["g1_transfer_validity"],x["g2_causal_read"],x["g3_pairwise_advantage"]],
        } for i,x in result["per_replica"].items()},
    },ensure_ascii=False),flush=True)
