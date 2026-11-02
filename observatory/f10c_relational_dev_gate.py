"""F10-C preregistered scientific gate, using stdlib only (no model/datasets)."""
from __future__ import annotations
import argparse,json
from pathlib import Path
CELLS=("signed|stable","unsigned|stable","signed|revised","unsigned|revised")
MODES=("read_clean","skip","read_stale","read_forged",
       "unary_retrained","recurrent_retrained",
       "f10b_frozen_read","f10b_frozen_recurrent")
N=64

def aggregate(reports):
    assert len(reports)==4
    result={}
    accuracy_count=0
    head2head_count=0
    causal_count=0
    positive_count=0
    for index,d in enumerate(reports):
        assert d["index"]==index and d["phase"]=="DEV_ONLY"
        assert d["worlds"]==N and d["cases"]==4*N
        assert d["dev_world_seeds"]==[14200000+index*10000+j for j in range(N)]
        assert d["f10b_weights_frozen"] and not d["f10b_worlds_reused"]
        assert not d["f10c_confirm_14300000_opened"] and not d["read_head_trained"]
        assert d["same_public_and_recurrent_before_read_intervention"]
        assert d["labels_derived_from_public_signed_records"] and d["full_four_query_histories_matched"]
        assert len(d["rows"])==4*N and set(d["by_cell"])==set(CELLS)
        histories={}
        for c in d["rows"]:
            assert c["seed"] in d["dev_world_seeds"] and c["cell"] in CELLS
            assert c["truth"] in (0,1)
            assert c["matching_reports"]==5 and c["contradictory_reports"]>=1
            assert set(c["p"])==set(MODES)
            assert all(0<=v<=1 for v in c["p"].values())
            histories.setdefault(c["seed"],[]).append(c)
        assert len(histories)==N and all(len(v)==4 and
                {x["cell"] for x in v}==set(CELLS) and
                len({x["history_sha256"] for x in v})==1 for v in histories.values())
        correct={name:sum((r["p"][name]>=.5)==r["truth"] for r in d["rows"])
                 for name in MODES}
        assert d["overall"]["correct"]==correct
        strata={}
        for cell in CELLS:
            cur=d["by_cell"][cell]
            records=[r for r in d["rows"] if r["cell"]==cell]
            assert len(records)==N and cur["cases"]==N
            for name in MODES:
                assert cur["correct"][name]==sum((r["p"][name]>=.5)==r["truth"]
                                                 for r in records)
            diffs=[int((r["p"]["read_clean"]>=.5)==r["truth"])-
                   int((r["p"]["skip"]>=.5)==r["truth"])
                   for r in records]
            expected={str(j):sum(x==j for x in diffs) for j in (-1,0,1)}
            assert cur["D_clean"]==expected
            strata[cell]={
                "read_correct":cur["correct"]["read_clean"],
                "unary_correct":cur["correct"]["unary_retrained"],
                "baseline_frozen_read":cur["correct"]["f10b_frozen_read"],
                "positive":expected["1"],"neutral":expected["0"],
                "harmful":expected["-1"],
                "support":expected["1"]>=8 and N-expected["1"]>=8,
            }
        all_support=all(x["support"] for x in strata.values())
        ci=d["overall"]["read_skip_bootstrap"]
        assert ci["draws"]==2000 and ci["unit"]=="world"
        gain=(correct["read_clean"]-correct["skip"])/(N*4)
        assert abs(gain-ci["mean"])<1e-7
        gain_recurrent=(correct["read_clean"]-correct["recurrent_retrained"])/(N*4)
        gain_unary=(correct["read_clean"]-correct["unary_retrained"])/(N*4)
        gain_hard=(strata["unsigned|revised"]["read_correct"]-
                   strata["unsigned|revised"]["unary_correct"])/N
        pass_accuracy=all(x["read_correct"]/N>=.80 for x in strata.values())
        pass_comparators=gain_unary>=.05 and gain_hard>=.08
        pass_causal=gain>=.10 and ci["ci95"][0]>0 and gain_recurrent>=.05
        accuracy_count+=int(pass_accuracy)
        head2head_count+=int(pass_comparators)
        causal_count+=int(pass_causal)
        positive_count+=int(all_support)
        result[str(index)]={
            "correct":correct,
            "delta_read_minus_skip":gain,
            "delta_read_minus_new_unary":gain_unary,
            "delta_read_minus_new_gru":gain_recurrent,
            "delta_unsigned_revised_minus_unary":gain_hard,
            "bootstrap_ci95":ci["ci95"],
            "accuracy_all_cells_pass":pass_accuracy,
            "comparators_pass":pass_comparators,
            "causal_utility_pass":pass_causal,
            "all_cells_support_pass":all_support,
            "cells":strata,
            "f10b_frozen_shas":d["source"]["f10b_frozen_checkpoint_shas"],
        }
    criteria={
        "provenance_temporal":accuracy_count>=3,
        "relational_vs_unary":head2head_count>=3,
        "causal_memory":causal_count>=3,
        "positivity_16_cells":positive_count==4,
    }
    return {
        "experiment":"F10-C multi-version relational evidence arbitration",
        "stage":"DEV_ONLY",
        "status":"F10C_DEV_PASS" if all(criteria.values()) else "F10C_DEV_FAIL",
        "gates":criteria,
        "counts":{
            "accuracy_80pct_each_cell_replicas":accuracy_count,
            "outperform_new_unary_replicas":head2head_count,
            "causal_read_replicas":causal_count,
            "fully_supported_replicas":positive_count,
        },
        "technical_integrity":True,
        "prior_f10b_frozen":True,"prior_f10b_data_reused":False,
        "f10c_confirm_14300000_opened":False,
        "head_trained":False,
        "per_replica":result,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gathered",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    reports=[json.loads((args.gathered/f"f10c-actor-{i}"/"dev-report.json").read_text())
             for i in range(4)]
    r=aggregate(reports)
    args.output.write_text(json.dumps(r,indent=2,ensure_ascii=False))
    brief={"status":r["status"],"gates":r["gates"],"counts":r["counts"],
           "actors":{k:{
               "correct":v["correct"],"delta_vs_skip":v["delta_read_minus_skip"],
               "delta_vs_unary":v["delta_read_minus_new_unary"],
               "delta_hard_vs_unary":v["delta_unsigned_revised_minus_unary"],
               "world_ci95":v["bootstrap_ci95"],
               "cells":v["cells"]} for k,v in r["per_replica"].items()}}
    print("F10C_INDEPENDENT_SCIENTIFIC_GATE",json.dumps(brief,ensure_ascii=False),flush=True)
