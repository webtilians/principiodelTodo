"""F10-A: preregistered new-actor robustness and causal support DEV gate (stdlib only)."""
import argparse,json
from pathlib import Path

LABELS=("f9o_original_bc","evidence_mixed","evidence_clean_control")
CELLS=("key|history_f4_normal","ball|history_f4_normal",
       "key|history_f4_postcue_off","ball|history_f4_postcue_off")
MODES=("skip","read_clean","read_adversarial")

def aggregate(reports):
    assert len(reports)==4
    table={}
    robust=0
    positivity=0
    all_contracts=True
    for index,report in enumerate(reports):
        assert report["index"]==index and report["phase"]=="DEV_ONLY"
        assert report["dev_worlds"]==32 and report["new_train_worlds"]==128
        assert report["new_dev_seeds"]==[12200000+index*10000+j for j in range(32)]
        assert report["actor_retrained"] and not report["reader_trained"]
        assert not report["previous_confirmations_reused"] and not report["future_confirm_12300000_opened"]
        assert set(report["actors"])==set(LABELS)
        actors=report["actors"]
        summary={}
        for label in LABELS:
            a=actors[label]
            assert len(a["states"])==128 and len(a["autonomous"])==128
            assert set(a["by_cell"])==set(CELLS)
            assert a["all_states"]["states"]==128
            for cell in CELLS:
                part=a["by_cell"][cell]
                assert part["states"]==32
                for effect in ("D_clean","D_adversarial"):
                    assert sum(part[effect].values())==32
                assert all(v["seed"] in report["new_dev_seeds"] for v in a["states"])
            summary[label]={
                "first_correct":a["all_states"]["first_correct"],
                "D_clean":a["all_states"]["D_clean"],
                "D_adversarial":a["all_states"]["D_adversarial"],
                "autonomous":a["autonomous_summary"],
                "cells":{
                    c:{"D_clean":a["by_cell"][c]["D_clean"],
                       "D_adversarial":a["by_cell"][c]["D_adversarial"]}
                    for c in CELLS},
            }
        mixed=actors["evidence_mixed"]["all_states"]["first_correct"]
        control=actors["evidence_clean_control"]["all_states"]["first_correct"]
        clean_diff=mixed["read_clean"]-control["read_clean"]
        false_diff=mixed["read_adversarial"]-control["read_adversarial"]
        actor_robust=clean_diff>=0 and false_diff>=5
        robust+=int(actor_robust)
        support={c:{
            "D_positive":actors["evidence_mixed"]["by_cell"][c]["D_clean"]["1"],
            "D_nonpositive":32-actors["evidence_mixed"]["by_cell"][c]["D_clean"]["1"],
        } for c in CELLS}
        actor_support=all(x["D_positive"]>=4 and x["D_nonpositive"]>=4
                          for x in support.values())
        positivity+=int(actor_support)
        summary["gate"]={
            "clean_first_correct_delta":clean_diff,
            "adversarial_first_correct_delta":false_diff,
            "actor_robust":actor_robust,
            "actor_support_all_4_cells":actor_support,
            "support_cells":support}
        table[str(index)]=summary
    robust_pass=robust>=3
    positive_pass=positivity==4
    status=("ACTOR_DEV_FEASIBLE" if robust_pass and positive_pass else
            "EVIDENCE_ROBUSTNESS_NOT_ESTABLISHED_AND_INSUFFICIENT_POSITIVITY"
            if not robust_pass and not positive_pass else
            "EVIDENCE_ROBUSTNESS_NOT_ESTABLISHED" if not robust_pass else
            "INSUFFICIENT_POSITIVITY")
    return {
        "experiment":"INFINITO F10-A new-generation actor robustness and causal READ support",
        "status":status,"robust_replicas":robust,
        "positivity_replicas_all_cells":positivity,
        "robustness_gate_pass":robust_pass,
        "positivity_gate_pass":positive_pass,
        "preregistered_robustness_3_of_4":True,
        "preregistered_positivity_16_of_16":True,
        "development_only":True,"no_reader_training":True,
        "future_confirm_12300000_opened":False,
        "per_replica":table,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gathered",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    reports=[json.loads((args.gathered/f"f10a-train-dev-seed-{i}"/"dev-report.json").read_text())
             for i in range(4)]
    result=aggregate(reports)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    compact={"status":result["status"],
             "robust_replicas":result["robust_replicas"],
             "positivity_replicas_all_cells":result["positivity_replicas_all_cells"],
             "replicas":{k:v["gate"] for k,v in result["per_replica"].items()}}
    print("F10A_PREREGISTERED_DEV_GATE",json.dumps(compact,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
