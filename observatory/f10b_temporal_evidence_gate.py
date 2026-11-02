"""F10-B precommitted independent DEV aggregate; Python stdlib only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CELLS=("signed|stable","unsigned|stable","signed|revised","unsigned|revised")
MODES=("read_clean","skip","read_stale","read_forged","recurrent_only")
WORLDS=64

def aggregate(reports):
    assert len(reports)==4
    replica={}
    cluster_pass=0
    independently_better=0
    veracity=0
    support_all=0
    integrity=True
    for index,report in enumerate(reports):
        assert report["phase"]=="DEV_ONLY" and report["index"]==index
        assert report["worlds"]==WORLDS and report["cases"]==WORLDS*4
        assert report["world_seeds"]==[13200000+10000*index+j for j in range(WORLDS)]
        assert report["shared_history_across_four_strata"]
        assert report["labels_from_public_signed_reports"]
        assert report["same_actor_and_hidden_for_read_skip"]
        assert report["no_confirmation_opened"] and report["no_old_worlds_reused"]
        assert report["no_reader_gate_trained"]
        assert set(report["by_cell"])==set(CELLS)
        assert len(report["rows"])==WORLDS*4
        byseed={}
        for row in report["rows"]:
            assert row["seed"] in report["world_seeds"]
            assert row["cell"] in CELLS
            assert row["truth"] in (0,1)
            assert set(row["probs"])==set(MODES)
            assert all(0<=p<=1 for p in row["probs"].values())
            assert row["D_clean"]==(
                int((row["probs"]["read_clean"]>=.5)==row["truth"])-
                int((row["probs"]["skip"]>=.5)==row["truth"]))
            byseed.setdefault(row["seed"],[]).append(row)
        assert len(byseed)==WORLDS
        assert all(len(set(v["history_sha256"] for v in rows))==1
                   and {v["cell"] for v in rows}==set(CELLS)
                   for rows in byseed.values())
        correct={mode:sum((row["probs"][mode]>=.5)==row["truth"]
                          for row in report["rows"]) for mode in MODES}
        assert all(report["overall"]["predictions"][m]["correct"]==correct[m]
                   for m in MODES)
        for cell in CELLS:
            table=report["by_cell"][cell]
            assert all(table["predictions"][mode]["cases"]==WORLDS
                       for mode in MODES)
            positives=table["D_clean"]["1"]
            negatives=table["D_clean"]["-1"]
            zeros=table["D_clean"]["0"]
            assert positives+negatives+zeros==WORLDS
        supports={cell:{
           "positive":report["by_cell"][cell]["D_clean"]["1"],
           "nonpositive":WORLDS-report["by_cell"][cell]["D_clean"]["1"],
        } for cell in CELLS}
        all_supported=all(v["positive"]>=8 and v["nonpositive"]>=8
                          for v in supports.values())
        support_all+=int(all_supported)
        ci=report["overall"]["delta_cluster_bootstrap"]
        assert ci["resampling_unit"]=="world" and ci["bootstrap_draws"]==2000
        delta=(correct["read_clean"]-correct["skip"])/(WORLDS*4)
        independent=(correct["read_clean"]-correct["recurrent_only"])/(WORLDS*4)
        assert abs(ci["mean"]-delta)<1e-6
        helpful=delta>=.10 and ci["95ci"][0]>0
        strong_independent=independent>.05
        reliable=all(report["by_cell"][name]["predictions"]["read_clean"]["accuracy"]>=.8
                     for name in CELLS)
        cluster_pass+=int(helpful)
        independently_better+=int(strong_independent)
        veracity+=int(reliable)
        replica[str(index)]={
            "n_worlds":WORLDS,"cases":WORLDS*4,
            "correct":correct,
            "delta_read_vs_skip":delta,
            "delta_read_vs_trained_recurrent":independent,
            "95ci_world_cluster":ci["95ci"],
            "causal_utility_pass":helpful,
            "trained_baseline_better_pass":strong_independent,
            "provenance_temporal_accuracy_pass":reliable,
            "all_four_cell_support_pass":all_supported,
            "strata":{cell:{
                "D_clean":report["by_cell"][cell]["D_clean"],
                "correct":{m:report["by_cell"][cell]["predictions"][m]["correct"]
                           for m in MODES},
            } for cell in CELLS},
        }
    utility=cluster_pass>=3 and independently_better>=3 and support_all==4
    source=veracity>=3
    status=("ACTOR_MEMORY_VALIDITY_DEV_PASS"
            if integrity and utility and source else "ACTOR_MEMORY_VALIDITY_DEV_FAIL")
    return {
      "experiment":"F10-B signed provenance and temporal validity",
      "phase":"DEV_ONLY",
      "status":status,
      "technical_integrity_pass":integrity,
      "causal_utility_gate_pass":utility,
      "provenance_temporal_gate_pass":source,
      "replicas_with_world_cluster_read_gain":cluster_pass,
      "replicas_better_than_trained_recurrent":independently_better,
      "replicas_with_all_four_Dplus_support_cells":support_all,
      "replicas_with_80percent_each_cell":veracity,
      "all_train_dev_worlds_new":True,
      "old_confirmations_reused":False,
      "f10b_confirm_13300000_opened":False,
      "reader_gate_trained":False,
      "per_replica":replica,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gathered",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    reports=[json.loads((args.gathered/f"f10b-actor-{i}"/"dev-report.json").read_text())
             for i in range(4)]
    result=aggregate(reports)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    brief={
      "status":result["status"],
      "causal_utility_gate_pass":result["causal_utility_gate_pass"],
      "provenance_temporal_gate_pass":result["provenance_temporal_gate_pass"],
      "replicas_with_all_four_Dplus_support_cells":result["replicas_with_all_four_Dplus_support_cells"],
      "per_replica":{k:{
          "correct":v["correct"],
          "delta_read_vs_skip":v["delta_read_vs_skip"],
          "delta_read_vs_trained_recurrent":v["delta_read_vs_trained_recurrent"],
          "95ci_world_cluster":v["95ci_world_cluster"],
          "strata":v["strata"],
          "all_four_cell_support_pass":v["all_four_cell_support_pass"],
      } for k,v in result["per_replica"].items()},
    }
    print("F10B_INDEPENDENT_DEV_GATE",json.dumps(brief,ensure_ascii=False),flush=True)

if __name__=="__main__":
    main()
