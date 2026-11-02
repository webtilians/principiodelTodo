"""F9-Z preregistered report: all actors, all cells, one common arm or STOP.

Support and causal value are computed from the complete, never-filtered pilot.
This deliberately does not train a head or open TRAIN/DEV/CONFIRM worlds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from observatory.f9z_causal_support_pilot import (
    BASE, WORLDS, ORDER, CELLS, seed_for, summarize,
)


def aggregate(reports):
    assert len(reports)==4
    for i,report in enumerate(reports):
        assert report["index"]==i and report["worlds"]==WORLDS
        assert report["stage"]=="PILOT_NOT_TRAIN_DEV"
        assert report["cases_count"]==WORLDS*36
        assert len(report["cases"])==WORLDS*36
        assert report["world_seeds"]==[seed_for(i,j) for j in range(WORLDS)]
        assert report["full_factorial_3x3"] is True
        assert report["anchor_original_f9v_w1_n8_passed"] is True
        assert report["skip_invariant_across_read_window"] is True
        for flag in ("actor_retrained","head_trained","F4_algorithm_modified",
                     "prior_holdouts_reused","future_F9Z_train_dev_confirm_opened"):
            assert report[flag] is False, (i, flag)
        for case in report["cases"]:
            assert case["actor_index"]==i
            assert case["seed"] in report["world_seeds"]
            assert case["public_equal_between_cues"] is True
            assert case["same_start_hc"] is True
            assert case["capacity"] in (1,4,8) and case["window"] in (1,4,8)
            assert (case["cue"],case["regime"]) in CELLS
            assert case["effect"]==case["read_branch"]["first_arm_correct"]-case["skip_branch"]["first_arm_correct"]
        assert summarize(report["cases"])==report["summary"]
    treatments={}
    eligible=[]
    for n,w in ORDER:
        name=f"N{n}_W{w}"
        actors=[]
        supported=0
        anypos=0
        total_counts={"-1":0,"0":0,"1":0}
        for r in reports:
            bycue=r["summary"][name]
            thisactor={"index":r["index"],"cells":{}}
            for cue,history in CELLS:
                k=f"{cue}|{history}"
                record=bycue[k]
                assert record["worlds"]==WORLDS
                e=record["effects"]
                assert sum(e.values())==WORLDS
                for label in total_counts:
                    total_counts[label]+=e[label]
                p,q=record["positive"],record["nonpositive"]
                good=p>=3 and q>=3
                supported+=int(good)
                anypos+=int(p>0)
                thisactor["cells"][k]={"positive":p,
                    "neutral":e["0"],"harmful":e["-1"],
                    "nonpositive":q,"supported":good,
                    "readout_nonempty":record["nonempty_readout"],
                    "read_correct":record["read_correct"],
                    "skip_correct":record["skip_correct"]}
            actors.append(thisactor)
        common=supported==16
        if common:eligible.append(name)
        treatments[name]={
            "N":n,"window":w,
            "total_cases":WORLDS*4*4,
            "effect_total":total_counts,
            "cells_with_at_least_one_positive":anypos,
            "cells_with_3_positive_and_3_nonpositive":supported,
            "common_arm_full_16_cell_support":common,
            "per_actor":actors,
        }
    status="SUPPORT_FEASIBLE_EXPLORATORY" if eligible else "SUPPORT_NOT_ACHIEVED"
    return {
        "experiment":"F9-Z-A full-factorial prospective causal positivity",
        "stage":"NEW_PILOT_ONLY",
        "original_actor_replicas":4,
        "worlds_per_actor":WORLDS,
        "unique_worlds":WORLDS*4,
        "full_factorial_cases":WORLDS*4*4*9,
        "paired_READ_SKIP_continuations":WORLDS*4*4*9*2,
        "fixed_order":[f"N{n}_W{w}" for n,w in ORDER],
        "status":status,
        "eligible_common_treatments":eligible,
        "preregistered_contracts_all_passed":True,
        "treatments":treatments,
        "new_future_train_dev_confirm_opened":False,
        "old_confirm_reused":False,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--gathered",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    reports=[json.loads((args.gathered/f"f9z-pilot-seed-{i}"/"report.json").read_text())
             for i in range(4)]
    result=aggregate(reports)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    summary={
      "status":result["status"],
      "eligible_common_treatments":result["eligible_common_treatments"],
      "treatments":{
          name:{
            "cells_with_3pos_3nonpos":v["cells_with_3_positive_and_3_nonpositive"],
            "cells_with_at_least_one_positive":v["cells_with_at_least_one_positive"],
            "effect_total":v["effect_total"],
            "actor_cell_positives":[{
                "index":r["index"],
                "key_normal":r["cells"]["key|history_f4_normal"]["positive"],
                "ball_normal":r["cells"]["ball|history_f4_normal"]["positive"],
                "key_blackout":r["cells"]["key|history_f4_postcue_off"]["positive"],
                "ball_blackout":r["cells"]["ball|history_f4_postcue_off"]["positive"],
            } for r in v["per_actor"]],
          } for name,v in result["treatments"].items()
      },
    }
    print("F9Z_PILOT_PREREGISTERED_GATE",json.dumps(summary,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
