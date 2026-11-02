"""F9-Y preregistered independent DEV aggregation and support-aware gate."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

CELL_COUNT=4
REPLICAS=4
DEV_WORLDS=32
BOOT=2000
BOOT_SEED=20261102
COMPARE=("loco_hc","loco_nuisance","loco_hc_shuffle_HC")


def bce(y,p):
    p=min(max(float(p),1e-7),1.-1e-7)
    return -math.log(p) if y else -math.log(1-p)


def prepare(reports):
    if len(reports)!=REPLICAS:raise AssertionError("Four original actor sources required")
    for i,r in enumerate(reports):
        assert r["phase"]=="TRAIN_DEV_ONLY" and r["index"]==i
        assert r["train_worlds"]==128 and r["dev_worlds"]==32
        assert len(r["train_cases"])==512 and len(r["dev_cases"])==128
        assert len(r["predictions"])==128
        assert len(r["new_train_seeds"])==128 and len(r["new_dev_seeds"])==32
        for name in ("f9v_8300000_reused","f9w_9300000_opened","f9x_9600000_opened",
                     "f9y_10300000_opened","f9t_6300000_opened","f9u_7300000_opened",
                     "historic_900000_opened","heads_trained_on_DEV",
                     "actor_retrained","F4_writer_changed"):
            assert r[name] is False,(i,name)
        for j in range(DEV_WORLDS):
            block=r["dev_cases"][4*j:4*(j+1)]
            assert {x["seed"] for x in block}=={10_200_000+i*10_000+j}
            assert {(
                x["cue"], x["regime"]) for x in block}=={
                    ("key","history_f4_normal"),
                    ("ball","history_f4_normal"),
                    ("key","history_f4_postcue_off"),
                    ("ball","history_f4_postcue_off")}
    return reports


def cell_idx(c):
    return [("key","history_f4_normal"),
            ("ball","history_f4_normal"),
            ("key","history_f4_postcue_off"),
            ("ball","history_f4_postcue_off")].index((c["cue"],c["regime"]))


def summary_metric(reports,method,which=None,actor=None):
    items=[]
    for r in reports:
        if actor is not None and r["index"]!=actor:continue
        for case,pred in zip(r["dev_cases"],r["predictions"]):
            if which is not None and cell_idx(case) not in which:continue
            items.append((int(case["effect"]==1), pred[method]))
    if not items:return None
    return {"n":len(items),"positive":sum(y for y,_ in items),
            "logloss":sum(bce(y,p) for y,p in items)/len(items),
            "reads":sum(p>=.5 for _,p in items),
            "mean_probability":sum(p for _,p in items)/len(items)}


def paired_differences(reports,cells,actor=None):
    data=[]
    for r in reports:
        if actor is not None and actor!=r["index"]:continue
        for c,p in zip(r["dev_cases"],r["predictions"]):
            if cell_idx(c) not in cells:continue
            y=int(c["effect"]==1)
            data.append((bce(y,p["loco_nuisance"])-bce(y,p["loco_hc"]),
                         bce(y,p["loco_hc_shuffle_HC"])-bce(y,p["loco_hc"])))
    if not data:return None
    return {"cases":len(data),
            "nuisance_minus_hc":sum(x[0] for x in data)/len(data),
            "shuffled_minus_hc":sum(x[1] for x in data)/len(data)}


def boot_ci(reports,cells):
    if not cells:return None
    blocks=[]
    for r in reports:
        actor_blocks=[]
        for j in range(DEV_WORLDS):
            deltas=[]
            for c,p in zip(r["dev_cases"][4*j:4*j+4],
                           r["predictions"][4*j:4*j+4]):
                if cell_idx(c) not in cells:continue
                y=int(c["effect"]==1)
                deltas.append((bce(y,p["loco_nuisance"])-bce(y,p["loco_hc"]),
                               bce(y,p["loco_hc_shuffle_HC"])-bce(y,p["loco_hc"])))
            assert len(deltas)==len(cells)
            actor_blocks.append(deltas)
        blocks.append(actor_blocks)
    rng=random.Random(BOOT_SEED)
    dist=[[],[]]
    for _ in range(BOOT):
        chosen=[rng.choice(blocks[actor]) for actor in range(REPLICAS)
                for _j in range(DEV_WORLDS)]
        for col in range(2):
            vals=[v[col] for world in chosen for v in world]
            dist[col].append(sum(vals)/len(vals))
    intervals=[]
    for draws in dist:
        draws.sort()
        intervals.append([draws[49],draws[1949]])
    return {"n_bootstrap":BOOT,"unit":"whole-world 4-case cluster, stratified by actor",
            "nuisance_minus_hc_95pct":intervals[0],
            "shuffled_minus_hc_95pct":intervals[1]}


def aggregate(reports):
    prepare(reports)
    support={}
    accepted=[]
    for cell in range(CELL_COUNT):
        counts=[sum(c["effect"]==1 for c in r["dev_cases"] if cell_idx(c)==cell)
                for r in reports]
        neg=[DEV_WORLDS-v for v in counts]
        valid=sum(counts)>=4 and sum(neg)>=4 and sum(v>=2 for v in counts)>=2
        support[str(cell)]={
            "positive_by_replica":counts,
            "negative_by_replica":neg,
            "total_positive":sum(counts),"total_negative":sum(neg),
            "actors_with_at_least_2_positives":sum(v>=2 for v in counts),
            "supported":valid,
            "loco_hc":summary_metric(reports,"loco_hc",{cell}),
            "loco_nuisance":summary_metric(reports,"loco_nuisance",{cell}),
            "delta":paired_differences(reports,{cell}),
        }
        if valid:accepted.append(cell)
    by_actor={str(i):paired_differences(reports,set(accepted),actor=i)
              for i in range(REPLICAS)}
    overall=paired_differences(reports,set(accepted))
    ci=boot_ci(reports,set(accepted))
    by_cell_positive=sum(support[str(i)]["delta"]["nuisance_minus_hc"]>0
                         for i in accepted)
    by_actor_positive=sum(v and v["nuisance_minus_hc"]>0
                          for v in by_actor.values())
    by_actor_shuffle=sum(v and v["shuffled_minus_hc"]>0
                         for v in by_actor.values())
    if len(accepted)<2:
        status="INCONCLUSIVE_SUPPORT"
    else:
        success=(
            overall["nuisance_minus_hc"]>=.02
            and ci["nuisance_minus_hc_95pct"][0]>0
            and by_actor_positive>=3 and by_cell_positive>=2
            and overall["shuffled_minus_hc"]>=.02
            and by_actor_shuffle>=3)
        status="PASS_DEV" if success else "FAIL_DEV"
    return {
        "experiment":"F9-Y conditional H/C predictive value and LOCO generalization",
        "phase":"DEV_ONLY_NO_CONFIRM",
        "actors":REPLICAS,"train_worlds_per_actor":128,
        "dev_worlds_per_actor":DEV_WORLDS,"dev_cases_total":512,
        "cell_support":support,
        "eligible_cells":accepted,
        "eval_on_supported_cells":overall,
        "per_actor_on_supported":by_actor,
        "bootstrap_95pct":ci,
        "preregistered_gate":{
            "status":status,
            "min_two_supported_cells":len(accepted)>=2,
            "delta_at_least_0_02_nats":bool(overall and overall["nuisance_minus_hc"]>=.02),
            "delta_CI_lower_above_0":bool(ci and ci["nuisance_minus_hc_95pct"][0]>0),
            "at_least_3_actors_gain":by_actor_positive>=3,
            "at_least_2_cells_gain":by_cell_positive>=2,
            "within_cell_HC_shuffle_degrades_at_least_0_02":bool(overall and overall["shuffled_minus_hc"]>=.02),
            "at_least_3_actors_shuffle_degrades":by_actor_shuffle>=3,
            "f9y_confirm_10300000_opened":False
        },
        "natural_full_dev_all_strata":{
            name:summary_metric(reports,name)
            for name in ("deploy_hc","deploy_public","controlled_hc",
                         "controlled_nuisance","loco_hc","loco_nuisance",
                         "loco_hc_shuffle_HC","loco_hc_shuffle_H",
                         "loco_hc_shuffle_C")},
        "by_replica":{
            str(r["index"]):{
                "train_counts":r["train_counts_by_cell"],
                "dev_by_cell":{i:{"positive":v["positive"],"negative":v["negative"],
                                  "loco_hc_logloss":v["models"]["loco_hc"]["logloss"],
                                  "loco_nuisance_logloss":v["models"]["loco_nuisance"]["logloss"],
                                  "loco_hc_reads":v["models"]["loco_hc"]["reads"],
                                  "loco_hc_auc":v["models"]["loco_hc"]["auc"]}
                               for i,v in r["dev_by_cell"].items()},
            } for r in reports},
        "old_holdouts_reused":False,
    }


def main():
    p=__import__("argparse").ArgumentParser()
    p.add_argument("--gathered",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    reports=[json.loads((args.gathered / f"f9y-train-dev-seed-{i}" / "report.json").read_text())
             for i in range(REPLICAS)]
    result=aggregate(reports)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print("F9Y_INDEPENDENT_DEV_GATE",json.dumps(result,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
