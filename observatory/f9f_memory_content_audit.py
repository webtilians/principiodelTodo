"""F9-F: audit existing F9-E real F4 Store/FIFO checkpoint trajectories.

Pure stdlib forensic analysis. NO training, inference, new worlds, new seeds
or heldout access. Uses four *already published* F9-E development artifacts.
This checks whether stored *readouts* distinguish paired changes in the
initial MiniGrid cue and at the last decision action.

The historical F9-E trace does not label before/after RL: per source code
S7/normal occurs twice (before first PPO learn, then after); S13/normal
once (after). We assert this shape and report both stages separately.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

SOURCE_RUN = 40908055330
EXPECTED_SOURCE_SEEDS = frozenset(range(4))


def readout_types(readout):
    """Semantic identities visible IN F4 slots, never world truth."""
    if len(readout) != 12:
        raise ValueError("Expected 3 F4 slots of 4 visible-symbol channels")
    out = []
    for j in range(0, 12, 4):
        k, b, _color, occupied = readout[j:j+4]
        if occupied > .5:
            if (k > .5) == (b > .5):
                raise ValueError("Occupied slot has ambiguous or absent identity")
            out.append("key" if k > .5 else "ball")
    return out


def assess_pair(record):
    """Evaluate actual F4 readout's ability to discriminate counterfactuals."""
    a, b = record["pair"]
    if a["seed"] != b["seed"] or a["cue"] == b["cue"]:
        raise AssertionError("Not a matched pair with a flipped cue")
    if a["correct_side"] == b["correct_side"]:
        raise AssertionError("Correct target did not flip")
    x, y = a["trace"], b["trace"]
    if not x or not y:
        raise AssertionError("Missing native action trace")

    f_a = next((v for v in x if v["cue_visible"]), None)
    f_b = next((v for v in y if v["cue_visible"]), None)
    saw_both = f_a is not None and f_b is not None
    if saw_both:
        first_readouts_same = f_a["memory_readout"] == f_b["memory_readout"]
        first_has_both = all(
            {"key", "ball"}.issubset(set(readout_types(t["memory_readout"])))
            for t in (f_a, f_b))
    else:
        first_readouts_same = None
        first_has_both = None

    last_same = x[-1]["memory_readout"] == y[-1]["memory_readout"]
    # A differing policy history can create differing memories even if cue
    # itself is not encoded; report exact action/pose equality explicitly.
    same_actions_and_positions = (
        [(v["action"], v["position"]) for v in x] ==
        [(v["action"], v["position"]) for v in y]
    )
    whole_store_trace_identical = (
        [v["memory_readout"] for v in x] ==
        [v["memory_readout"] for v in y]
    )
    last_contains_both = all(
        {"key", "ball"}.issubset(set(readout_types(v[-1]["memory_readout"])))
        for v in (x, y)
    )

    return {
        "seed": a["seed"],
        "cue_original": a["cue"],
        "cue_swapped": b["cue"],
        "side_original": a["side"],
        "side_swapped": b["side"],
        "both_cues_observed": saw_both,
        "same_f4_readout_at_first_cue": first_readouts_same,
        "both_object_types_in_f4_at_first_cue": first_has_both,
        "same_f4_readout_before_final_action": last_same,
        "both_types_in_f4_before_final_action": last_contains_both,
        "same_complete_f4_readout_sequence": whole_store_trace_identical,
        "same_action_and_position_sequence": same_actions_and_positions,
        "same_physical_branch": a["side"] == b["side"],
        "both_correct": bool(a["success"] and b["success"]),
        "original_correct": bool(a["success"]),
        "swapped_correct": bool(b["success"]),
        "initial_f4_original": x[0]["memory_readout"],
        "initial_f4_swapped": y[0]["memory_readout"],
        "last_f4_original": x[-1]["memory_readout"],
        "last_f4_swapped": y[-1]["memory_readout"],
    }


def summarize(pairs):
    if not pairs:
        raise ValueError("No audit pairs")
    def count(field):
        return sum(1 for row in pairs if row[field] is True)
    return {
        "paired_worlds": len(pairs),
        "both_policies_observed_cue": count("both_cues_observed"),
        "cue_readouts_identical_when_both_observed": count(
            "same_f4_readout_at_first_cue"),
        "both_object_types_present_at_first_cue": count(
            "both_object_types_in_f4_at_first_cue"),
        "last_readouts_identical": count(
            "same_f4_readout_before_final_action"),
        "both_object_types_present_at_final_action": count(
            "both_types_in_f4_before_final_action"),
        "entire_f4_readout_sequences_identical": count(
            "same_complete_f4_readout_sequence"),
        "action_and_position_sequences_identical": count(
            "same_action_and_position_sequence"),
        "same_chosen_exit": count("same_physical_branch"),
        "both_conditions_successful": count("both_correct"),
        "original_success": count("original_correct"),
        "counterfactual_success": count("swapped_correct"),
    }


def audit_seed(directory):
    directory=Path(directory)
    source=directory/"report.json"
    tracefile=directory/"trajectories.jsonl"
    report=json.loads(source.read_text())
    if report.get("experiment") != "F9-E genuine F4 Store/FIFO native navigation":
        raise AssertionError("Wrong source experiment")
    seed=int(report["seed_index"])
    if seed not in EXPECTED_SOURCE_SEEDS:
        raise AssertionError("Unexpected training seed")
    if report["heldout_opened"] is not False or report["shaping"] is not False:
        raise AssertionError("Source experiment did not preserve test boundary")
    models=report["models"]
    if set(models) != {"raw","zero","f4"}:
        raise AssertionError("Missing original three-policy control")
    if models["zero"]["input_width"] != models["f4"]["input_width"]:
        raise AssertionError("F4 versus zero has unequal input width")
    if models["zero"]["parameters"] != models["f4"]["parameters"]:
        raise AssertionError("F4 versus zero has unequal parameter count")
    # Verify historical checkpoint bytes when artifacts provide them.
    checkpoint_hashes = {}
    for filename, field in (("f4_pre_rl.zip", "pre_rl_sha256"),
                            ("f4_post_rl.zip", "post_rl_sha256")):
        path=directory/filename
        if not path.exists():
            raise FileNotFoundError(path)
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != models["f4"][field]:
            raise AssertionError("F4 historical checkpoint SHA mismatch")
        checkpoint_hashes[filename]=sha

    rows=[json.loads(line) for line in tracefile.read_text().splitlines()]
    n=int(report["config"]["evaluation_pairs"])
    # Historical source emits S7 normal twice, BC first and post-RL second,
    # and S13 normal only once after RL. Enforce exact cardinality.
    f7=[r for r in rows if r["mode"]=="f4" and r["size"]==7
        and r["control"]=="normal" and r["index"]==seed]
    f13=[r for r in rows if r["mode"]=="f4" and r["size"]==13
         and r["control"]=="normal" and r["index"]==seed]
    if len(f7) != 2*n or len(f13) != n:
        raise AssertionError(
            f"Historical F9-E stage order unsupported: S7={len(f7)}, S13={len(f13)}, expected {2*n}/{n}"
        )
    groups={"S7_before_rl":f7[:n],
            "S7_after_rl":f7[n:],
            "S13_zero_shot_after_rl":f13}
    summarized = {}
    paired_records = {}
    for stage, records in groups.items():
        results=[assess_pair(record) for record in records]
        summarized[stage]=summarize(results)
        paired_records[stage]=results
    return {
        "seed_index":seed,
        "source_report_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "source_trajectory_sha256":hashlib.sha256(tracefile.read_bytes()).hexdigest(),
        "frozen_f4_checkpoints":checkpoint_hashes,
        "stages":summarized,
        "pairs":paired_records,
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    archives=sorted(a.input.glob("f9e-f4-development-seed-*"))
    if len(archives)!=4:
        raise AssertionError(f"Expected all 4 original F9-E artifacts: {archives}")
    records=[audit_seed(path) for path in archives]
    if {x["seed_index"] for x in records} != EXPECTED_SOURCE_SEEDS:
        raise AssertionError("Not four independent source training seed indices")

    stages=("S7_before_rl","S7_after_rl","S13_zero_shot_after_rl")
    totals={}
    for stage in stages:
        names=records[0]["stages"][stage]
        totals[stage]={
            k:sum(int(r["stages"][stage][k]) for r in records)
            for k in names
        }
        n=totals[stage]["paired_worlds"]
        print("F9F_MEMORY_CONTENT",stage,f"{n} paired worlds",
              json.dumps(totals[stage],sort_keys=True),flush=True)

    output={
        "experiment":"F9-F retrospective forensic audit of F9-E F4 FIFO readout",
        "phase":"development_reanalysis_no_training",
        "source_workflow_run":SOURCE_RUN,
        "source_branch":"experiment/f9e-f4-autonomous",
        "heldout_opened":False,
        "training_executed":False,
        "inference_executed":False,
        "claim_scope":(
            "A readout that is identical within cue-inverted pairs cannot "
            "encode which cue member was shown for those same paired histories. "
            "The actor still receives raw visual frames and recurrent state. "
            "No conclusion is drawn about F4 LearnedRelevance, F5 or any "
            "unimplemented role-aware memory."
        ),
        "stage_counts":totals,
        "by_training_seed":records,
    }
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/"report.json").write_text(json.dumps(output,indent=2),encoding="utf8")
    with (a.output/"counterfactual_pairs.jsonl").open("w") as stream:
        for record in records:
            for stage,rows in record["pairs"].items():
                for row in rows:
                    stream.write(json.dumps(
                        {"training_seed":record["seed_index"],
                         "stage":stage, **row})+"\n")
    print("F9F_FINAL",json.dumps(totals),flush=True)


if __name__=="__main__":
    main()
