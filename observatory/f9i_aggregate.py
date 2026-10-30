"""Validate every preassigned cell, summarize per-seed results without selection."""
import json
import os
import statistics
from pathlib import Path

root=Path("archives")
reports=[json.loads(p.read_text()) for p in root.glob("f9i-[0-3]-*/report.json")]
assert len(reports)==8
assert {(r["seed_index"],r["event_capacity"]) for r in reports} == {(i,c) for i in range(4) for c in (8,64)}
lines=["# F9-I development results","",
       "4 training seeds, 10 paired worlds/size/seed. S13 zero-shot. No heldout.",
       "All rows after PPO. Branch counts use 20 episodes, not 20 independent training seeds.","",
       "|Capacity|Seed|Policy / control|S7 success|S13 success|S13 both correct|S13 branch reached|S13 initial retained at branch|S13 any cue retained at branch|Corruption coverage S13|",
       "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in sorted(reports,key=lambda r:(r["event_capacity"],r["seed_index"])):
    assert not r["heldout_opened"] and not r["teleport"] and not r["shaping"]
    assert r["native_actions"]==7
    if os.environ.get("EXPECTED_SOURCE_SHA"):
        assert r["git_sha"]==os.environ["EXPECTED_SOURCE_SHA"]
    assert set(r["models"])=={"zero","fifo","context","random"}
    for field in ("parameters","input_width","actual_rl_steps"):
        assert len({v[field] for v in r["models"].values()})==1
    assert r["models"]["fifo"]["input_width"]==660
    for mode,m in r["models"].items():
        for control in (("normal","mask_cue") if mode=="zero" else ("normal","erase","corrupt_initial","mask_cue")):
            a=m["after_rl"]["S7/"+control]; b=m["after_rl"]["S13/"+control]
            lines.append(f'|{r["event_capacity"]}|{r["seed_index"]}|{mode}/{control}|{a["combined_success"]:.1%}|{b["combined_success"]:.1%}|{b["both_correct"]}/10|{b["fork_reached_rate"]:.0%}|{b["initial_retained_at_fork_count"]}/20|{b["any_cue_retained_at_fork_count"]}/20|{b["corruption_applied_episodes"]}/20|')
diagnoses=[json.loads(p.read_text()) for p in root.glob("f9i-original-diagnosis-*/diagnosis.json")]
assert len(diagnoses)==4 and {d["seed_index"] for d in diagnoses}=={0,1,2,3}
expected={7:[.8,1.,1.,1.],13:[.5,.25,.45,.35]}
lines += ["", "## Original F9-H checkpoint diagnosis", "",
          "|Seed|Size|Success|Reached branch|First written at branch|First retained at branch|Any cue retained at branch|Corruption applied|",
          "|---|---|---:|---:|---:|---:|---:|---:|"]
for d in sorted(diagnoses,key=lambda d:d["seed_index"]):
    assert not d["heldout_opened"]
    for size in (7,13):
        m=d["metrics"][f"S{size}/normal"]
        assert abs(m["combined_success"]-expected[size][d["seed_index"]])<1e-9, "F9-H checkpoint reproduction mismatch"
        c=d["metrics"][f"S{size}/corrupt_initial"]
        lines.append(f'|{d["seed_index"]}|S{size}|{m["combined_success"]:.1%}|{m["fork_reached_rate"]:.0%}|{m["initial_written_at_fork_count"]}|{m["initial_retained_at_fork_count"]}|{m["any_cue_retained_at_fork_count"]}|{c["corruption_applied_episodes"]}/20|')
    print("F9I_ORIGINAL_DIAGNOSIS",json.dumps(d))
evictions={}
for path in sorted(root.glob("f9i-original-diagnosis-*/traces.jsonl")):
    for line in path.read_text().splitlines():
        row=json.loads(line)
        if row["intervention"]!="normal":
            continue
        key=row["environment"]
        out=evictions.setdefault(key,{"episodes":0,"initial_ever_lost":0,
            "all_cue_ever_lost":0,"initial_lost_without_reaching_branch":0})
        for ep in row["paired_rollouts"]:
            out["episodes"]+=1
            initial_lost=any(t["initial_written"] and not t["initial_retained"] for t in ep["trace"])
            all_lost=any(t["initial_written"] and not t["any_cue_retained"] for t in ep["trace"])
            out["initial_ever_lost"]+=int(initial_lost)
            out["all_cue_ever_lost"]+=int(all_lost)
            out["initial_lost_without_reaching_branch"]+=int(initial_lost and ep["fork_audit"] is None)
assert evictions["S7"]["episodes"]==evictions["S13"]["episodes"]==80
lines += ["","Original-checkpoint full-trajectory loss counts (timing/causality not inferred):",json.dumps(evictions,indent=2)]
print("F9I_EVICTION_AUDIT",json.dumps(evictions))
deltas={}
for cap in (8,64):
    rs=sorted([r for r in reports if r["event_capacity"]==cap],key=lambda r:r["seed_index"])
    for size in (7,13):
        for mode in ("context","random","zero"):
            values=[r["models"][mode]["after_rl"][f"S{size}/normal"]["combined_success"]-
                    r["models"]["fifo"]["after_rl"][f"S{size}/normal"]["combined_success"] for r in rs]
            deltas[f"{cap}/S{size}/{mode}-fifo"]={"per_training_seed":values,"mean":statistics.mean(values)}
lines+=["","## Paired success differences", "",json.dumps(deltas,indent=2),
        "","Retention and success must be interpreted separately; corruption no-ops are not causal null results.",
        "The 95% F9-H S7 result does not transfer to S13."]
summary={"heldout_opened":False,"reports":reports,"original_checkpoint_diagnoses":diagnoses,"original_eviction_counts":evictions,"paired_deltas":deltas}
Path("f9i_results.json").write_text(json.dumps(summary,indent=2))
Path("f9i_results.md").write_text("\n".join(lines))
Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text("\n".join(lines))
print("F9I_PAIRED_DELTAS",json.dumps(deltas))
for r in reports:
    print("F9I_SUMMARY",json.dumps({"index":r["seed_index"],"capacity":r["event_capacity"],
        "models":{k:v["after_rl"] for k,v in r["models"].items()}}))
