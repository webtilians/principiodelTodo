"""F10-P independent data-only audit: NO torch/model/world imports, no training.

Reconstruct every learned event counter from public attempted-program logs without
private traces, then verify original SHA and preregistered 2k base-world
bootstrap gates, including negative controls.
"""
from __future__ import annotations
import argparse,gzip,hashlib,itertools,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

OPS=("P","S","R","X")
PROGRAMS=((),)+tuple(seq for n in (1,2,3)
                    for seq in itertools.product(OPS,repeat=n))
ARMS=("E100","E50","E0","E_noX","E_shuffleX",
      "B_reward","I_privileged")
EVENT_ARMS=ARMS[:5]
MODES=("normal","shuffle","x_disabled","public_randomized")
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCES=("iid","signed_source0","signed_source1","alternating")
CONTEXTS=(("PUBLIC",0),("PUBLIC",1),("MEMORY",0))
assert len(PROGRAMS)==85

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def accuracy(rows,mode="normal"):
    assert len(rows)
    return sum(r["outcomes"][mode]["success"] for r in rows)/len(rows)

def grouped_mean_ci(rows,rep):
    grouped=defaultdict(list)
    for r in rows:grouped[r["seed"]].append(r["outcomes"]["normal"]["success"])
    assert len(grouped)==128 and all(len(v)==4 for v in grouped.values())
    v=np.array([np.mean(x) for _,x in sorted(grouped.items())])
    rng=np.random.default_rng(291000+100*rep)
    boot=v[rng.integers(0,len(v),size=(2000,len(v)))].mean(axis=1)
    return [float(x) for x in np.quantile(boot,(.025,.975))]

def paired_contrast(left,right,rep,offset,right_mode="normal"):
    a={(r["seed"],r["variant"]):r for r in left}
    b={(r["seed"],r["variant"]):r for r in right}
    assert a.keys()==b.keys() and len(a)==512
    seeds=sorted({k[0] for k in a})
    diffs=np.array([
        sum(a[(seed,v)]["outcomes"]["normal"]["success"]-
            b[(seed,v)]["outcomes"][right_mode]["success"]
            for v in VARIANTS)/4 for seed in seeds])
    rng=np.random.default_rng(291000+100*rep+offset)
    boot=diffs[rng.integers(0,128,size=(2000,128))].mean(axis=1)
    return {"estimate":float(diffs.mean()),
            "ci95":[float(x) for x in np.quantile(boot,(.025,.975))],
            "n_base_worlds":128,"bootstrap_unit":"base_world_four_variants"}

def reconstruct_public_memory(trials,arm):
    p={0:Counter(),1:Counter()}
    x={0:Counter(),1:Counter()}
    s_ack=0;r_valid=0;r_error=0;events_seen=0
    for row in trials:
        pub=int(row["public_bit"])
        events=row["projected_events"][arm]
        last=None
        saw_ack=False
        for e in events:
            op=e["op"];output=e["output"]
            if output is None:
                if op=="S":saw_ack=False
                if op in ("P","R","X"):last=None
                continue
            events_seen+=1
            if op=="S":
                assert output=={"ack":True}
                s_ack+=1;saw_ack=True
            elif op=="P":
                assert set(output)=={"value"}
                v=output["value"];assert v in (0,1)
                p[pub][v]+=1;last=v
            elif op=="R":
                if "value" in output:
                    assert output["value"] in (0,1)
                    if saw_ack:r_valid+=1
                    last=output["value"]
                else:
                    assert output=={"error":True}
                    if saw_ack:r_error+=1
                    last=None
            elif op=="X":
                if "value" in output:
                    v=output["value"];assert v in (0,1)
                    if last in (0,1):x[last][v]+=1
                    last=v
                else:
                    assert output=={"error":True}
                    last=None
            else:raise AssertionError(op)
    return {
        "schema":"F10P_TOOL_OUTPUT_EVENTS_V1",
        "episodes":len(trials),
        "events_seen":events_seen,
        "minimum_observations":10,
        "consistency_threshold":.90,
        "P":{str(b):dict(sorted(p[b].items())) for b in (0,1)},
        "X":{str(b):dict(sorted(x[b].items())) for b in (0,1)},
        "S_ack":s_ack,
        "R_valid_after_S_ack":r_valid,
        "R_error_after_S_ack":r_error,
    }

def read_and_verify_trials(folder,meta,rep):
    f=folder/meta["public_trial_log"]["path"]
    assert sha(f)==meta["public_trial_log"]["sha256"]
    trials=[]
    source_counts=Counter()
    program_counts=Counter()
    expected={
        "epoch","seed","variant","case_sha","source","flip","public_bit",
        "program_id","program","answer","success","reward",
        "public_events","projected_events"}
    for line in gzip.open(f,"rt"):
        r=json.loads(line)
        assert set(r)==expected,"Public trial log contains hidden-state fields"
        assert set(r["projected_events"])==set(EVENT_ARMS)
        assert (r["source"],r["flip"]) in CONTEXTS
        assert r["source"]!="MEMORY" or r["flip"]==0
        assert r["epoch"] in range(1,17)
        assert r["variant"] in VARIANTS
        assert r["success"] in (0,1)
        assert r["answer"] in (0,1)
        assert r["program_id"] in range(85)
        assert PROGRAMS[r["program_id"]] not in (("S","R","X"),)
        assert r["program"]==("".join(PROGRAMS[r["program_id"]]) or "EMPTY")
        assert r["reward"]>=-1.34 and r["reward"]<=1.
        assert r["seed"]==(
            28_100_000+10_000*rep+1000*CONTEXTS.index(
                (r["source"],r["flip"]))+
            (r["seed"]-(28_100_000+10_000*rep+
             1000*CONTEXTS.index((r["source"],r["flip"]))))
        )
        j=r["seed"]-(28_100_000+10_000*rep+
                     1000*CONTEXTS.index((r["source"],r["flip"])))
        assert 0<=j<256
        assert len(r["public_events"])==len(PROGRAMS[r["program_id"]])
        for name,proj in r["projected_events"].items():
            assert len(proj)==len(r["public_events"])
            for base,event in zip(r["public_events"],proj):
                assert set(base)==set(event)=={"op","output"}
                assert event["op"]==base["op"]
                assert set(base["output"]) in ({"value"},{"ack"},{"error"})
                assert event["output"] is None or (
                    set(event["output"]) in ({"value"},{"ack"},{"error"}))
                if name=="E100":assert base==event
                elif name=="E0":assert event["output"] is None
                elif name=="E_noX" and event["op"]=="X":
                    assert event["output"] is None
                elif name!="E_shuffleX":
                    assert event["output"] is None or base==event
                elif name=="E_shuffleX" and event["op"]!="X":
                    assert base==event
        trials.append(r)
        program_counts[r["program_id"]]+=1
        source_counts[(r["source"],r["flip"])]+=1
    assert len(trials)==98304
    assert len({x["case_sha"] for x in trials})==3072
    assert all(source_counts[c]==32768 for c in CONTEXTS)
    assert [program_counts[i] for i in range(85)]==meta["program_counts"]
    assert meta["program_counts"][PROGRAMS.index(("S","R","X"))]==0
    for arm in EVENT_ARMS:
        got=reconstruct_public_memory(trials,arm)
        state=json.loads((folder/(arm+".json")).read_text())
        # JSON maps stringify int keys; compare normalized JSON.
        assert json.dumps(got,sort_keys=True)==json.dumps(state,sort_keys=True)
    b=json.loads((folder/"B_reward.json").read_text())
    assert b["episodes"]==98304
    assert sum(s["count"] for s in b["scores"])==98304
    assert all(s["program_id"]!=PROGRAMS.index(("S","R","X"))
               for s in b["scores"])
    i=json.loads((folder/"I_privileged.json").read_text())
    assert i["episodes"]==98304
    return {"unique_training_cases":3072,"episodes":98304,
            "observed_public_events":{
                arm:reconstruct_public_memory(trials,arm)["events_seen"]
                for arm in EVENT_ARMS
            }}

def audit(root,g0):
    output={
        "experiment":"F10-P public tool outcome identifiability",
        "technical":"F10P_TECHNICAL_COMPLETE",
        "original_f10l_run":41685499939,
        "confirm_28300000_sealed":True,
        "replicas":{},
    }
    all_hashes=set()
    for rep in range(4):
        folder=root/str(rep)
        meta=json.loads((folder/"frozen.json").read_text())
        rows=json.loads((folder/"eval_rows.json").read_text())
        assert meta["replica"]==rep
        assert meta["state"]=="ALL_MEMORY_SHA_FROZEN_BEFORE_DEV_OOD"
        assert meta["no_train_srx"] and meta["no_train_memory_flip1"]
        assert meta["confirm_28300000_sealed"]
        assert meta["train_cases"]==3072 and meta["epochs"]==16
        assert meta["trials_executed"]==98304
        assert meta["all_arms_share_exact_programs"]
        assert meta["original_skills"]["all_5_original_checkpoint_sha"]==(
            g0["replicas"][str(rep)]["model_sha256"])
        assert len(meta["trained_memory_files"])==7
        assert len(meta["train_case_shas"])==3072
        assert len(set(meta["train_case_shas"]))==3072
        for desc in meta["trained_memory_files"].values():
            assert sha(folder/desc["path"])==desc["sha256"]
        integrity=read_and_verify_trials(folder,meta,rep)
        tr=set(meta["train_case_shas"])
        assert not tr&all_hashes
        all_hashes.update(tr)
        assert len(rows)==7*(768+512)
        assert len({(r["arm"],r["case_sha"]) for r in rows})==len(rows)
        for split in ("dev","ood"):
            baseline=[r for r in rows if r["split"]==split and r["arm"]==ARMS[0]]
            assert len(baseline)==(768 if split=="dev" else 512)
            hashes={r["case_sha"] for r in baseline}
            assert len(hashes)==len(baseline)
            assert not hashes&all_hashes
            all_hashes.update(hashes)
            assert len({r["seed"] for r in baseline})==(192 if split=="dev" else 128)
            if split=="dev":
                assert {(r["task"],r["flip"]) for r in baseline}==set(CONTEXTS)
                assert {r["seed"] for r in baseline}=={
                    28_200_000+10000*rep+1000*t+j
                    for t in range(3) for j in range(64)}
            else:
                assert {(r["task"],r["flip"]) for r in baseline}=={
                    ("MEMORY",1)}
                assert {r["seed"] for r in baseline}=={
                    28_400_000+10000*rep+j for j in range(128)}
            fingerprint=None
            for arm in ARMS:
                block=[r for r in rows if r["split"]==split and r["arm"]==arm]
                ident={(r["seed"],r["variant"]):
                       (r["case_sha"],r["truth"],r["winner"],
                        r["selected"],r["public_bit"],r["source_mode"],r["n"])
                       for r in block}
                assert len(ident)==len(baseline)
                if fingerprint is None:fingerprint=ident
                else:assert fingerprint==ident
            grouped=defaultdict(list)
            for r in baseline:grouped[r["seed"]].append(r)
            assert all(len(v)==4 and {r["variant"] for r in v}==set(VARIANTS)
                       for v in grouped.values())
            for world in grouped.values():
                variants={r["variant"]:r for r in world}
                if world[0]["task"]=="MEMORY":
                    assert variants["U_before"]["truth"]==variants["U_after"]["truth"]
                    assert variants["U_after"]["truth"]==variants["S_before"]["truth"]
                    assert variants["S_after"]["truth"]!=variants["U_after"]["truth"]
        for r in rows:
            assert r["replica"]==rep
            assert r["arm"] in ARMS
            assert r["program_id"] in range(85)
            assert r["program"]==("".join(PROGRAMS[r["program_id"]]) or "EMPTY")
            assert r["n"] in (5,7,9,11)
            assert r["truth"] in (0,1)
            assert set(r["outcomes"])==set(MODES)
            for mode,v in r["outcomes"].items():
                assert v["answer"] in (0,1)
                assert v["success"]==int(v["answer"]==r["truth"])
                assert v["operations"] in range(4)
                assert v["invalid"] in range(4)
        arms={arm:[r for r in rows if r["split"]=="ood" and r["arm"]==arm]
              for arm in ARMS}
        by={arm:{
                "accuracy":accuracy(x),
                "correct":sum(r["outcomes"]["normal"]["success"] for r in x),
                "program_counts":dict(Counter(r["program"] for r in x)),
                "average_actions":sum(r["outcomes"]["normal"]["operations"]
                                      for r in x)/len(x),
                "by_n":{str(n):accuracy([r for r in x if r["n"]==n])
                        for n in (9,11)},
                "by_source":{s:accuracy([r for r in x if r["source_mode"]==s])
                             for s in SOURCES},
                "by_variant":{v:accuracy([r for r in x if r["variant"]==v])
                              for v in VARIANTS},
              } for arm,x in arms.items()}
        c=arms["E100"]
        dev={
            str(src)+"_"+str(flip):accuracy([
                r for r in rows if r["split"]=="dev" and r["arm"]=="E100"
                and (r["task"],r["flip"])==(src,flip)])
            for src,flip in CONTEXTS}
        ci=grouped_mean_ci(c,rep)
        contrasts={
            "E100_minus_F4shuffle":paired_contrast(c,c,rep,1,"shuffle"),
            "E100_minus_Xdisabled":paired_contrast(c,c,rep,2,"x_disabled"),
            "E100_minus_B":paired_contrast(c,arms["B_reward"],rep,3),
            "E100_minus_E0":paired_contrast(c,arms["E0"],rep,4),
            "E100_minus_noX":paired_contrast(c,arms["E_noX"],rep,5),
            "E100_minus_shuffleX":paired_contrast(c,arms["E_shuffleX"],rep,6),
            "E100_minus_E50":paired_contrast(c,arms["E50"],rep,7),
        }
        cstat=by["E100"]
        def passes_contrast(name,threshold):
            x=contrasts[name]
            return x["estimate"]>=threshold and x["ci95"][0]>0
        gates={
            "G0_seen":all(v>=.90 for v in dev.values()),
            "G1_unexecuted_composition":(
                cstat["accuracy"]>=.80 and ci[0]>.70
                and min(cstat["by_n"].values())>=.70
                and min(cstat["by_source"].values())>=.70
                and min(cstat["by_variant"].values())>=.70
                and cstat["program_counts"].get("SRX",0)>=.80*512
            ),
            "G2_causal_F4_X":(
                passes_contrast("E100_minus_F4shuffle",.20)
                and passes_contrast("E100_minus_Xdisabled",.20)
            ),
            "G3_vs_reward":passes_contrast("E100_minus_B",.15),
            "G4_feedback_identification":all(
                passes_contrast(k,.15)
                for k in ("E100_minus_E0","E100_minus_noX",
                          "E100_minus_shuffleX")
            ),
            "G5_E50_diagnostic":by["E50"]["accuracy"]>=.80,
            "G6_privileged_diagnostic":by["I_privileged"]["accuracy"]>=.80,
        }
        output["replicas"][str(rep)]={
            "dev_seen":dev,"ood":by,"E100_accuracy_ci95":ci,
            "contrasts":contrasts,"gates":gates,
            "integrity":integrity,
            "memory_sha256":meta["trained_memory_files"],
            "original_skill_sha256":
                meta["original_skills"]["all_5_original_checkpoint_sha"],
        }
        print("F10P_REPLICA",rep,
              "OOD",{k:round(v["accuracy"],4) for k,v in by.items()},
              "E100_PROGRAMS",cstat["program_counts"],
              "GATES",gates,flush=True)
    assert len(all_hashes)==4*(3072+768+512)
    output["unique_case_hashes"]=len(all_hashes)
    keys=list(output["replicas"]["0"]["gates"])
    output["gate_counts"]={
        k:sum(int(v["gates"][k]) for v in output["replicas"].values())
        for k in keys}
    output["status"]=(
        "F10P_OBSERVABLE_EFFECTS_PASS" if all(
            output["gate_counts"][k]>=3 for k in
            ("G0_seen","G1_unexecuted_composition",
             "G2_causal_F4_X","G3_vs_reward","G4_feedback_identification"))
        else "F10P_OBSERVABLE_EFFECTS_FAIL"
    )
    output["pooled_descriptive"]={
        arm:{
            "correct":sum(v["ood"][arm]["correct"]
                          for v in output["replicas"].values()),
            "total":2048}
        for arm in ARMS}
    print("F10P_ORIGINAL_SCIENTIFIC_RESULT",output["status"],
          output["gate_counts"],flush=True)
    return output

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--original-g0",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    a=ap.parse_args()
    record=audit(a.root,json.loads(a.original_g0.read_text()))
    a.out.write_text(json.dumps(record,indent=2))
