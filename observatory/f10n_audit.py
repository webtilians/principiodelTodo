"""F10-N INDEPENDENT scientific and data-integrity audit, no Torch/model imports.

World is unit of bootstrap (four related variants grouped), SHA manifests
frozen before holdouts. Outcome of a technically successful CI job is NOT
automatically a scientific PASS.
"""
from __future__ import annotations
import argparse,gzip,hashlib,json
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np

ARMS=("reward_table","additive_reward_table",
      "learned_effects_uniform","learned_effects_directed")
MODES=("normal","shuffle","x_disabled","public_randomized")
VARIANTS=("U_before","U_after","S_before","S_after")
SOURCES=("iid","signed_source0","signed_source1","alternating")
CONTEXTS=(("PUBLIC",0),("PUBLIC",1),("MEMORY",0))
HELDOUT="SRX"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def success(row,mode="normal"):
    return row["outcomes"][mode]["success"]

def rate(rows,mode="normal"):
    assert len(rows)
    return sum(success(r,mode) for r in rows)/len(rows)

def group_bootstrap(a,b,rep,mode_b="normal",offset=0):
    left={(r["seed"],r["variant"]):r for r in a}
    right={(r["seed"],r["variant"]):r for r in b}
    assert left.keys()==right.keys()
    seeds=sorted({key[0] for key in left})
    assert len(seeds)==128
    delta=np.array([
        sum(success(left[(s,v)])-success(right[(s,v)],mode_b)
            for v in VARIANTS)/4 for s in seeds],np.float64)
    rng=np.random.default_rng(267000+100*rep+offset)
    samples=delta[rng.integers(0,len(seeds),size=(2000,len(seeds)))].mean(1)
    return {"estimate":float(delta.mean()),
            "ci95":[float(x) for x in np.quantile(samples,(.025,.975))],
            "resamples":2000,"unit":"base_world"}

def accuracy_ci(rows,rep):
    group=defaultdict(list)
    for r in rows:group[r["seed"]].append(success(r))
    assert len(group)==128 and all(len(x)==4 for x in group.values())
    values=np.array([np.mean(x) for _,x in sorted(group.items())],np.float64)
    rng=np.random.default_rng(267000+100*rep)
    samples=values[rng.integers(0,len(values),size=(2000,len(values)))].mean(1)
    return [float(x) for x in np.quantile(samples,(.025,.975))]

def stats(rows):
    programs=Counter(r["program"] for r in rows)
    return {
        "cases":len(rows),"worlds":len({r["seed"] for r in rows}),
        "accuracy":rate(rows),
        "read_fraction":sum(r["program"].find("S")>=0 and
                            r["program"].find("R")>r["program"].find("S")
                            for r in rows)/len(rows),
        "program_counts":dict(programs.most_common()),
        "srx_fraction":programs["SRX"]/len(rows),
        "mean_reward":sum(r["outcomes"]["normal"]["reward"] for r in rows)/len(rows),
        "mean_operations":sum(r["outcomes"]["normal"]["operations"] for r in rows)/len(rows),
        "normal_minus_shuffle":rate(rows)-rate(rows,"shuffle"),
        "normal_minus_x_disabled":rate(rows)-rate(rows,"x_disabled"),
        "by_n":{str(n):rate([r for r in rows if r["n"]==n])
                for n in sorted({r["n"] for r in rows})},
        "by_source":{s:rate([r for r in rows if r["source_mode"]==s])
                     for s in sorted({r["source_mode"] for r in rows})},
        "by_variant":{v:rate([r for r in rows if r["variant"]==v])
                      for v in VARIANTS},
    }

def audit_training_log(folder,meta,stream):
    descriptor=meta["experience_log_files"][stream]
    path=folder/descriptor["file"]
    assert sha(path)==descriptor["sha256"]
    count=0
    program_counts=Counter()
    transitions=defaultdict(Counter)
    invalid=0
    nsteps=0
    context_counts=Counter()
    allowed={"phase","epoch","case_sha","seed","variant","source",
             "flip","public_bit","program_id","program","reward","success",
             "invalid","steps"}
    allowed_step={"op","before","after","invalid_delta","read_observed"}
    with gzip.open(path,"rt") as z:
        for line in z:
            r=json.loads(line)
            assert set(r)==allowed,"Training log exposes a hidden oracle field!"
            assert r["program"]!=HELDOUT
            assert (r["source"],r["flip"]) in CONTEXTS
            assert 1<=r["epoch"]<=16
            assert r["success"] in (0,1)
            assert r["variant"] in VARIANTS
            assert r["program_id"] in range(85)
            assert r["phase"] in (("uniform",) if stream=="uniform"
                                  else ("directed_uniform","directed_uncertainty"))
            program_counts[r["program_id"]]+=1
            context_counts[(r["source"],r["flip"])]+=1
            for t in r["steps"]:
                assert set(t)==allowed_step
                assert t["op"] in ("P","S","R","X")
                assert t["before"][0] in (0,1)
                assert t["before"][1] in (-1,0,1)
                assert t["after"][0] in (0,1)
                assert t["after"][1] in (-1,0,1)
                assert t["invalid_delta"] in (0,1)
                assert t["read_observed"] in (-1,0,1)
                # A true read bit is only available AFTER a valid READ.
                if t["op"]!="R" or not t["before"][0]:
                    assert t["read_observed"]==-1
                else:
                    assert t["read_observed"] in (0,1)
                key=",".join(map(str,(
                    t["op"],int(t["before"][0]),int(t["before"][1]),
                    int(r["public_bit"]) if t["op"]=="P" else -1,
                    int(t["read_observed"]) if t["op"]=="R" else -1)))
                target=",".join(map(str,(
                    t["after"][0],t["after"][1],t["invalid_delta"])))
                transitions[key][target]+=1
                invalid+=t["invalid_delta"]
                nsteps+=1
            assert len(r["steps"])<=3
            assert r["invalid"]==sum(t["invalid_delta"] for t in r["steps"])
            count+=1
    assert count==98304
    assert len(program_counts)>=65
    assert all(context_counts[c]==32768 for c in CONTEXTS)
    expected=meta["attempt_counts"][stream]
    assert sum(expected)==98304
    assert all(program_counts[pid]==v for pid,v in enumerate(expected))
    assert expected[meta["heldout_program_id"]]==0
    snapshot_path=folder/(
        "learned_effects_uniform.json" if stream=="uniform"
        else "learned_effects_directed.json")
    snapshot=json.loads(snapshot_path.read_text())
    assert snapshot["episodes"]==count
    assert snapshot["steps"]==nsteps and snapshot["invalid"]==invalid
    assert snapshot["transition_signatures"]==len(transitions)
    assert {k:dict(v) for k,v in transitions.items()}==snapshot["transitions"]
    return {
        "episodes":count,"observed_action_steps":nsteps,
        "invalid_preconditions":invalid,
        "unique_effect_signatures":len(transitions),
        "distinct_programs_executed":len(program_counts),
    }

def audit(root,g0):
    output={
        "experiment":"F10-N explicit experience memory and unexecuted program synthesis",
        "technical":"F10N_TECHNICAL_COMPLETE",
        "original_f10l_run":41685499939,
        "original_f10m_ood_not_reused":True,
        "confirm_26300000_unopened":True,
        "replicas":{},
    }
    all_hashes=set()
    for rep in range(4):
        folder=root/str(rep)
        meta=json.loads((folder/"training.json").read_text())
        rows=json.loads((folder/"rows.json").read_text())
        assert meta["status"]=="MEMORIES_SHA_FROZEN_BEFORE_DEV_OOD"
        assert meta["replica"]==rep
        assert meta["original_sha_verified_after_training"]
        assert meta["no_trainable_f10l_or_f10m_weights"]
        assert meta["no_program_truth_or_winner_targets"]
        assert meta["train_heldout_srx_count"]==0
        assert meta["unseen_memory_flip1_never_trained"]
        assert meta["confirm_26300000_sealed"]
        assert meta["epochs"]==16 and meta["train_examples"]==3072
        assert meta["episodes_per_condition"]==98304
        assert meta["two_programs_executed_per_case_per_epoch"]==2
        assert meta["train_contexts"]=={"PUBLIC_0":1024,"PUBLIC_1":1024,"MEMORY_0":1024}
        assert meta["original_program_catalog_size"]==85
        assert meta["train_program_catalog_size"]==84
        assert meta["heldout_program"]==HELDOUT
        assert meta["original_skills"]["all_5_original_checkpoint_sha"]==(
            g0["replicas"][str(rep)]["model_sha256"])
        assert meta["original_skills"]["pretrained_xor_bit_flip1_lut"]==[1,0]
        hashes=set(meta["train_case_shas"])
        assert len(hashes)==3072 and not hashes&all_hashes
        all_hashes.update(hashes)
        for desc in meta["memory_files"].values():
            assert sha(folder/desc["file"])==desc["sha256"]
        assert set(meta["memory_files"])=={
            "reward_table","learned_effects_uniform","learned_effects_directed"}
        rm=json.loads((folder/"reward_table.json").read_text())
        assert rm["episodes"]==98304
        assert sum(x["count"] for x in rm["scores"])==98304
        assert all(x["program_id"]!=meta["heldout_program_id"] for x in rm["scores"])
        integrity={
            stream:audit_training_log(folder,meta,stream)
            for stream in ("uniform","directed")
        }
        assert integrity["uniform"]["episodes"]==integrity["directed"]["episodes"]
        assert len(rows)==4*(768+512)
        assert len({(r["arm"],r["case_sha"]) for r in rows})==len(rows)
        canonical=None
        for arm in ARMS:
            sub=[r for r in rows if r["arm"]==arm]
            identity={(r["split"],r["seed"],r["variant"]):
                      (r["case_sha"],r["task"],r["flip"],
                       r["truth"],r["winner"],r["selected"],r["n"],
                       r["source_mode"],r["public_bit"])
                      for r in sub}
            assert len(identity)==1280
            if canonical is None:canonical=identity
            else:assert canonical==identity
        for split in ("dev","ood"):
            block=[r for r in rows if r["arm"]==ARMS[0] and r["split"]==split]
            assert len(block)==(768 if split=="dev" else 512)
            hashes={r["case_sha"] for r in block}
            assert len(hashes)==len(block) and not hashes&all_hashes
            all_hashes.update(hashes)
            if split=="dev":
                assert {(r["task"],r["flip"]) for r in block}==set(CONTEXTS)
                assert {r["seed"] for r in block}=={
                    26_200_000+10_000*rep+1000*t+j
                    for t in range(3) for j in range(64)}
            else:
                assert all(r["task"]=="MEMORY" and r["flip"]==1 for r in block)
                assert {r["seed"] for r in block}=={
                    26_400_000+10_000*rep+j for j in range(128)}
                for source in SOURCES:
                    for n in (9,11):
                        assert len({r["seed"] for r in block if
                                    r["source_mode"]==source and r["n"]==n})==16
            for seed in {r["seed"] for r in block}:
                siblings={r["variant"]:r for r in block if r["seed"]==seed}
                assert set(siblings)==set(VARIANTS)
                assert len({r["task"] for r in siblings.values()})==1
                if siblings["U_before"]["task"]=="MEMORY":
                    assert siblings["U_before"]["truth"]==siblings["U_after"]["truth"]
                    assert siblings["U_after"]["truth"]==siblings["S_before"]["truth"]
                    assert siblings["S_after"]["truth"]!=siblings["U_after"]["truth"]
                else:
                    assert len({r["truth"] for r in siblings.values()})==1
        for r in rows:
            assert r["replica"]==rep
            assert set(r["outcomes"])==set(MODES)
            assert r["truth"] in (0,1)
            assert r["winner"] in range(12)
            assert r["selected"] in range(12)
            assert r["oracle_srx_accuracy_diagnostic"] in (0,1)
            assert r["program_id"] in range(85)
            assert isinstance(r["predicted_score"],(int,float))
            for x in r["outcomes"].values():
                assert x["success"]==int(x["answer"]==r["truth"])
                assert x["operations"] in range(4)
                assert x["invalid"] in range(4)
                assert x["reward"]>=-1.34 and x["reward"]<=1.
        by_split={}
        for split in ("dev","ood"):
            by_split[split]={}
            for arm in ARMS:
                sub=[r for r in rows if r["split"]==split and r["arm"]==arm]
                if split=="dev":
                    by_split[split][arm]={
                        f"{task}_{flip}":stats([r for r in sub
                            if r["task"]==task and r["flip"]==flip])
                        for task,flip in CONTEXTS}
                else:
                    by_split[split][arm]=stats(sub)
        dev=by_split["dev"]["learned_effects_uniform"]
        x={arm:[r for r in rows if r["split"]=="ood" and r["arm"]==arm]
           for arm in ARMS}
        c=x["learned_effects_uniform"]
        contrasts={
            "C_minus_B":group_bootstrap(c,x["additive_reward_table"],rep,offset=3),
            "D_minus_C":group_bootstrap(x["learned_effects_directed"],c,rep,offset=4),
            "C_minus_shuffle":group_bootstrap(c,c,rep,mode_b="shuffle",offset=1),
            "C_minus_x_disabled":group_bootstrap(c,c,rep,mode_b="x_disabled",offset=2),
        }
        ci=accuracy_ci(c,rep)
        comp=by_split["ood"]["learned_effects_uniform"]
        g0=all(dev[f"{src}_{flip}"]["accuracy"]>=.90 for src,flip in CONTEXTS)
        g1=(comp["accuracy"]>=.80 and ci[0]>.70 and comp["srx_fraction"]>=.80
            and min(comp["by_n"].values())>=.70
            and min(comp["by_source"].values())>=.70
            and min(comp["by_variant"].values())>=.70)
        g2=all(contrasts[k]["estimate"]>=.20 and
               contrasts[k]["ci95"][0]>0 for k in (
                   "C_minus_shuffle","C_minus_x_disabled"))
        g3=(contrasts["C_minus_B"]["estimate"]>=.15 and
            contrasts["C_minus_B"]["ci95"][0]>0)
        g4=(contrasts["D_minus_C"]["estimate"]>=.05 and
            contrasts["D_minus_C"]["ci95"][0]>0)
        output["replicas"][str(rep)]={
            "metrics":by_split,"contrasts":contrasts,
            "world_bootstrap_ci95":ci,
            "integrity":integrity,
            "gates":{"G0_seen_tasks":g0,"G1_never_executed_SRX":g1,
                     "G2_F4_and_X_causality":g2,
                     "G3_effect_vs_reward":g3,
                     "G4_directed_vs_uniform":g4},
            "original_f10l_sha":meta["original_skills"]["all_5_original_checkpoint_sha"],
            "frozen_memory_sha":meta["memory_files"],
        }
        print("F10N_REPLICA",rep,
              "DEV_C",{k:round(v["accuracy"],4) for k,v in dev.items()},
              "OOD",{a:round(by_split["ood"][a]["accuracy"],4) for a in ARMS},
              "C_PROGRAMS",comp["program_counts"],
              "CI",ci,"GATES",output["replicas"][str(rep)]["gates"],flush=True)
    assert len(all_hashes)==4*(3072+768+512)
    output["unique_case_identity_hashes"]=len(all_hashes)
    counts={k:sum(int(r["gates"][k]) for r in output["replicas"].values())
            for k in ("G0_seen_tasks","G1_never_executed_SRX",
                      "G2_F4_and_X_causality","G3_effect_vs_reward",
                      "G4_directed_vs_uniform")}
    output["gate_counts"]=counts
    output["status"]=(
        "F10N_EFFECT_MEMORY_COMPOSITION_PASS"
        if all(counts[k]>=3 for k in (
            "G0_seen_tasks","G1_never_executed_SRX",
            "G2_F4_and_X_causality","G3_effect_vs_reward"))
        else "F10N_EFFECT_MEMORY_COMPOSITION_FAIL")
    print("F10N_FIXED_SCIENTIFIC_RESULT",output["status"],counts,flush=True)
    return output

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,required=True)
    p.add_argument("--original-g0",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    result=audit(a.root,json.loads(a.original_g0.read_text()))
    a.out.write_text(json.dumps(result,indent=2))
