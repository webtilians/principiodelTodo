"""F9-V: causal READ/SKIP from identical LSTM H/C, environment, and F4 state.

EcoDispatch-inspired hierarchical objective: preserve correct first fork choice
first, then minimize number of memory reads. This is a TRAIN-only
counterfactual teacher; deployment head never gets outcomes, F4 content,
target side, regime labels, or privileged coordinates.

Matched teacher prefix means causal autonomy is only AFTER the fork.
No external LLM. Old F9 checkpoints and F4 writer remain frozen.
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss

from observatory.f9_external_minigrid import ensure_dependencies,file_sha
from observatory.f9q_train_cue_probes import validate_sources
from observatory.f9q_causal_fork import (
    REGIMES,matched_pair,history_pair,replay_to_decision,clone_state,
)
from observatory.f9n_stage_autonomy import StageObserver
from observatory.f9j_recurrent_memory_attribution import frozen_action
from observatory.f9t_learned_f4_read_gate import (
    ReadHead,gate_features,gated_actor_input,ACTOR_DIM,DIM,
)

TRAIN_BASE=8_100_000
DEV_BASE=8_200_000
SEALED_BASE=8_300_000
SIZES={"train":32,"dev":16}
INDEX_COUNT=4
HEADS=("lex_hc","lex_public","scalar_hc")
EPOCHS=200
LR=0.001
EVALUATE_MODES=(*HEADS,"always","never","oracle")
SCALAR_READ_WEIGHT=1.1


def seed_for(split,index,j):
    if split not in SIZES or index not in range(INDEX_COUNT):
        raise ValueError("F9-V supports only new TRAIN/DEV")
    if not 0<=j<SIZES[split]:
        raise ValueError("Index out of new TRAIN/DEV bounds")
    return (TRAIN_BASE if split=="train" else DEV_BASE)+index*10_000+j


def lex_choice(read_correct,skip_correct):
    """EcoDispatch tier: first preserve capability; then reduce resource cost."""
    if read_correct not in (0,1) or skip_correct not in (0,1):
        raise ValueError("A first-choice correctness signal must be binary")
    read_tuple=(read_correct,-1)
    skip_tuple=(skip_correct,0)
    return int(read_tuple>skip_tuple)


def scalar_choice(read_correct,skip_correct):
    """Deliberately cost-dominant negative control, NOT a strong baseline."""
    return int(read_correct-SCALAR_READ_WEIGHT>skip_correct)


def one_shot_branch(model,index,pair,receiver,snapshot,read):
    """Same fork, same pre-action H/C. READ/SKIP differs only in F4 input."""
    env,raw,labels=replay_to_decision(
        pair["seed"],bool(receiver),
        pair["actions"][receiver][:pair["times"]["fork"]],
        snapshot["raw"])
    state=clone_state(snapshot["state"])
    observer=StageObserver(labels,13)
    observer.observe(env.unwrapped.agent_pos,0)
    first_action=None
    t=0
    try:
        encoded=np.concatenate((raw,snapshot["natural_f4"])).astype(np.float32)
        if encoded.shape!=(DIM,):
            raise AssertionError("Invalid actor 660-input")
        actor_input=gated_actor_input(raw,encoded,bool(read))
        assert np.array_equal(actor_input[:ACTOR_DIM],raw)
        while True:
            if t:
                actor_input=np.concatenate((
                    np.asarray(raw,np.float32),
                    np.zeros(DIM-ACTOR_DIM,np.float32)))
            action,state=frozen_action(model,actor_input,state,first=False)
            if first_action is None:
                first_action=action
            raw,reward,terminated,truncated,_=env.step(action)
            t+=1
            observer.observe(env.unwrapped.agent_pos,t)
            if observer.first_commit_step is not None or terminated or truncated:
                break
        summary=observer.summary()
        return {
            "first_action":int(first_action),
            "first_arm":summary["first_committed_side"],
            "first_arm_correct":summary["first_choice_correct"],
            "committed":summary["first_committed"],
            "steps_until_first_arm_or_exit":t,
            "read":int(bool(read)),
        }
    finally:
        env.close()


def one_seed_cases(model,index,seed):
    """Counterfactual worlds always generated in fixed preregistered order."""
    pair=matched_pair(model,index,seed)
    data=[]
    for regime in REGIMES:
        states=history_pair(model,index,pair,regime)
        for receiver in range(2):
            snapshot=states[receiver]
            a=one_shot_branch(model,index,pair,receiver,snapshot,1)
            b=one_shot_branch(model,index,pair,receiver,snapshot,0)
            # Factual and counterfactual have the same public view + H/C,
            # and only differ by a real F4 read at the single fork.
            r=a["first_arm_correct"]
            s=b["first_arm_correct"]
            label=lex_choice(r,s)
            if a["read"]!=1 or b["read"]!=0:
                raise AssertionError("READ/SKIP intervention not applied")
            data.append({
                "seed":seed,"regime":regime,
                "reverse":bool(receiver),
                "cue":pair["labels"][receiver]["cue"],
                "correct_side":pair["labels"][receiver]["correct_side"],
                "public_fork_equal":bool(np.array_equal(states[0]["raw"],states[1]["raw"])),
                "public":snapshot["raw"].tolist(),
                "hc":np.concatenate([x.flatten() for x in snapshot["state"]]).tolist(),
                "f4_nonempty":int(np.any(snapshot["natural_f4"])),
                "label":label,
                "scalar_label":scalar_choice(r,s),
                "effect":int(r-s),
                "read_branch":a,
                "skip_branch":b,
                "fork_step":int(pair["times"]["fork"]),
                "blackout_steps":int(snapshot["effective_prefix_masks"]),
            })
    return data


def head_inputs(data,use_hc):
    rows=[]
    for case in data:
        raw=np.asarray(case["public"],np.float32)
        if use_hc:
            v=np.asarray(case["hc"],np.float32)
            if v.shape!=(128,):
                raise AssertionError("Unexpected H/C")
            x=np.concatenate((raw,v))
        else:
            x=np.concatenate((raw,np.zeros(128,np.float32)))
        if x.shape!=(276,) or not np.isfinite(x).all():
            raise AssertionError("Only public and prior H/C can reach head")
        rows.append(x)
    return torch.tensor(np.asarray(rows,np.float32))


def fit_heads(data,index):
    if index not in range(INDEX_COUNT):
        raise ValueError(index)
    reports={}
    models={}
    for name in HEADS:
        torch.manual_seed(91_000+index)
        head=ReadHead()
        inputs=head_inputs(data,use_hc=name!="lex_public")
        target_key="scalar_label" if name=="scalar_hc" else "label"
        targets=torch.tensor([x[target_key] for x in data],dtype=torch.float32)
        if not set(targets.tolist()).issubset({0.,1.}):
            raise AssertionError("Invalid causal teacher labels")
        optimizer=torch.optim.Adam(head.parameters(),lr=LR)
        criterion=BCEWithLogitsLoss()
        for step in range(EPOCHS):
            logits=head.net(inputs).squeeze(-1)
            loss=criterion(logits,targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        head.eval()
        models[name]=head
        with torch.no_grad():
            pred=(torch.sigmoid(head.net(inputs).squeeze(-1))>=0.5).to(torch.int)
            train_acc=float((pred==targets).float().mean().item())
        reports[name]={
            "epochs":EPOCHS,"updates":EPOCHS,
            "samples":len(data),
            "positive_targets":int(targets.sum().item()),
            "train_accuracy":train_acc,
            "final_bce":float(loss.detach().item()),
            "features":"public_148+previous_HC_128" if name!="lex_public" else "public_148+zero_128",
        }
    return models,reports


def model_read(head,case,use_hc):
    x=head_inputs([case],use_hc=use_hc)[0]
    with torch.no_grad():
        probability=float(torch.sigmoid(head(x)).item())
    return int(probability>=0.5),probability


def eval_case(data,heads):
    results={}
    for name in EVALUATE_MODES:
        if name=="always":
            choice,prob=1,None
        elif name=="never":
            choice,prob=0,None
        elif name=="oracle":
            choice,prob=data["label"],None
        else:
            choice,prob=model_read(heads[name],data,name!="lex_public")
        outcome=data["read_branch"] if choice else data["skip_branch"]
        results[name]={
            "read":choice,
            "prob":prob,
            "first_arm_correct":outcome["first_arm_correct"],
            "first_arm":outcome["first_arm"],
            "committed":outcome["committed"],
            "steps":outcome["steps_until_first_arm_or_exit"],
            "label_match":int(choice==data["label"]),
        }
    return results


def summarize(data,choices):
    if len(data)!=len(choices):
        raise AssertionError("Incomplete evaluation")
    results={}
    for mode in EVALUATE_MODES:
        rows=[x[mode] for x in choices]
        result={
            "cases":len(rows),
            "first_arm_correct":sum(x["first_arm_correct"] for x in rows),
            "reads":sum(x["read"] for x in rows),
            "read_rate":statistics.mean(x["read"] for x in rows),
            "label_match":sum(x["label_match"] for x in rows),
            "committed":sum(x["committed"] for x in rows),
            "regimes":{},
        }
        for regime in REGIMES:
            idx=[i for i,c in enumerate(data) if c["regime"]==regime]
            selected=[rows[i] for i in idx]
            # Entries alternate original/inverted within each seed/regime.
            pairs=[selected[i:i+2] for i in range(0,len(selected),2)]
            if any(len(p)!=2 for p in pairs):
                raise AssertionError("Broken counterfactual cue pairing")
            result["regimes"][regime]={
                "cases":len(selected),
                "first_arm_correct":sum(x["first_arm_correct"] for x in selected),
                "reads":sum(x["read"] for x in selected),
                "pairs_both_first_correct":sum(
                    int(all(e["first_arm_correct"] for e in p)) for p in pairs),
                "pairs_same_side":sum(int(
                    p[0]["first_arm"]==p[1]["first_arm"]
                    and p[0]["first_arm"] in ("upper","lower")) for p in pairs),
            }
        results[mode]=result
    return results


def execute(args):
    if args.phase not in ("smoke","dev") or args.index not in range(INDEX_COUNT):
        raise ValueError("Only 4 sources, smoke or DEV; no confirmation")
    if args.phase=="smoke" and args.train_pairs!=1:
        raise ValueError("One-world technical smoke only")
    if args.phase=="dev" and args.train_pairs!=SIZES["train"]:
        raise ValueError("Preregistered TRAIN budget locked")
    torch.set_num_threads(1)
    ensure_dependencies()
    paths,shas,provenance=validate_sources(
        args.bc_source,args.ppo_source,args.index)
    from sb3_contrib import RecurrentPPO
    actor=RecurrentPPO.load(str(paths["ppo_sham"]),device="cpu")
    actor.policy.set_training_mode(False)
    for p in actor.policy.parameters():
        p.requires_grad_(False)
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("F9-V output must not exist")
    train=[]
    for j in range(args.train_pairs):
        train.extend(one_seed_cases(actor,args.index,seed_for("train",args.index,j)))
    heads,train_report=fit_heads(train,args.index)
    for name,head in heads.items():
        torch.save(head.state_dict(),root/f"{name}.pt")
    eval_split="train" if args.phase=="smoke" else "dev"
    eval_pairs=1 if args.phase=="smoke" else SIZES["dev"]
    dev=[]
    for j in range(eval_pairs):
        seed=seed_for(eval_split,args.index,j)
        cases=one_seed_cases(actor,args.index,seed)
        dev.extend(cases)
        print("F9V_PAIR",args.phase,args.index,j,
              [(x["regime"],x["effect"]) for x in cases],flush=True)
    results=[eval_case(case,heads) for case in dev]
    summary=summarize(dev,results)
    causal_effects=Counter(x["effect"] for x in train)
    dev_effects=Counter(x["effect"] for x in dev)
    report={
        "experiment":"F9-V EcoDispatch-inspired causal lexicographic memory read",
        "phase":args.phase,
        "index":args.index,
        "source_shas":shas,
        "source_provenance":provenance,
        "train_worlds":args.train_pairs,
        "dev_worlds":eval_pairs,
        "train_causal_effects":{str(k):int(v) for k,v in causal_effects.items()},
        "dev_causal_effects":{str(k):int(v) for k,v in dev_effects.items()},
        "train_heads":train_report,
        "dev_results":summary,
        "train_cases":train,
        "dev_cases":dev,
        "dev_choices":results,
        "actor_retrained":False,
        "F4_writer_changed":False,
        "uses_external_llm":False,
        "confirmation_8300000_opened":False,
        "f9u_7300000_opened":False,
        "f9t_6300000_opened":False,
        "historic_900000_opened":False,
        "protocol_sha256":file_sha(Path(
            "docs/INFINITO_F9V_LEXICOGRAPHIC_CAUSAL_F4_PROTOCOL.md")),
        "code_sha256":file_sha(Path(__file__)),
    }
    (root/"report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("F9V_SUMMARY",args.phase,args.index,json.dumps({
        "train_effect":report["train_causal_effects"],
        "dev_effect":report["dev_causal_effects"],
        "results":summary},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--phase",choices=("smoke","dev"),required=True)
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--train-pairs",type=int,required=True)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    execute(p.parse_args())


if __name__=="__main__":
    main()
