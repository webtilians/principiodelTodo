"""F9-V independent 8300000+ confirmation on immutably frozen DEV read heads.

NO training, no checkpoint selection, no new thresholds. Every policy weight
is SHA256-gated against the pre-confirmation lock JSON. Original F9-P actor
weights and provenance are checked separately. Research oracle cannot act
as a deployed policy; its first-fork counterfactual choices are diagnostic.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch

from observatory.f9_external_minigrid import file_sha,ensure_dependencies
from observatory.f9q_train_cue_probes import validate_sources
from observatory.f9t_learned_f4_read_gate import ReadHead
from observatory.f9v_lexicographic_causal_f4 import (
    INDEX_COUNT,HEADS,EVALUATE_MODES,REGIMES,one_seed_cases,eval_case,summarize,
)

CONFIRM_BASE=8_300_000
CONFIRM_PAIRS=40
LOCK_PATH=Path("docs/INFINITO_F9V_CONFIRMATION_LOCK.json")
PROTOCOL_PATH=Path("docs/INFINITO_F9V_LEXICOGRAPHIC_CAUSAL_F4_PROTOCOL.md")
SOURCE_PATH=Path("observatory/f9v_lexicographic_causal_f4.py")


def confirmation_seed(index,j):
    if index not in range(INDEX_COUNT) or not 0<=j<CONFIRM_PAIRS:
        raise ValueError("Confirmation only in precommitted disjoint 8300000+ worlds")
    return CONFIRM_BASE+index*10_000+j


def validate_lock(index,heads_dir,actor_sha):
    """Strict hashes, code, protocol, official DEV result and original actor."""
    if index not in range(INDEX_COUNT):
        raise ValueError(index)
    lock=json.loads(LOCK_PATH.read_text(encoding="utf8"))
    if (lock.get("experiment")!="F9-V" or
            lock.get("status")!="frozen_before_confirmation" or
            lock.get("development_run")!=41345497955 or
            lock.get("dev_gate_passed") is not True or
            lock.get("training_adjustments_after_dev") is not False or
            lock.get("confirm_cannot_train") is not True or
            lock.get("split")!={"base":CONFIRM_BASE,"per_replica":CONFIRM_PAIRS,"replicas":4}):
        raise AssertionError("Invalid preconfirmation lock")
    if (file_sha(PROTOCOL_PATH)!=lock["protocol_file_sha256"]
            or file_sha(SOURCE_PATH)!=lock["code_file_sha256"]):
        raise AssertionError("F9-V DEV protocol or causal implementation changed after locking")
    item=lock["weights"][str(index)]
    if item["actor_sha256"]!=actor_sha:
        raise AssertionError("Actor hash differs from original F9-P source")
    heads_dir=Path(heads_dir)
    for name in HEADS:
        if file_sha(heads_dir/(name+".pt"))!=item["heads"][name]:
            raise AssertionError("Frozen read head modified after DEV: "+name)
    if file_sha(heads_dir/"report.json")!=item["report_sha256"]:
        raise AssertionError("Original DEV report hash mismatch")
    dev=json.loads((heads_dir/"report.json").read_text(encoding="utf8"))
    if (dev["phase"]!="dev" or dev["index"]!=index or
            dev["source_shas"]["ppo_sham"]!=actor_sha or
            dev["code_sha256"]!=lock["code_file_sha256"] or
            dev["protocol_sha256"]!=lock["protocol_file_sha256"] or
            dev["train_worlds"]!=32 or dev["dev_worlds"]!=16 or
            len(dev["train_cases"])!=128 or len(dev["dev_cases"])!=64 or
            dev["confirmation_8300000_opened"] is not False):
        raise AssertionError("DEV report does not match frozen experiment")
    return lock,item,dev


def load_frozen_heads(heads_dir):
    heads={}
    for name in HEADS:
        head=ReadHead()
        head.load_state_dict(torch.load(
            Path(heads_dir)/(name+".pt"),
            map_location="cpu",weights_only=True))
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        heads[name]=head
    return heads


def run(args):
    if args.index not in range(INDEX_COUNT) or args.pairs!=CONFIRM_PAIRS:
        raise ValueError("Only preregistered 4x40 confirmation")
    torch.set_num_threads(1)
    ensure_dependencies()
    paths,source_shas,source_provenance=validate_sources(
        args.bc_source,args.ppo_source,args.index)
    lock,item,dev=validate_lock(
        args.index,args.heads_source,source_shas["ppo_sham"])
    from sb3_contrib import RecurrentPPO
    actor=RecurrentPPO.load(str(paths["ppo_sham"]),device="cpu")
    actor.policy.set_training_mode(False)
    for p in actor.policy.parameters():
        p.requires_grad_(False)
    heads=load_frozen_heads(args.heads_source)
    root=Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    if any(root.iterdir()):
        raise FileExistsError("Confirmation requires clean destination")
    cases=[]
    choices=[]
    for j in range(CONFIRM_PAIRS):
        seed=confirmation_seed(args.index,j)
        rows=one_seed_cases(actor,args.index,seed)
        if len(rows)!=4:
            raise AssertionError("Two history regimes and two cues required")
        cases.extend(rows)
        choices.extend(eval_case(case,heads) for case in rows)
        print("F9V_CONFIRM_PAIR",args.index,j,
              [(c["regime"],c["cue"],c["effect"]) for c in rows],
              flush=True)
    if len(cases)!=CONFIRM_PAIRS*4:
        raise AssertionError("Incomplete prospective confirmation")
    summary=summarize(cases,choices)
    effects=Counter(x["effect"] for x in cases)
    report={
        "experiment":"INFINITO F9-V independent one-shot causal F4 confirmation",
        "phase":"CONFIRM",
        "index":args.index,"confirm_pairs":CONFIRM_PAIRS,
        "only_confirm_seed_base":CONFIRM_BASE,
        "all_confirm_seeds":[confirmation_seed(args.index,j) for j in range(CONFIRM_PAIRS)],
        "frozen_development_run":lock["development_run"],
        "frozen_weights_hashes":item["heads"],
        "source_actor_sha256":source_shas["ppo_sham"],
        "source_provenance":source_provenance,
        "f9v_protocol_sha256":lock["protocol_file_sha256"],
        "f9v_code_sha256":lock["code_file_sha256"],
        "f9v_lock_sha256":file_sha(LOCK_PATH),
        "effects":{str(d):int(effects.get(d,0)) for d in (-1,0,1)},
        "summary":summary,
        "cases":cases,
        "choices":choices,
        "train_performed":False,"head_updated":False,"actor_updated":False,
        "f9v_confirm_used":True,
        "f9t_6300000_unopened":True,
        "f9u_7300000_unopened":True,
        "f9_historical_900000_unopened":True,
        "no_llm":True,
        "not_end_to_end":True,
    }
    (root/"report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    print("F9V_CONFIRM_RESULT",args.index,
          json.dumps({"effects":report["effects"],"summary":summary},ensure_ascii=False),flush=True)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--pairs",type=int,required=True)
    p.add_argument("--bc-source",type=Path,required=True)
    p.add_argument("--ppo-source",type=Path,required=True)
    p.add_argument("--heads-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())


if __name__=="__main__":
    main()
