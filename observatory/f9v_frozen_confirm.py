"""F9-V frozen confirmation. EXACT pretrained DEV heads; no retraining.

This file is only an evaluation harness. The original F9-V actor, intervention
code, objective, predictor, fixed decision threshold, and frozen model weights
are loaded as they existed BEFORE any 8_300_000+ world was evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from observatory.f9_external_minigrid import ensure_dependencies,file_sha
from observatory.f9q_train_cue_probes import validate_sources
from observatory.f9v_lexicographic_causal_f4 import (
    ReadHead,one_seed_cases,eval_case,summarize,EVALUATE_MODES,
)

ORIGINAL_F9V_COMMIT="898b6e43f6f1dd08b970fe4a0e4651d271944e80"
DEV_RUN=41345497955
ORIGINAL_CODE_SHA256="5bf29a5c4b178c8bdc7c40c64901fd52488c482e3ed9860b0ba162f6bc513532"
ORIGINAL_PROTOCOL_SHA256="73b3b19805f75edff093af169dcd485bc11307dbc72dc3a600c6b3ddb127ac2e"
# Frozen source artifacts from DEV run, indexed by original F9-P actor seed.
DEV_ARTIFACTS={
    0:13098763143,1:13098733338,2:13099027297,3:13099500331,
}
# SHA256 of serialized .pt bytes, inspected BEFORE opening the confirmation.
FROZEN_WEIGHTS={
    0:{"lex_hc":"a63c5f7e25eb12203301567e26018f1ea775d8774f8b241ac854b616e3ae3867",
       "lex_public":"5f534c0cfd48faf1e1b239392f05041e55c861154edb62c90449257611054639",
       "scalar_hc":"22557e8f86a648dd1d4927fa87577ec5a1939e42f2d84cf2cfc4ba8cd9ead114"},
    1:{"lex_hc":"a609b7667fb42f56074105beec67c99b600b35766d2c4e506f8912eb5bec75b2",
       "lex_public":"ec7538c800d146c8877a62238de2a1d7f7cd35bedea1fc08e76a11178c781fd9",
       "scalar_hc":"d385a03fe893edee97770ba816b7a01fbfae1f10ca53592e21e99490d9685027"},
    2:{"lex_hc":"fbc1df832fa784fe2995ee3266dd7c35c36bc3bf62e9f04a5409973401b5cab6",
       "lex_public":"8028987431da04c405b3d1e25e394c03681a1e5369b807a9837a72af167d248",
       "scalar_hc":"a1533d3c3947b43cb8e26de9fa758a941dced3d5b1dbbf523c25067afbd9775c"},
    3:{"lex_hc":"f84a085f0b56ed7eaab70b6bf74f42823b85d37c039bfce59f01b1c695c7f645",
       "lex_public":"76e12ddf552c73b809b4a95be3e7d1d0204192b36ceb58e14a9a819bac32f7a0",
       "scalar_hc":"600dcefda854c846d1896812cd1d85e11a857c458b3b5f2e99857f3b2295ab59"},
}
CONFIRM_BASE=8_300_000
N_WORLDS=40

def confirm_seed(index,j):
    if index not in range(4) or not 0<=j<N_WORLDS:
        raise ValueError("Only previously sealed F9-V 8300000+ confirm is allowed")
    return CONFIRM_BASE+10_000*index+j

def validate_original_implementation():
    path=Path("observatory/f9v_lexicographic_causal_f4.py")
    protocol=Path("docs/INFINITO_F9V_LEXICOGRAPHIC_CAUSAL_F4_PROTOCOL.md")
    if file_sha(path)!=ORIGINAL_CODE_SHA256 or file_sha(protocol)!=ORIGINAL_PROTOCOL_SHA256:
        raise AssertionError("Any change to F9-V experiment code/protocol invalidates confirmation")

def load_dev_models(root,index):
    if index not in DEV_ARTIFACTS:
        raise ValueError(index)
    r=json.loads((root/"report.json").read_text(encoding="utf8"))
    assert r["experiment"]=="F9-V EcoDispatch-inspired causal lexicographic memory read"
    assert r["phase"]=="dev" and r["index"]==index
    assert r["train_worlds"]==32 and r["dev_worlds"]==16
    assert len(r["train_cases"])==128 and len(r["dev_cases"])==64
    assert r["code_sha256"]==ORIGINAL_CODE_SHA256
    assert r["protocol_sha256"]==ORIGINAL_PROTOCOL_SHA256
    assert not r["confirmation_8300000_opened"]
    assert not r["actor_retrained"] and not r["F4_writer_changed"]
    assert not r["uses_external_llm"]
    models={}
    for name,sha in FROZEN_WEIGHTS[index].items():
        p=root/(name+".pt")
        if file_sha(p)!=sha:
            raise AssertionError(f"Frozen checkpoint mismatch: {name} index {index}")
        head=ReadHead()
        head.load_state_dict(torch.load(p,map_location="cpu",weights_only=True),strict=True)
        head.eval()
        for parameter in head.parameters():
            parameter.requires_grad_(False)
        models[name]=head
    return models,r

def execute(index,bc_source,ppo_source,weights_source,output):
    validate_original_implementation()
    if index not in range(4):raise ValueError("Only four original actors")
    torch.set_num_threads(1)
    ensure_dependencies()
    paths,shas,provenance=validate_sources(bc_source,ppo_source,index)
    from sb3_contrib import RecurrentPPO
    actor=RecurrentPPO.load(str(paths["ppo_sham"]),device="cpu")
    actor.policy.set_training_mode(False)
    for p in actor.policy.parameters():p.requires_grad_(False)
    heads,dev_report=load_dev_models(weights_source,index)
    if shas["ppo_sham"]!=dev_report["source_shas"]["ppo_sham"]:
        raise AssertionError("Actor changed since DEV")
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):raise FileExistsError("Cannot overwrite confirmation")
    cases=[]
    choices=[]
    for j in range(N_WORLDS):
        seed=confirm_seed(index,j)
        next_cases=one_seed_cases(actor,index,seed)
        assert len(next_cases)==4
        assert all(c["seed"]==seed and c["public_fork_equal"] for c in next_cases)
        cases.extend(next_cases)
        choices.extend(eval_case(c,heads) for c in next_cases)
        print("F9V_FROZEN_CONFIRM_WORLD",index,j,seed,flush=True)
    summary=summarize(cases,choices)
    assert all(v["cases"]==160 for v in summary.values())
    effect=Counter(c["effect"] for c in cases)
    # A HEAD never receives the D label, future outcome or correct side in
    # inference; these are EVALUATOR-ONLY after reading both rollouts.
    report={
        "experiment":"F9-V independent frozen 8300000+ confirmation",
        "phase":"confirm","index":index,
        "original_commit":ORIGINAL_F9V_COMMIT,"original_dev_run":DEV_RUN,
        "original_dev_artifact_id":DEV_ARTIFACTS[index],
        "original_code_sha256":ORIGINAL_CODE_SHA256,
        "original_protocol_sha256":ORIGINAL_PROTOCOL_SHA256,
        "frozen_weights_sha256":FROZEN_WEIGHTS[index],
        "source_shas":shas,"source_provenance":provenance,
        "confirm_worlds":N_WORLDS,"confirm_states":160,
        "effects":{str(k):int(v) for k,v in effect.items()},
        "summary":summary,
        "cases":cases,"choices":choices,
        "heads_retrained":False,"actor_retrained":False,
        "F4_writer_changed":False,"external_llm_used":False,
        "old_f9_holdout_900000_opened":False,
        "f9t_6300000_opened":False,"f9u_7300000_opened":False,
    }
    (output/"confirm.json").write_text(
        json.dumps(report,indent=2,ensure_ascii=False),encoding="utf8")
    print("F9V_FROZEN_CONFIRM_RESULT",index,json.dumps({
        "effects":report["effects"],"summary":summary},ensure_ascii=False),flush=True)
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--index",type=int,required=True)
    parser.add_argument("--bc-source",type=Path,required=True)
    parser.add_argument("--ppo-source",type=Path,required=True)
    parser.add_argument("--weights-source",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    execute(a.index,a.bc_source,a.ppo_source,a.weights_source,a.output)

if __name__=="__main__":main()
