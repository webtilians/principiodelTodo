"""F10-F: instrument EXACT frozen RelationalActor forward, no training.

Key distinctions:
- topk5 SELECTS by public key alone, never by verified version.
- own/pairwise/softmax MIX selected reports; head maps GRU + pooled read.
- diagnostic patches use observer-selected PUBLIC last signed; not agent policy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from observatory.f9_external_minigrid import file_sha
from observatory.f10b_train_evidence_actors import EvidenceActor
from observatory.f10b_temporal_evidence_task import (
    KEYS, CAPACITY, EVENT_FEATURES, WORLD_EVENTS,
)
from observatory.f10c_relational_actor_training import RelationalActor
from observatory.f10f_mechanism_worlds import world, dev_seed, N_WORLDS, VARIANTS, COUNTS, TIMINGS

MODES=("real","skip","unary","patch_signed_only","patch_winner_onehot",
       "patch_winner_include","patch_onehot_after_inclusion")
CATEGORIES=("MISSING_WINNER","HEAD_OR_HIDDEN_LIMIT","MIXING_OR_WEIGHTING_SENSITIVE")

def load_frozen(root,index):
    root=Path(root)
    manifest=json.loads((root/"report.json").read_text())
    assert manifest["index"]==index and manifest["phase"]=="TRAIN_ONLY"
    assert manifest["train_worlds"]==512 and manifest["train_cases"]==2048
    assert manifest["epochs"]==12 and not manifest["confirm_14300000_opened"]
    models={}
    for key,cls in (("relational_read",RelationalActor),
                    ("unary_read_control",EvidenceActor)):
        metadata=manifest["arms"][key]
        location=root/metadata["checkpoint"]
        assert file_sha(location)==metadata["sha256"],("Frozen SHA mismatch",location)
        model=cls()
        model.load_state_dict(torch.load(location,weights_only=True,map_location="cpu"))
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        models[key]=model
    return models,{"source_run":41447171697,
                   "original_training_manifest_sha256":file_sha(root/"report.json"),
                   "checkpoint_shas":{name:manifest["arms"][name]["sha256"]
                                      for name in models}}

def head_logit(actor,hidden,read_value,read_signed,available):
    x=torch.cat((hidden,
                 torch.as_tensor([[read_value]],dtype=hidden.dtype),
                 torch.as_tensor([[read_signed]],dtype=hidden.dtype),
                 torch.as_tensor([[available]],dtype=hidden.dtype)),dim=1)
    return float(actor.head(x).item())

def analyze_set(actor,mem,query,hidden,chosen,signed_only=False):
    """Copy the EXACT F10-C selector's 7-feature pairwise equations."""
    selected=mem.gather(1,chosen[:,:,None].expand(-1,-1,EVENT_FEATURES))
    valid=(selected[:,:,:KEYS]*query[:,None,:]).sum(-1)>0
    sign=selected[:,:,KEYS+1]
    moment=selected[:,:,KEYS+2]
    val=selected[:,:,KEYS]
    src=selected[:,:,KEYS+3:KEYS+5]
    same_src=(src[:,:,None,:]*src[:,None,:,:]).sum(-1)
    n=selected.shape[1]
    si=sign[:,:,None].expand(-1,-1,n)
    sj=sign[:,None,:].expand(-1,n,-1)
    vi=val[:,:,None].expand(-1,-1,n)
    vj=val[:,None,:].expand(-1,n,-1)
    td=moment[:,:,None]-moment[:,None,:]
    rel=torch.stack((si,sj,td,vi,vj,same_src,si*sj),dim=-1)
    score=actor.pairwise(rel).squeeze(-1)
    peer_valid=(valid[:,:,None]&valid[:,None,:])
    rank=(score*peer_valid.float()).sum(-1)/peer_valid.sum(-1).clamp(min=1)
    own=actor.unary(selected[:,:,KEYS+1:KEYS+5]).squeeze(-1)
    logits=(rank+own)*4.
    logits=logits.masked_fill(~valid,-1e5)
    effective_mask=valid if not signed_only else (valid&(sign>0.5))
    if signed_only and not bool(effective_mask.any().item()):
        return None
    if signed_only:
        logits=logits.masked_fill(~effective_mask,-1e5)
    weights=torch.softmax(logits,dim=1)
    weights=torch.where(effective_mask,weights,torch.zeros_like(weights))
    read_value=(weights*val).sum(1)
    read_sign=(weights*sign).sum(1)
    available=valid.any(1).float()
    features=torch.cat((hidden,read_value[:,None],
                        read_sign[:,None],available[:,None]),dim=1)
    final=actor.head(features).flatten()
    return {
        "logit":float(final.item()),
        "weights":weights[0].tolist(),
        "own":own[0].tolist(),
        "pairwise_rank":rank[0].tolist(),
        "attention_logit":logits[0].tolist(),
        "read_value":float(read_value.item()),
        "read_signed":float(read_sign.item()),
        "available":float(available.item()),
        "valid":valid[0].tolist(),
    }

def witness(memory,q):
    matches=[slot for slot in range(CAPACITY)
             if memory[slot,q]>0.5 and memory[slot,KEYS+1]>0.5]
    assert matches, "No signed report (public contract violation)"
    winner=max(matches,key=lambda slot:float(memory[slot,KEYS+2]))
    return winner

def prob(logit):
    return float(torch.sigmoid(torch.tensor(logit)).item())

def result_label(p,truth):
    return int(p>=0.5)==truth

def inspect(actor,unary,case):
    x=torch.from_numpy(case["public"][None])
    m=torch.from_numpy(case["memory"][None])
    query=x[:,-1,:KEYS]
    with torch.no_grad():
        _,last=actor.gru(x)
        hidden=last[-1]
        match=(m[:,:,:KEYS]*query[:,None,:]).sum(-1)
        chosen=torch.topk(match,5,dim=1).indices
        real=analyze_set(actor,m,query,hidden,chosen)
        reference=float(actor(x,m).item())
        assert abs(reference-real["logit"])<=1e-5,("READ forward mismatch",reference,real["logit"])
        zero=torch.zeros_like(m)
        chosen0=torch.topk((zero[:,:,:KEYS]*query[:,None,:]).sum(-1),5,dim=1).indices
        skip=analyze_set(actor,zero,query,hidden,chosen0)
        skip_reference=float(actor(x,zero).item())
        assert abs(skip_reference-skip["logit"])<=1e-5,("SKIP forward mismatch",skip_reference,skip["logit"])
        unary_p=prob(float(unary(x,m).item()))
        m_np=case["memory"]
        winner=witness(m_np,case["key"])
        indices=[int(v) for v in chosen[0].tolist()]
        present=winner in indices
        winner_pos=indices.index(winner) if present else None
        signed_only=analyze_set(actor,m,query,hidden,chosen,signed_only=True)
        if present:
            patched=real
        else:
            override=chosen.clone()
            override[0,-1]=winner
            patched=analyze_set(actor,m,query,hidden,override)
        winner_onehot=head_logit(actor,hidden,float(m_np[winner,KEYS]),
                                 float(m_np[winner,KEYS+1]),1.)
        # Observers select correct public signed record, not model.
        scores={}
        for mode,logit in (
            ("real",real["logit"]),
            ("skip",skip["logit"]),
            ("unary",None),
            ("patch_signed_only",None if signed_only is None else signed_only["logit"]),
            ("patch_winner_onehot",winner_onehot if present else None),
            ("patch_winner_include",patched["logit"]),
            ("patch_onehot_after_inclusion",winner_onehot),
        ):
            scores[mode]=unary_p if mode=="unary" else (None if logit is None else prob(logit))
        assert abs(scores["real"]-prob(reference))<=1e-6
        assert abs(scores["skip"]-prob(skip_reference))<=1e-6
        correct=result_label(scores["real"],case["truth"])
        if correct:
            category="CORRECT"
        elif not present:
            category="MISSING_WINNER"
        elif not result_label(scores["patch_winner_onehot"],case["truth"]):
            category="HEAD_OR_HIDDEN_LIMIT"
        else:
            category="MIXING_OR_WEIGHTING_SENSITIVE"
        serialized=[]
        for j,slot in enumerate(indices):
            row=m_np[slot]
            serialized.append({
                "slot":slot,
                "report_time":float(row[KEYS+2]),
                "value":int(row[KEYS]),
                "credential":int(row[KEYS+1]),
                "source":int(np.argmax(row[KEYS+3:KEYS+5])) if row[:KEYS].sum()>0 else None,
                "same_key":bool(row[case["key"]]>0.5),
                "is_public_signed_winner":slot==winner,
                "own":real["own"][j],
                "pairwise_rank":real["pairwise_rank"][j],
                "attention_logit":real["attention_logit"][j],
                "attention_weight":real["weights"][j],
            })
        selected_signed_count=sum(x["same_key"] and x["credential"]==1 for x in serialized)
        winner_attention=(float(real["weights"][winner_pos]) if present else 0.)
        winner_is_max=(present and winner_pos==int(np.argmax(real["weights"])))
        record={
            "seed":case["seed"],"variant":case["variant"],
            "key":case["key"],"truth":case["truth"],
            "base_value":case["base_value"],"record_count":case["record_count"],
            "timing":case["timing"],
            "history_sha256":case["history_sha256"],
            "winner_slot":winner,
            "winner_credential":int(m_np[winner,KEYS+1]),
            "winner_value":int(m_np[winner,KEYS]),
            "winner_report_time":float(m_np[winner,KEYS+2]),
            "indices":indices,
            "winner_in_topk":present,
            "winner_attention_weight":winner_attention,
            "winner_is_max_attention":winner_is_max,
            "selected_signed_count":selected_signed_count,
            "topk_reports":serialized,
            "pooled_value":real["read_value"],
            "pooled_credential":real["read_signed"],
            "scores":scores,
            "real_correct":correct,
            "error_category":category,
            "read_logit_exact_parity_error":abs(reference-real["logit"]),
            "skip_logit_exact_parity_error":abs(skip_reference-skip["logit"]),
        }
        assert record["winner_credential"]==1 and record["winner_value"]==case["truth"]
        assert sum(float(x["attention_weight"]) for x in serialized)<=1.000001
        assert len(indices)==5 and len(set(indices))==5
        if present:
            assert abs(scores["real"]-scores["patch_winner_include"])<=1e-6
        else:
            assert scores["patch_winner_onehot"] is None
        if not correct:assert category in CATEGORIES
        return record

def aggregate_group(rows):
    cases=len(rows)
    errors=[x for x in rows if not x["real_correct"]]
    counts={kind:sum(x["error_category"]==kind for x in rows) for kind in CATEGORIES}
    assert sum(counts.values())==len(errors)
    return {
        "cases":cases,"correct":cases-len(errors),"errors":len(errors),
        "error_categories":counts,
        "winner_in_topk":sum(x["winner_in_topk"] for x in rows),
        "winner_is_max_attention":sum(x["winner_is_max_attention"] for x in rows),
        "winner_weight_sum":sum(x["winner_attention_weight"] for x in rows),
        "onehot_correct_when_selected":sum(
            x["winner_in_topk"] and result_label(x["scores"]["patch_winner_onehot"],x["truth"])
            for x in rows),
        "include_correct":sum(
            result_label(x["scores"]["patch_winner_include"],x["truth"]) for x in rows),
        "onehot_with_observer_winner_correct":sum(
            result_label(x["scores"]["patch_onehot_after_inclusion"],x["truth"]) for x in rows),
        "signed_only_applicable":sum(x["scores"]["patch_signed_only"] is not None for x in rows),
        "signed_only_correct":sum(
            x["scores"]["patch_signed_only"] is not None and
            result_label(x["scores"]["patch_signed_only"],x["truth"]) for x in rows),
        "read_correct":sum(x["real_correct"] for x in rows),
        "skip_correct":sum(result_label(x["scores"]["skip"],x["truth"]) for x in rows),
        "unary_correct":sum(result_label(x["scores"]["unary"],x["truth"]) for x in rows),
    }

def pair_details(rows):
    grouped={}
    for row in rows: grouped.setdefault(row["seed"],{})[row["variant"]]=row
    assert len(grouped)==N_WORLDS
    pairs={}
    for name,a,b in (("time_signed","S_before","S_after"),
                     ("credential_after","U_after","S_after"),
                     ("time_unsigned","U_before","U_after")):
        set_changed=0
        winner_presence_toggled=0
        weight_delta=0.
        among_both=0
        for cases in grouped.values():
            p,q=cases[a],cases[b]
            assert p["key"]==q["key"]
            set_changed+=int(set(p["indices"])!=set(q["indices"]))
            winner_presence_toggled+=int(p["winner_in_topk"]!=q["winner_in_topk"])
            if p["winner_in_topk"] and q["winner_in_topk"]:
                among_both+=1
                weight_delta+=q["winner_attention_weight"]-p["winner_attention_weight"]
        pairs[name]={
            "worlds":len(grouped),"topk_set_changed":set_changed,
            "winner_presence_toggled":winner_presence_toggled,
            "winner_both_selected":among_both,
            "mean_winner_weight_delta_when_both":(
                weight_delta/among_both if among_both else None),
        }
    return pairs

def run(args):
    if args.index not in range(4):raise ValueError("Four frozen F10-C actor replicas only")
    torch.set_num_threads(1)
    models,prov=load_frozen(args.f10c_source,args.index)
    actor=models["relational_read"]
    unary=models["unary_read_control"]
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    rows=[]
    for j in range(N_WORLDS):
        seed=dev_seed(args.index,j)
        cases=world(seed,args.index,j)
        assert len(cases)==4
        for case in cases:rows.append(inspect(actor,unary,case))
        if j%18==0:print("F10F_INSTRUMENTED_NEW_WORLD",args.index,j,seed,flush=True)
    assert len(rows)==N_WORLDS*4
    strat={
        "by_variant":{v:aggregate_group([r for r in rows if r["variant"]==v])
                      for v in VARIANTS},
        "by_k":{str(k):aggregate_group([r for r in rows if r["record_count"]==k])
                for k in COUNTS},
        "by_timing":{t:aggregate_group([r for r in rows if r["timing"]==t])
                     for t in TIMINGS},
    }
    critical=strat["by_variant"]["S_after"]
    n=critical["errors"]
    prevalent=[c for c,v in critical["error_categories"].items()
               if n>=10 and v/n>0.5]
    assert len(prevalent)<=1
    dominance=(("DOMINANT_"+prevalent[0]) if prevalent else "MIXED_OR_UNDERPOWERED")
    report={
        "experiment":"F10-F frozen stagewise retrieval mechanism audit",
        "stage":"NEW_WORLD_MECHANISTIC_DIAGNOSTIC_ONLY",
        "replica":args.index,
        "seeds":[dev_seed(args.index,j) for j in range(N_WORLDS)],
        "worlds":N_WORLDS,"cases":len(rows),
        "actor_source":prov,
        "training_performed":False,"model_weights_changed":False,
        "read_head_trained":False,"old_dev_worlds_revisited":False,
        "confirm_17300000_opened":False,
        "intervention_uses_public_winner_oracle_for_diagnosis_only":True,
        "observer_patches_not_operational_policy":True,
        "read_and_skip_exact_forward_parity_all":True,
        "max_read_forward_logit_error":max(x["read_logit_exact_parity_error"] for x in rows),
        "max_skip_forward_logit_error":max(x["skip_logit_exact_parity_error"] for x in rows),
        "overall":aggregate_group(rows),"stratified":strat,
        "S_after_dominant_error_mechanism":dominance,
        "paired_variant_probes":pair_details(rows),
        "rows":rows,
    }
    (output/"mechanism-report.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    print("F10F_MECHANISM_REPLICA_SUMMARY",args.index,
          json.dumps({
             "overall":report["overall"],
             "S_after":critical,
             "S_after_dominance":dominance,
             "paired":report["paired_variant_probes"],
             "by_k":strat["by_k"],
             "max_logit_error":[report["max_read_forward_logit_error"],
                                report["max_skip_forward_logit_error"]],
          },ensure_ascii=False),flush=True)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--index",type=int,required=True)
    p.add_argument("--f10c-source",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    run(p.parse_args())
