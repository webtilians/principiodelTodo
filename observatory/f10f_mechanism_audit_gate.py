"""F10-F independent stdlib technical gate and mechanistic classification.

No model load, no training, no hidden label interventions, no tuning. Results
are diagnostic descriptions only, NOT a confirmation of internal reasoning.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path

MODES=("real","skip","unary","patch_signed_only","patch_winner_onehot",
       "patch_winner_include","patch_onehot_after_inclusion")
CATEGORIES=("MISSING_WINNER","HEAD_OR_HIDDEN_LIMIT","MIXING_OR_WEIGHTING_SENSITIVE")
VARIANTS=("U_before","U_after","S_before","S_after")
COUNTS=("3","5","8")
TIMINGS=("early","middle","late")
N=108

def correct(r,p):
    return int(r["scores"][p]>=0.5)==r["truth"]

def verify(reports):
    assert len(reports)==4
    complete={}
    dominance={}
    for i,report in enumerate(reports):
        assert report["replica"]==i and report["stage"]=="NEW_WORLD_MECHANISTIC_DIAGNOSTIC_ONLY"
        assert report["worlds"]==N and report["cases"]==4*N
        assert report["seeds"]==[17200000+10000*i+j for j in range(N)]
        assert report["actor_source"]["source_run"]==41447171697
        assert set(report["actor_source"]["checkpoint_shas"])=={
            "relational_read","unary_read_control"}
        assert not report["training_performed"] and not report["model_weights_changed"]
        assert not report["read_head_trained"] and not report["old_dev_worlds_revisited"]
        assert not report["confirm_17300000_opened"]
        assert report["intervention_uses_public_winner_oracle_for_diagnosis_only"]
        assert report["observer_patches_not_operational_policy"]
        assert report["read_and_skip_exact_forward_parity_all"]
        assert report["max_read_forward_logit_error"]<=1e-5
        assert report["max_skip_forward_logit_error"]<=1e-5
        rows=report["rows"]
        assert len(rows)==4*N
        worlds={}
        for r in rows:
            assert r["seed"] in report["seeds"]
            assert r["variant"] in VARIANTS
            assert r["record_count"] in (3,5,8)
            assert r["timing"] in TIMINGS
            assert r["truth"] in (0,1)
            assert set(r["scores"])==set(MODES)
            assert all(p is None or 0<=p<=1 for p in r["scores"].values())
            assert r["scores"]["real"] is not None
            assert r["scores"]["skip"] is not None
            assert r["scores"]["unary"] is not None
            assert r["scores"]["patch_winner_include"] is not None
            assert r["scores"]["patch_onehot_after_inclusion"] is not None
            assert r["read_logit_exact_parity_error"]<=1e-5
            assert r["skip_logit_exact_parity_error"]<=1e-5
            assert len(r["indices"])==5 and len(set(r["indices"]))==5
            assert all(0<=s<64 for s in r["indices"])
            assert len(r["topk_reports"])==5
            assert r["winner_credential"]==1 and r["winner_value"]==r["truth"]
            assert abs(sum(s["attention_weight"] for s in r["topk_reports"])-1)<1e-5
            assert r["winner_in_topk"]==(r["winner_slot"] in r["indices"])
            assert r["real_correct"]==correct(r,"real")
            if r["winner_in_topk"]:
                assert r["scores"]["patch_winner_onehot"] is not None
                assert abs(r["scores"]["patch_winner_include"]-r["scores"]["real"])<1e-6
                assert abs(r["scores"]["patch_winner_onehot"]-
                           r["scores"]["patch_onehot_after_inclusion"])<1e-6
                assert abs(r["winner_attention_weight"]-
                    next(s["attention_weight"] for s in r["topk_reports"]
                         if s["is_public_signed_winner"]))<1e-6
            else:
                assert r["scores"]["patch_winner_onehot"] is None
                assert r["winner_attention_weight"]==0
            if r["real_correct"]:
                assert r["error_category"]=="CORRECT"
            elif not r["winner_in_topk"]:
                assert r["error_category"]=="MISSING_WINNER"
            elif not correct(r,"patch_winner_onehot"):
                assert r["error_category"]=="HEAD_OR_HIDDEN_LIMIT"
            else:
                assert r["error_category"]=="MIXING_OR_WEIGHTING_SENSITIVE"
            worlds.setdefault(r["seed"],{})[r["variant"]]=r
        assert len(worlds)==N and all(set(c)==set(VARIANTS) for c in worlds.values())
        grid={(k,t):0 for k in (3,5,8) for t in TIMINGS}
        for group in worlds.values():
            variants=[group[name] for name in VARIANTS]
            assert len({r["key"] for r in variants})==1
            assert len({r["truth"] for r in variants})==2
            assert len({r["base_value"] for r in variants})==1
            assert len({r["record_count"] for r in variants})==1
            assert len({r["timing"] for r in variants})==1
            assert len({r["history_sha256"] for r in variants})==4
            b=variants[0]["base_value"]
            assert all(group[name]["truth"]==(
                1-b if name=="S_after" else b) for name in VARIANTS)
            grid[(variants[0]["record_count"],variants[0]["timing"])]+=1
        assert all(n==12 for n in grid.values())
        all_errors=[r for r in rows if not r["real_correct"]]
        calculated={cat:sum(r["error_category"]==cat for r in all_errors)
                    for cat in CATEGORIES}
        assert calculated==report["overall"]["error_categories"]
        assert len(all_errors)==report["overall"]["errors"]
        assert sum(calculated.values())==len(all_errors)
        assert set(report["stratified"])=={"by_variant","by_k","by_timing"}
        for axis,values,field,n_expected in (
            ("by_variant",VARIANTS,"variant",N),
            ("by_k",COUNTS,"record_count",N//3*4),
            ("by_timing",TIMINGS,"timing",N//3*4)):
            assert set(report["stratified"][axis])==set(values)
            for key in values:
                c=[r for r in rows if str(r[field])==str(key)]
                summ=report["stratified"][axis][key]
                assert len(c)==n_expected and summ["cases"]==n_expected
                errors=[r for r in c if not r["real_correct"]]
                counts={cat:sum(r["error_category"]==cat for r in errors)
                        for cat in CATEGORIES}
                assert summ["error_categories"]==counts
                assert summ["errors"]==len(errors)
                assert summ["read_correct"]==n_expected-len(errors)
                assert summ["winner_in_topk"]==sum(r["winner_in_topk"] for r in c)
                assert summ["signed_only_applicable"]==sum(
                    r["scores"]["patch_signed_only"] is not None for r in c)
                assert summ["include_correct"]==sum(
                    correct(r,"patch_winner_include") for r in c)
                assert summ["onehot_with_observer_winner_correct"]==sum(
                    correct(r,"patch_onehot_after_inclusion") for r in c)
        after=report["stratified"]["by_variant"]["S_after"]
        errs=after["errors"]
        cats=after["error_categories"]
        dominating=[k for k,v in cats.items() if errs>=10 and v/errs>0.5]
        assert len(dominating)<=1
        label="DOMINANT_"+dominating[0] if dominating else "MIXED_OR_UNDERPOWERED"
        assert report["S_after_dominant_error_mechanism"]==label
        dominance[str(i)]=label
        complete[str(i)]={
            "overall":report["overall"],
            "S_after":after,
            "by_variant":report["stratified"]["by_variant"],
            "by_k":report["stratified"]["by_k"],
            "by_timing":report["stratified"]["by_timing"],
            "paired":report["paired_variant_probes"],
            "max_read_forward_logit_error":report["max_read_forward_logit_error"],
            "max_skip_forward_logit_error":report["max_skip_forward_logit_error"],
            "dominant_error_class":label,
            "frozen_sha":report["actor_source"]["checkpoint_shas"],
        }
    agreed={kind:sum(x==kind for x in dominance.values())
            for kind in set(dominance.values())}
    consensus=[key for key,value in agreed.items() if value>=3 and
               key!="MIXED_OR_UNDERPOWERED"]
    assert len(consensus)<=1
    return {
        "experiment":"F10-F exact frozen topk5 mechanism audit",
        "phase":"MECHANISTIC_DIAGNOSTIC_ONLY",
        "status":"F10F_MECHANISM_AUDIT_COMPLETE",
        "technical_forward_parity_verified":True,
        "original_frozen_weights_verified":True,
        "new_worlds_only":True,
        "training_performed":False,
        "old_confirmations_opened":False,
        "observer_patches_operational":False,
        "dominant_S_after_error_class_across_replicas":(
            consensus[0] if consensus else "NO_THREE_REPLICA_DOMINANT_CLASS"),
        "dominance_by_replica":dominance,
        "per_replica":complete,
    }

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--gathered",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    reports=[json.loads((a.gathered/f"f10f-actor-{i}"/"mechanism-report.json").read_text())
             for i in range(4)]
    r=verify(reports)
    a.output.write_text(json.dumps(r,indent=2,ensure_ascii=False))
    print("F10F_INDEPENDENT_MECHANISM_AUDIT",json.dumps({
        "status":r["status"],
        "dominance_by_replica":r["dominance_by_replica"],
        "consensus":r["dominant_S_after_error_class_across_replicas"],
        "per_replica":{k:{
            "overall":v["overall"],
            "S_after":v["S_after"],
            "by_k":v["by_k"],
            "paired":v["paired"],
            "read_parity":v["max_read_forward_logit_error"],
            "skip_parity":v["max_skip_forward_logit_error"],
        } for k,v in r["per_replica"].items()},
    },ensure_ascii=False),flush=True)
