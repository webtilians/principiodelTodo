"""F12-O-F: 2x2 cursor format/wording factorial on novel worlds.

Prospectively preregistered at b18d66e25f2c; execution must follow
a successful independent F12-O-R replay gate. Never reuse F12-N worlds here.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from itertools import permutations
from collections import Counter
from pathlib import Path
from observatory import f12n_public_error_cursor as prior
from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY="F12-O-F"
ROLES=prior.ROLES
USED=(
    set(prior.USED_J_MAPS)|set(prior.USED_K_MAPS)|
    set(prior.USED_L_MAPS)|set(prior.USED_M_MAPS)|set(prior.ROLE_MAPS))
TOOLS=tuple(f"s{i}" for i in range(6))
ACTIONS=(*TOOLS,"verify")
ARMS=("A00","A01","A10","A11")
CONDITIONS=("stable","changed","forced")
HORIZON=7
SHARD_COUNT=4
GOAL="Use the hidden bit, invert it, and verify success."
SYSTEM=prior.SYSTEM.replace('"q0"','"s0"').replace("q0 by","s0 by")
CATALOG=prior.CATALOG.replace("q0..q5","s0..s5")
WORLD_LOCK="F12O_FACTOR_LOCK_2026-11-09:"
BIT_LOCK="F12O_BITS_LOCK_2026-11-09:"

def sha(s):return hashlib.sha256(s.encode()).hexdigest()

def selected_maps():
    all_maps=[tuple(p) for p in permutations(sorted(ROLES))
              if tuple(p) not in USED]
    all_maps.sort(key=lambda p:(sha(WORLD_LOCK+",".join(p)),p))
    result=tuple(all_maps[:24])
    if len(result)!=24 or len(set(result))!=24 or set(result)&USED:
        raise ValueError("24 independent F12-O-F worlds required")
    return result

ROLE_MAPS=selected_maps()
_PAIR_POOL=((0,0),(0,1),(1,0),(1,1))*6
_BIT_ORDER=sorted(range(24),key=lambda i:(sha(BIT_LOCK+str(i)),i))
BIT_PAIRS=tuple(_PAIR_POOL[i] for i in _BIT_ORDER)

class World:
    def __init__(self,wid,public_bit,hidden_bit,condition):
        self.roles=dict(zip(TOOLS,ROLE_MAPS[wid]))
        self._public=public_bit
        self._hidden=hidden_bit
        self._condition=condition
        self._selected=False
        self._work=None
        self.verified=False
    def execute(self,action):
        role=self.roles.get(action,action)
        if role=="LOAD":
            self._work=self._public
            return {"value":self._work}
        if role=="ENABLE":
            self._selected=True
            return {"ack":True}
        if role=="READ":
            if not self._selected:return {"error":"PRECONDITION"}
            self._work=self._hidden
            return {"value":self._work}
        if role=="PRIMARY" and self._condition=="changed":
            return {"error":"DISABLED_TOOL"}
        if role in ("PRIMARY","BACKUP","HOLD"):
            if self._work is None:return {"error":"UNINITIALIZED"}
            if role!="HOLD":self._work=1-self._work
            return {"value":self._work}
        if role=="verify":
            self.verified=self._work is not None and self._work==1-self._hidden
            return {"verified":self.verified}
        return {"error":"INVALID_ACTION"}

def collect_probes(wid):
    inverse={role:tool for tool,role in zip(TOOLS,ROLE_MAPS[wid])}
    result=[]
    for i,roles in enumerate(prior.PROBES):
        world=World(wid,prior.PUBLIC[i],prior.HIDDEN[i],"stable")
        events=[]
        for role in roles:
            action=inverse[role]
            events.append({"action":action,"response":world.execute(action)})
        result.append({"source_id":f"P{i}","public_input":prior.PUBLIC[i],"events":events})
    return result

def save_probes(store,records):
    for r in records:
        store.add(MemoryRecord(
            content=json.dumps(r,sort_keys=True,separators=(",",":")),
            kind=MemoryKind.EPISODIC,importance=.9,
            metadata={"study":STUDY,"source_id":r["source_id"],"observed_only":True}
        ))

def index(plan,backup,events):
    return prior.expected_index(plan,backup,events)

def has_valid_public_error(plan_info,history):
    plan=plan_info["plan"]
    return (
        index(plan,plan_info["backup"],history)==len(plan)-2
        and plan_info["backup"] in ACTIONS
        and {"P4","P5","P6","P7"}.issubset(set(plan_info["sources"]))
        and any(
            e["origin"] in ("llm","controlled_prefix")
            and e["action"]==plan[-2]
            and e["response"]=={"error":"DISABLED_TOOL"}
            for e in history
        )
    )

def cue(plan_info,history):
    plan=plan_info["plan"]
    return (plan_info["backup"] if has_valid_public_error(plan_info,history)
            else plan[index(plan,plan_info["backup"],history)])

def prompt_for(context,plan_info,history,arm):
    if arm not in ARMS:raise ValueError("Unknown factorial arm")
    plan=plan_info["plan"]
    position=index(plan,plan_info["backup"],history)
    action=cue(plan_info,history)
    if arm=="A00":
        instruction=f"NEXT EXPECTED PLAN STEP ({position+1}/{len(plan)}): {action}. "
    elif arm=="A01":
        instruction=f"PLAN POSITION ({position+1}/{len(plan)}). NEXT EXPECTED PLAN STEP: {action}. "
    elif arm=="A10":
        instruction=f"NEXT ADMISSIBLE ACTION ({position+1}/{len(plan)}): {action}. "
    elif arm=="A11":
        instruction=f"PLAN POSITION ({position+1}/{len(plan)}). NEXT ADMISSIBLE ACTION: {action}. "
    instruction+="This cursor advances only after a successful public tool response."
    visible=[{"action":e["action"],"response":e["response"]} for e in history]
    return (
        f"PUBLIC GOAL: {GOAL}\nCATALOG: {CATALOG}\n"
        f"VALID PLAN (derived from PUBLIC TOOL evidence): {json.dumps(plan)}\n"
        f"ALTERNATIVE observed tool for the inversion step: {plan_info['backup']}\n"
        f"PUBLIC EXPERIENCE REFERENCES:\n{context}\n"
        f"OBSERVED EVENTS IN CURRENT EPISODE: "
        f"{json.dumps(visible,separators=(',',':'))}\n"
        f"INSTRUCTION: {instruction}\nReturn one strict JSON action."
    )

def parse_action(raw):
    try:v=json.loads(raw)
    except (TypeError,ValueError):return "INVALID_LLM_OUTPUT"
    return v["action"] if type(v) is dict and set(v)=={"action"} and v["action"] in ACTIONS else "INVALID_LLM_OUTPUT"

def deterministic_reference(wid,condition,public_bit,hidden_bit,plan):
    world=World(wid,public_bit,hidden_bit,condition)
    events=[]
    for action in plan["plan"]:
        response=world.execute(action)
        events.append({"action":action,"response":response})
        if response=={"error":"DISABLED_TOOL"}:
            alternate=plan["backup"]
            events.append({"action":alternate,"response":world.execute(alternate)})
    return {"success":world.verified,"actions":len(events),"events":events}

def frozen_preflight():
    if (len(ROLE_MAPS)!=24 or set(ROLE_MAPS)&USED or
        Counter(BIT_PAIRS)!={(0,0):6,(0,1):6,(1,0):6,(1,1):6}):
        raise ValueError("F12-O-F world map or bits violated")
    result=[]
    for wid in range(24):
        st=prior.new_store()
        try:
            probes=collect_probes(wid)
            if len(probes)!=10 or any(e["action"]=="verify"
                                     for record in probes for e in record["events"]):
                raise ValueError("Probe invalid")
            save_probes(st,probes)
            observed=prior.from_store(st)
            plan=prior.grounded_plan(observed)
            if not plan:raise ValueError(f"Ungrounded policy V{wid}")
            context=prior.public_context(observed,plan)
            public,hidden=BIT_PAIRS[wid]
            controls={cond:deterministic_reference(wid,cond,public,hidden,plan)
                      for cond in ("stable","changed")}
            if not all(c["success"] and c["actions"]<=HORIZON for c in controls.values()):
                raise ValueError("Unsolvable reference")
            w=World(wid,public,hidden,"changed")
            prefix=[w.execute(tool) for tool in plan["plan"][:3]]
            if prefix[-1]!={"error":"DISABLED_TOOL"}:raise ValueError("No authentic disabled prefix")
            result.append({
                "world":f"V{wid}","world_index":wid,
                "roles_audit_only":list(ROLE_MAPS[wid]),
                "public_bit_audit_only":public,"hidden_bit_audit_only":hidden,
                "observed":observed,"plan":plan,"context":context,
                "preflight":controls,"forced_prefix_responses":prefix,
            })
        finally:st.close()
    return result

def run_episode(wid,condition,public_bit,hidden_bit,model,context,plan_info,arm):
    if arm not in ARMS or condition not in CONDITIONS:raise ValueError("Unknown arm/condition")
    world=World(wid,public_bit,hidden_bit,"changed" if condition=="forced" else condition)
    events=[]
    if condition=="forced":
        for action in plan_info["plan"][:3]:
            events.append({
                "step":len(events),"action":action,"response":world.execute(action),
                "origin":"controlled_prefix","raw":None,"exact_prompt":None,
                "prompt_sha256":None,"real_cursor_index_before":index(plan_info["plan"],plan_info["backup"],events),
                "recommended_action":None,"switched_after_public_error":False,
                "prompt_tokens":0,"output_tokens":0,"inference_seconds":0,
            })
        if events[-1]["response"]!={"error":"DISABLED_TOOL"}:
            raise ValueError("Forced prefix missing real error")
    for _ in range(HORIZON-len(events)):
        p=prompt_for(context,plan_info,events,arm)
        before=model.metadata()
        raw=model.chat(SYSTEM,p)
        after=model.metadata()
        action=parse_action(raw)
        recommendation=cue(plan_info,events)
        switched=has_valid_public_error(plan_info,events)
        response=world.execute(action)
        events.append({
            "step":len(events),"action":action,"response":response,
            "origin":"llm","raw":raw,"exact_prompt":p,
            "prompt_sha256":sha(p),"real_cursor_index_before":index(plan_info["plan"],plan_info["backup"],events),
            "recommended_action":recommendation,"switched_after_public_error":switched,
            "prompt_tokens":after.get("prompt_tokens",0)-before.get("prompt_tokens",0),
            "output_tokens":after.get("completion_tokens",0)-before.get("completion_tokens",0),
            "inference_seconds":after.get("inference_seconds",0)-before.get("inference_seconds",0),
        })
        if response=={"verified":True}:break
    failure_index=next((i for i,e in enumerate(events) if e["action"]==plan_info["plan"][-2]
                        and e["response"]=={"error":"DISABLED_TOOL"}),None)
    return {
        "success":world.verified,"events":events,"actions_total":len(events),
        "calls":sum(e["origin"]=="llm" for e in events),
        "prefixed_actions":sum(e["origin"]=="controlled_prefix" for e in events),
        "switches":sum(e["switched_after_public_error"] for e in events),
        "recovered_after_error":any(
            e["action"]==plan_info["backup"] and "value" in e["response"]
            and failure_index is not None and i>failure_index
            for i,e in enumerate(events)),
        "repeated_primary_after_error":any(
            e["action"]==plan_info["plan"][-2] and failure_index is not None and i>failure_index
            for i,e in enumerate(events)),
        "tokens_in":sum(e["prompt_tokens"] for e in events),
        "tokens_out":sum(e["output_tokens"] for e in events),
    }

def metrics(rows):
    return {
        "episodes":len(rows),"successes":sum(r["success"] for r in rows),
        "switching_episodes":sum(r["switches"]>0 for r in rows),
        "recovered_after_error":sum(r["recovered_after_error"] for r in rows),
        "repeated_primary_after_error":sum(r["repeated_primary_after_error"] for r in rows),
        "calls":sum(r["calls"] for r in rows),
        "prompt_tokens":sum(r["tokens_in"] for r in rows),
        "output_tokens":sum(r["tokens_out"] for r in rows),
        "total_actions":sum(r["actions_total"] for r in rows),
    }

def run_shard(model,shard,out=None):
    if shard not in range(SHARD_COUNT):raise ValueError("Invalid shard")
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    all_worlds=frozen_preflight()
    worlds=[w for w in all_worlds if w["world_index"]%SHARD_COUNT==shard]
    if len(worlds)!=6:raise ValueError("Each shard must contain six worlds")
    outcomes=[]
    for w in worlds:
        wid=w["world_index"]
        for cond in CONDITIONS:
            rotated=(wid+CONDITIONS.index(cond))%len(ARMS)
            for arm in ARMS[rotated:]+ARMS[:rotated]:
                r=run_episode(wid,cond,w["public_bit_audit_only"],
                              w["hidden_bit_audit_only"],model,w["context"],w["plan"],arm)
                outcomes.append({**r,"world":w["world"],"world_index":wid,
                                 "condition":cond,"arm":arm})
    result={
        "study":STUDY,"shard":shard,"model":model.metadata(),
        "worlds":worlds,"outcomes":outcomes,
        "status":"SHARD_ONLY_NO_SCIENTIFIC_CONCLUSION",
        "scientific_pass":False,"product_integrated":False,
    }
    if out is not None:
        (out/f"f12o_factor_shard{shard}.json").write_text(
            json.dumps(result,indent=2,sort_keys=True,ensure_ascii=False))
    return result

def paired(outcomes,cond,a,b):
    ma={r["world_index"]:r["success"] for r in outcomes if r["condition"]==cond and r["arm"]==a}
    mb={r["world_index"]:r["success"] for r in outcomes if r["condition"]==cond and r["arm"]==b}
    if set(ma)!=set(range(24)) or set(mb)!=set(range(24)):
        raise ValueError("Incomplete paired contrast")
    return {
        "a_only":sum(ma[i] and not mb[i] for i in range(24)),
        "b_only":sum(mb[i] and not ma[i] for i in range(24)),
        "both":sum(ma[i] and mb[i] for i in range(24)),
        "neither":sum(not ma[i] and not mb[i] for i in range(24)),
    }

def aggregate(shards,out=None):
    if len(shards)!=4 or {s["shard"] for s in shards}!={0,1,2,3}:
        raise ValueError("Four unique shards required")
    worlds=[w for s in shards for w in s["worlds"]]
    outcomes=[r for s in shards for r in s["outcomes"]]
    ids=[(r["world_index"],r["condition"],r["arm"]) for r in outcomes]
    expected={(w,c,a) for w in range(24) for c in CONDITIONS for a in ARMS}
    if len(worlds)!=24 or len(outcomes)!=288 or len(set(ids))!=288 or set(ids)!=expected:
        raise ValueError("Incomplete or duplicate factor observations")
    wby={w["world_index"]:w for w in worlds}
    if len(wby)!=24:raise ValueError("Duplicate world")
    world_valid=all(
        w["roles_audit_only"]==list(ROLE_MAPS[w["world_index"]]) and
        (w["public_bit_audit_only"],w["hidden_bit_audit_only"])==BIT_PAIRS[w["world_index"]]
        and len(w["observed"])==10
        and [r["source_id"] for r in w["observed"]]==[f"P{i}" for i in range(10)]
        and all(e["action"]!="verify" for rec in w["observed"] for e in rec["events"])
        for w in worlds)
    model_valid=all(
        s["model"].get("real_model") is True and
        s["model"].get("weights_sha256")==FIXED_MODEL_SHA256 and
        s["model"].get("revision")=="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031" and
        s["model"].get("backend_version")=="0.3.36" and
        s["model"].get("temperature")==0.0 and
        s["model"].get("n_threads")==4
        for s in shards)
    ref_valid=all(w["preflight"][c]["success"] and w["preflight"][c]["actions"]<=7
                  for w in worlds for c in ("stable","changed"))
    forced_valid=all(w["forced_prefix_responses"][-1]=={"error":"DISABLED_TOOL"} for w in worlds)
    trace_valid=True
    switch_valid=True
    for row in outcomes:
        w=wby[row["world_index"]]
        ev=row["events"]
        if len(ev)>7 or row["actions_total"]!=len(ev) or row["calls"]!=sum(e["origin"]=="llm" for e in ev):
            trace_valid=False
        if row["condition"]=="forced":
            if len(ev)<3 or row["prefixed_actions"]!=3 or row["calls"]>4 or (
                [e["origin"] for e in ev[:3]]!=["controlled_prefix"]*3
                or ev[2]["response"]!={"error":"DISABLED_TOOL"}):
                trace_valid=False
        elif row["prefixed_actions"]!=0:trace_valid=False
        history=[]
        for i,e in enumerate(ev):
            if e["step"]!=i or e["real_cursor_index_before"]!=index(
                    w["plan"]["plan"],w["plan"]["backup"],history):
                trace_valid=False
            if e["origin"]=="llm":
                p=prompt_for(w["context"],w["plan"],history,row["arm"])
                if (e["exact_prompt"]!=p or sha(p)!=e["prompt_sha256"]
                    or e["action"]!=parse_action(e["raw"])
                    or e["recommended_action"]!=cue(w["plan"],history)):
                    trace_valid=False
                if ("roles_audit_only" in p or "hidden_bit_audit_only" in p
                    or "ROLE_MAPS" in p or "_hidden" in p):
                    trace_valid=False
                if e["switched_after_public_error"]!=has_valid_public_error(w["plan"],history):
                    switch_valid=False
            elif e["origin"]=="controlled_prefix":
                if (e["exact_prompt"] is not None or e["raw"] is not None
                    or e["prompt_sha256"] is not None or e["prompt_tokens"]!=0
                    or e["output_tokens"]!=0 or e["inference_seconds"]!=0
                    or e["switched_after_public_error"]):
                    trace_valid=False
            else:trace_valid=False
            history.append(e)
    scores={cond:{arm:metrics([r for r in outcomes if r["condition"]==cond and r["arm"]==arm])
                  for arm in ARMS} for cond in CONDITIONS}
    effects={}
    for cond in CONDITIONS:
        a={arm:scores[cond][arm]["successes"] for arm in ARMS}
        effects[cond]={
            "split_avg_net":((a["A01"]-a["A00"])+(a["A11"]-a["A10"]))/2,
            "wording_avg_net":((a["A10"]-a["A00"])+(a["A11"]-a["A01"]))/2,
            "interaction":(a["A11"]-a["A01"])-(a["A10"]-a["A00"]),
            "paired":{
                "split_expected":paired(outcomes,cond,"A01","A00"),
                "split_admissible":paired(outcomes,cond,"A11","A10"),
                "wording_inline":paired(outcomes,cond,"A10","A00"),
                "wording_split":paired(outcomes,cond,"A11","A01"),
            },
        }
    stable_no_switch=all(not r["switches"] for r in outcomes if r["condition"]=="stable")
    disjoint=not set(ROLE_MAPS)&USED and Counter(BIT_PAIRS)=={(0,0):6,(0,1):6,(1,0):6,(1,1):6}
    gates={
        "F0_real_24_disjoint_288_complete":model_valid and world_valid and disjoint and trace_valid,
        "F1_public_reference_and_forced_prefix":ref_valid and forced_valid,
        "F2_same_public_switch_stable_noop":stable_no_switch and switch_valid,
        "F3_A11_at_least_22_each":all(scores[c]["A11"]["successes"]>=22 for c in CONDITIONS),
    }
    split_supported=(effects["changed"]["split_avg_net"]>=4
                     and effects["forced"]["split_avg_net"]>=2
                     and effects["stable"]["split_avg_net"]>=0)
    wording_supported=(effects["changed"]["wording_avg_net"]>=4
                       and effects["forced"]["wording_avg_net"]>=2
                       and effects["stable"]["wording_avg_net"]>=0)
    gates["F4_split_signal"]=split_supported
    gates["F4_wording_signal"]=wording_supported
    gates["F5_trace_provenance_no_autonomous_claim"]=trace_valid and switch_valid
    avg_tokens={c:{a:scores[c][a]["prompt_tokens"]/max(scores[c][a]["calls"],1)
                   for a in ARMS} for c in CONDITIONS}
    cost_ratios={c:(max(v.values())/min(v.values()) if min(v.values())>0 else float("inf"))
                 for c,v in avg_tokens.items()}
    cost_confound=any(x>1.10 for x in cost_ratios.values())
    if not model_valid:status="FAKE_TEST_ONLY"
    elif not all(gates[k] for k in ("F0_real_24_disjoint_288_complete",
                                   "F1_public_reference_and_forced_prefix",
                                   "F2_same_public_switch_stable_noop",
                                   "F5_trace_provenance_no_autonomous_claim")):
        status="INVALID_FACTOR"
    elif not gates["F3_A11_at_least_22_each"]:status="PILOT_NO_GO"
    elif (split_supported or wording_supported) and not cost_confound:
        status="PILOT_FACTOR_GO_TO_NEW_CONFIRM"
    else:status="PILOT_MIXED"
    report={
        "study":STUDY,"status":status,"gates":gates,
        "scores":scores,"effects":effects,
        "prompt_tokens_per_llm_call":avg_tokens,
        "max_min_cost_ratio":cost_ratios,"text_and_cost_confound":cost_confound,
        "scientific_pass":False,"product_integrated":False,
        "scope":"2x2 prompt contrast on novel permutations of one binary synthetic task; external planner and public cursor",
        "shards":[{"id":s["shard"],"model":s["model"]} for s in shards],
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12o_factor_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        (out/"f12o_factor_full.json").write_text(
            json.dumps({**report,"worlds":worlds,"outcomes":outcomes},
                       indent=2,sort_keys=True,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int,choices=(0,1,2,3))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12o_factor"))
    args=p.parse_args()
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12o_factor_shard*.json"))
        if len(files)!=4:raise ValueError("Four factor shards required")
        result=aggregate([json.loads(f.read_text()) for f in files],args.out)
        print(json.dumps({k:result[k] for k in
                          ("study","status","gates","scores","effects","text_and_cost_confound")},
                         sort_keys=True))
        return 0
    if args.shard is None:p.error("Specify --shard or --aggregate-dir")
    result=run_shard(FrozenGGUF(),args.shard,args.out)
    print(json.dumps({"study":STUDY,"shard":args.shard,
                      "episodes":len(result["outcomes"]),"model":result["model"]},
                     sort_keys=True))
    return 0

if __name__=="__main__":raise SystemExit(main())
