"""F12-M: causal diagnosis of tool selection, finalization and error recovery.

Preregistered at ececbe9e7846 BEFORE this implementation. The LLM never
infers the plan; an evidence-only extractor supplies a mechanically valid
plan and documented alternative; public-only supervision is logged separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from itertools import permutations
from collections import Counter
from pathlib import Path

from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256
from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY="F12-M"

from observatory.f12j_policy_execution_revision import ROLE_MAPS as USED_J_MAPS
from observatory.f12k_cursor_causal_isolation import ROLE_MAPS as USED_K_MAPS
from observatory.f12lc_cursor_independent_confirmation import ROLE_MAPS as USED_L_MAPS

ROLES=("LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD")
LOCK_WORLD="F12M_DIAGNOSTIC_LOCK_2026-11-08:"
LOCK_BITS="F12M_BITS_2026-11-08:"

def hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def frozen_world_maps():
    excluded=set(USED_J_MAPS)|set(USED_K_MAPS)|set(USED_L_MAPS)
    candidates=[tuple(p) for p in permutations(sorted(ROLES))
                if tuple(p) not in excluded]
    candidates.sort(key=lambda p:(hash_text(LOCK_WORLD+",".join(p)),p))
    result=tuple(candidates[:24])
    if len(result)!=24 or len(set(result))!=24 or set(result)&excluded:
        raise ValueError("F12-M world selection failed frozen uniqueness")
    return result

ROLE_MAPS=frozen_world_maps()
_PAIR_POOL=((0,0),(0,1),(1,0),(1,1))*6
_BIT_ORDER=sorted(range(24),key=lambda i:(hash_text(LOCK_BITS+str(i)),i))
BIT_PAIRS=tuple(_PAIR_POOL[i] for i in _BIT_ORDER)
SHARD_COUNT=4

TOOLS=tuple(f"z{i}" for i in range(6))
ACTIONS=(*TOOLS,"verify")
SELECTION_ARMS=("S0","S1","T1")
RECOVERY_ARMS=("R0","R1","R2")
ARMS=SELECTION_ARMS+RECOVERY_ARMS
CONDITIONS=("stable","changed")
HORIZON=7
CONTEXT_LIMIT=3500
SYSTEM=(
    'You operate a simulated stateful tool world. Return exactly '
    '{"action":"z0"} replacing z0 by one allowed tool or verify. '
    'Read real tool feedback. You have a valid provisional plan built from '
    'public observations; no private hidden goal or roles are exposed.'
)
CATALOG=(
    "Opaque tools z0..z5 can change a binary register, enable operations, "
    "or leave it unchanged. They return public value 0/1, ack, or error. "
    "verify returns true/false. The public goal is to set the complement "
    "of a HIDDEN bit and verify. You may rely only on the observed history."
)
REPAIR=(
    "If the current inversion step returns error DISABLED_TOOL, select "
    "the alternative inversion tool supported by the public episodes, "
    "then verify. Do not retry a permanently unavailable operation."
)
NEUTRAL=(
    "For clarity, the plan and all available tools above are unchanged. "
    "This note supplies no new evidence or operating rule."
).ljust(len(REPAIR))


class World:
    """Private; source roles, register contents and target stay here."""
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

# A single independent world state for every partial probe. No target program.
PROBES=(("LOAD",),("LOAD",),
        ("ENABLE","READ"),("ENABLE","READ"),
        ("LOAD","PRIMARY"),("LOAD","PRIMARY"),
        ("LOAD","BACKUP"),("LOAD","BACKUP"),
        ("READ",),("LOAD","HOLD"))
PUBLIC=(0,1,0,1,0,1,0,1,0,0)
HIDDEN=(0,1,0,1,0,1,0,1,0,0)

def collect_probes(wid):
    inverse={role:tool for tool,role in zip(TOOLS,ROLE_MAPS[wid])}
    seen=[]
    for i,roles in enumerate(PROBES):
        world=World(wid,PUBLIC[i],HIDDEN[i],"stable")
        events=[]
        for role in roles:
            action=inverse[role]
            events.append({"action":action,"response":world.execute(action)})
        seen.append({"source_id":f"P{i}","public_input":PUBLIC[i],"events":events})
    return seen

def new_store():
    return SQLiteCognitiveMemoryStore(path=":memory:",embedding_provider=HashEmbeddingProvider())

def save_probes(st,records):
    for r in records:
        st.add(MemoryRecord(
            content=json.dumps(r,sort_keys=True,separators=(",",":")),
            kind=MemoryKind.EPISODIC,importance=.9,
            metadata={"study":STUDY,"source_id":r["source_id"],"observed_only":True}
        ))

def from_store(st):
    # Hybrid retrieval has a relevance floor; source-ID queries ensure the
    # same ten preregistered public probes, with no hidden world information.
    out=[]
    for i in range(10):
        sid=f"P{i}"
        matches=st.search(sid,top_k=20)
        rec=next((r for r in matches if r.kind==MemoryKind.EPISODIC
                  and r.metadata.get("source_id")==sid),None)
        if rec is None:raise ValueError(f"Missing public probe {sid}")
        out.append(json.loads(rec.content))
    out.sort(key=lambda r:int(r["source_id"][1:]))
    if [r["source_id"] for r in out]!=[f"P{i}" for i in range(10)]:
        raise ValueError("INFINITO SQLite probe retrieval incomplete")
    if any(set(r)!={"source_id","public_input","events"} for r in out):
        raise ValueError("Private record leaked")
    return out

def grounded_plan(observed):
    """Strong external control: derive plan solely from recorded tool outputs."""
    d={int(r["source_id"][1:]):r for r in observed}
    def e(i,k):return d[i]["events"][k]
    load=e(0,0)["action"]
    if not(e(0,0)["response"]=={"value":0}
           and e(1,0)=={"action":load,"response":{"value":1}}):
        return None
    enable,read=e(2,0)["action"],e(2,1)["action"]
    if not(e(2,0)["response"]=={"ack":True}
           and e(3,0)=={"action":enable,"response":{"ack":True}}
           and e(2,1)["response"]=={"value":0}
           and e(3,1)=={"action":read,"response":{"value":1}}
           and e(8,0)=={"action":read,"response":{"error":"PRECONDITION"}}):
        return None
    primary,backup=e(4,1)["action"],e(6,1)["action"]
    for left,right,tool in ((4,5,primary),(6,7,backup)):
        if not(e(left,0)["action"]==load and e(right,0)["action"]==load
               and e(left,1)["response"]=={"value":1}
               and e(right,1)=={"action":tool,"response":{"value":0}}):
            return None
    if len({load,enable,read,primary,backup})!=5:return None
    return {
        "plan":[enable,read,primary,"verify"],
        "backup":backup,
        "sources":["P0","P1","P2","P3","P4","P5","P6","P7","P8"],
    }

def public_context(observed,plan):
    text="\n".join(json.dumps(r,sort_keys=True,separators=(",",":")) for r in observed)
    if len(text)>CONTEXT_LIMIT:
        raise ValueError("Public evidence too long; no truncation permitted")
    if any(k not in plan for k in ("plan","backup","sources")):
        raise ValueError("No evidence grounded policy")
    return text

def expected_index(plan,backup,events):
    """Public cursor, not private oracle. Disabled operation never advances."""
    index=0
    for event in events:
        if index>=len(plan):break
        action=event["action"]
        response=event["response"]
        success="error" not in response and response!={"verified":False}
        if (success and (action==plan[index]
                         or (plan[index]==plan[-2] and action==backup))):
            index+=1
    return min(index,len(plan)-1)

def shown_action(plan,cursor,mode,backup=None,history=None):
    if mode in ("S1","T1","R0"):
        return plan[cursor]
    if mode=="R2":
        had_error=any(e["response"]=={"error":"DISABLED_TOOL"}
                      for e in (history or []))
        if cursor==len(plan)-2 and had_error:
            return backup
        return plan[cursor]
    return None

def prompt_for(public_goal,observed,plan_info,history,mode):
    plan=plan_info["plan"]
    cursor=expected_index(plan,plan_info["backup"],history)
    visible=[{"action":e["action"],"response":e["response"]} for e in history]
    if mode in ("S0","R1"):
        instructions=(f"CURRENT PLAN POSITION ({cursor+1}/{len(plan)}). "
                      "This cursor advances only after a successful public tool response.")
    elif mode in ("S1","T1","R0","R2"):
        cue=shown_action(plan,cursor,mode,plan_info["backup"],history)
        instructions=(f"NEXT EXPECTED PLAN STEP ({cursor+1}/{len(plan)}): {cue}. "
                      "This cursor advances only after a successful public tool response.")
    else:raise ValueError("Unregistered F12-M arm")
    return (
        f"PUBLIC GOAL: {public_goal}\nCATALOG: {CATALOG}\n"
        f"VALID PLAN (derived from PUBLIC TOOL evidence): {json.dumps(plan)}\n"
        f"ALTERNATIVE observed tool for the inversion step: {plan_info['backup']}\n"
        f"PUBLIC EXPERIENCE REFERENCES:\n{observed}\n"
        f"OBSERVED EVENTS IN CURRENT EPISODE: "
        f"{json.dumps(visible,separators=(',',':'))}\n"
        f"INSTRUCTION: {instructions}\nReturn one strict JSON action."
    )

def parse_action(raw):
    try:x=json.loads(raw)
    except (ValueError,TypeError):return "INVALID_LLM_OUTPUT"
    if type(x) is not dict or set(x)!={"action"} or x["action"] not in ACTIONS:
        return "INVALID_LLM_OUTPUT"
    return x["action"]


def _event(world,plan_info,events,action,origin,prompt=None,raw=None,delta=None):
    idx=expected_index(plan_info["plan"],plan_info["backup"],events)
    cue=shown_action(plan_info["plan"],idx,origin.get("arm"),plan_info["backup"],events) if origin.get("arm") else None
    response=world.execute(action)
    tokens=delta or {"prompt_tokens":0,"output_tokens":0,"inference_seconds":0}
    row={
        "step":len(events),"action":action,"response":response,
        "origin":origin["kind"],"arm":origin.get("arm"),
        "real_cursor_index_before":idx,
        "displayed_action_cue":cue,
        "expected_action":plan_info["plan"][idx],
        "raw_llm_response":raw,
        "exact_prompt":prompt,
        "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest() if prompt is not None else None,
        "prompt_tokens":tokens["prompt_tokens"],
        "output_tokens":tokens["output_tokens"],
        "inference_seconds":tokens["inference_seconds"],
    }
    events.append(row)
    return response

def _model_step(world,plan_info,events,model,context,arm):
    goal="Use the hidden bit, invert it, and verify success."
    prompt=prompt_for(goal,context,plan_info,events,arm)
    before=model.metadata() if hasattr(model,"metadata") else {}
    raw=model.chat(SYSTEM,prompt)
    after=model.metadata() if hasattr(model,"metadata") else {}
    action=parse_action(raw)
    return _event(world,plan_info,events,action,{"kind":"llm","arm":arm},prompt,raw,{
        "prompt_tokens":after.get("prompt_tokens",0)-before.get("prompt_tokens",0),
        "output_tokens":after.get("completion_tokens",0)-before.get("completion_tokens",0),
        "inference_seconds":after.get("inference_seconds",0)-before.get("inference_seconds",0),
    })

def make_result(world,plan_info,events):
    return {
        "success":world.verified,"events":events,
        "calls":sum(e["origin"]=="llm" for e in events),
        "supervisor_calls":sum(e["origin"]=="supervisor" for e in events),
        "prefixed_actions":sum(e["origin"]=="controlled_prefix" for e in events),
        "actions_total":len(events),
        "prompt_tokens":sum(e["prompt_tokens"] for e in events),
        "output_tokens":sum(e["output_tokens"] for e in events),
        "recovered_after_error":any(
            e["action"]==plan_info["backup"] and "value" in e["response"]
            and any(prev["response"]=={"error":"DISABLED_TOOL"} for prev in events[:i])
            for i,e in enumerate(events)),
        "preemptive_backup":any(
            e["action"]==plan_info["backup"] and "value" in e["response"]
            and not any(prev["response"]=={"error":"DISABLED_TOOL"} for prev in events[:i])
            for i,e in enumerate(events)),
    }

def run_selection_termination_case(wid,condition,public_bit,hidden_bit,model,context,plan_info,arm):
    if arm not in SELECTION_ARMS:raise ValueError("Selection arm expected")
    world=World(wid,public_bit,hidden_bit,condition)
    events=[]
    for t in range(HORIZON):
        index=expected_index(plan_info["plan"],plan_info["backup"],events)
        if arm=="T1" and index==len(plan_info["plan"])-1:
            # Supervisor deliberately receives ONLY the public cursor.
            response=_event(world,plan_info,events,"verify",
                            {"kind":"supervisor","arm":arm})
        else:
            response=_model_step(world,plan_info,events,model,context,arm)
        if response=={"verified":True}:break
    return make_result(world,plan_info,events)

def run_recovery_case(wid,public_bit,hidden_bit,model,context,plan_info,arm):
    if arm not in RECOVERY_ARMS:raise ValueError("Recovery arm expected")
    world=World(wid,public_bit,hidden_bit,"changed")
    events=[]
    # Controlled public prefix: same actual state/feedback for all arms.
    for action in plan_info["plan"][:3]:
        response=_event(world,plan_info,events,action,{"kind":"controlled_prefix","arm":arm})
    if events[-1]["response"]!={"error":"DISABLED_TOOL"}:
        raise ValueError("Forced prefix must expose real public DISABLED_TOOL")
    for t in range(HORIZON-3):
        response=_model_step(world,plan_info,events,model,context,arm)
        if response=={"verified":True}:break
    return make_result(world,plan_info,events)

def deterministic_case(wid,condition,public_bit,hidden_bit,plan_info):
    """Reference executability control; never supplied as model behavior."""
    world=World(wid,public_bit,hidden_bit,condition)
    events=[]
    for action in plan_info["plan"]:
        response=world.execute(action)
        events.append({"action":action,"response":response})
        if response=={"error":"DISABLED_TOOL"}:
            alt=plan_info["backup"]
            events.append({"action":alt,"response":world.execute(alt)})
        if len(events)>=HORIZON:break
    return {
        "success":world.verified,"events":events,"calls":len(events),
        "prompt_tokens":0,"output_tokens":0,
        "recovered_after_error":any(e["action"]==plan_info["backup"] for e in events),
    }


def metrics(rows):
    first_good=0
    for r in rows:
        first=next((e for e in r["events"] if e["origin"]=="llm"),None)
        if first and first["action"]==first["expected_action"]:first_good+=1
    return {
        "episodes":len(rows),
        "successes":sum(r["success"] for r in rows),
        "first_llm_action_matches_cursor":first_good,
        "recovered_after_error":sum(r["recovered_after_error"] for r in rows),
        "preemptive_backup":sum(r["preemptive_backup"] for r in rows),
        "calls":sum(r["calls"] for r in rows),
        "supervisor_calls":sum(r["supervisor_calls"] for r in rows),
        "prefixed_actions":sum(r["prefixed_actions"] for r in rows),
        "actions_total":sum(r["actions_total"] for r in rows),
        "prompt_tokens":sum(r["prompt_tokens"] for r in rows),
        "output_tokens":sum(r["output_tokens"] for r in rows),
        "inference_seconds":sum(e["inference_seconds"] for r in rows for e in r["events"]),
    }

def failure_reasons(row):
    if row["success"]:return []
    ev=row["events"]
    reasons=[]
    llm=next((e for e in ev if e["origin"]=="llm"),None)
    if llm is not None and llm["action"]!=llm["expected_action"]:
        reasons.append("wrong_first_action")
    if not any(e["real_cursor_index_before"]==3 for e in ev):
        reasons.append("never_reached_cursor_3")
    elif not any(e["action"]=="verify" for e in ev
                 if e["real_cursor_index_before"]==3):
        reasons.append("reached_cursor_3_no_verify")
    if any(e["response"]=={"verified":False} for e in ev):
        reasons.append("verify_false")
    if sum(e["response"]=={"error":"DISABLED_TOOL"} for e in ev)>1:
        reasons.append("repeated_disabled")
    if not reasons:reasons.append("other")
    return reasons


def frozen_preflight():
    worlds=[]
    used=set(USED_J_MAPS)|set(USED_K_MAPS)|set(USED_L_MAPS)
    if len(ROLE_MAPS)!=24 or len(set(ROLE_MAPS))!=24 or set(ROLE_MAPS)&used:
        raise ValueError("F12-M worlds not independent")
    if Counter(BIT_PAIRS)!={(0,0):6,(0,1):6,(1,0):6,(1,1):6}:
        raise ValueError("F12-M unbalanced input strata")
    for wid in range(24):
        st=new_store()
        try:
            original=collect_probes(wid)
            if len(original)!=10 or any(
                ev["action"]=="verify" for rec in original for ev in rec["events"]):
                raise ValueError("Invalid public probes")
            save_probes(st,original)
            observed=from_store(st)
            plan=grounded_plan(observed)
            if not plan:raise ValueError(f"Ungrounded plan N{wid}")
            ctx=public_context(observed,plan)
            pub,hid=BIT_PAIRS[wid]
            pre={c:deterministic_case(wid,c,pub,hid,plan)
                 for c in CONDITIONS}
            if not all(r["success"] and r["calls"]<=HORIZON for r in pre.values()):
                raise ValueError("Unsolvable reference world")
            w=World(wid,pub,hid,"changed")
            pref=[w.execute(action) for action in plan["plan"][:3]]
            if pref[-1]!={"error":"DISABLED_TOOL"}:
                raise ValueError("Broken seeded-error contract")
            worlds.append({
                "world":f"N{wid}","world_index":wid,
                "roles_audit_only":list(ROLE_MAPS[wid]),
                "public_bit_audit_only":pub,
                "hidden_bit_audit_only":hid,
                "observed":observed,"context":ctx,"plan":plan,
                "preflight":pre,"controlled_error_prefix_responses":pref,
            })
        finally:st.close()
    return worlds


def run_shard(model,shard,out=None):
    if shard not in range(SHARD_COUNT):raise ValueError("Unregistered shard")
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    worlds_all=frozen_preflight()
    worlds=[w for w in worlds_all if w["world_index"]%SHARD_COUNT==shard]
    if len(worlds)!=6:raise ValueError("Each shard requires six worlds")
    outcomes=[]
    for w in worlds:
        wid=w["world_index"]
        pub,hid=w["public_bit_audit_only"],w["hidden_bit_audit_only"]
        for condition in CONDITIONS:
            rotation=(wid+(condition=="changed"))%len(SELECTION_ARMS)
            for arm in SELECTION_ARMS[rotation:]+SELECTION_ARMS[:rotation]:
                result=run_selection_termination_case(
                    wid,condition,pub,hid,model,w["context"],w["plan"],arm)
                outcomes.append({
                    **result,"world":w["world"],"world_index":wid,
                    "condition":condition,"arm":arm,"sources":w["plan"]["sources"],
                    "failure_reasons":failure_reasons(result),
                })
        rotation=wid%len(RECOVERY_ARMS)
        for arm in RECOVERY_ARMS[rotation:]+RECOVERY_ARMS[:rotation]:
            result=run_recovery_case(wid,pub,hid,model,w["context"],w["plan"],arm)
            outcomes.append({
                **result,"world":w["world"],"world_index":wid,
                "condition":"changed","arm":arm,
                "sources":w["plan"]["sources"],"failure_reasons":failure_reasons(result),
            })
    metadata=model.metadata() if hasattr(model,"metadata") else {}
    report={"study":STUDY,"shard":shard,"shard_count":SHARD_COUNT,
            "study_status":"SHARD_ONLY_NO_SCIENTIFIC_CONCLUSION",
            "model":metadata,"worlds":worlds,"outcomes":outcomes,
            "scientific_pass":False,"product_integrated":False}
    if out is not None:
        (out/f"f12m_shard{shard}.json").write_text(
            json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True))
    return report

def paired_counts(outcomes,c,arm_a,arm_b):
    def m(arm):
        return {r["world_index"]:bool(r["success"]) for r in outcomes
                if r["condition"]==c and r["arm"]==arm}
    a,b=m(arm_a),m(arm_b)
    if set(a)!=set(range(24)) or set(b)!=set(range(24)):
        raise ValueError("Paired world coverage missing")
    return {"a_only":sum(a[i] and not b[i] for i in range(24)),
            "b_only":sum(b[i] and not a[i] for i in range(24)),
            "both":sum(a[i] and b[i] for i in range(24)),
            "neither":sum(not a[i] and not b[i] for i in range(24))}


def aggregate(shard_reports,out=None):
    if len(shard_reports)!=SHARD_COUNT or {r["shard"] for r in shard_reports}!=set(range(4)):
        raise ValueError("Incomplete four-shard protocol")
    worlds=[w for sh in shard_reports for w in sh["worlds"]]
    outcomes=[r for sh in shard_reports for r in sh["outcomes"]]
    expected={(wid,c,a) for wid in range(24) for c in CONDITIONS for a in SELECTION_ARMS}
    expected.update((wid,"changed",a) for wid in range(24) for a in RECOVERY_ARMS)
    actual=[(r["world_index"],r["condition"],r["arm"]) for r in outcomes]
    if len(worlds)!=24 or len(outcomes)!=216 or len(set(actual))!=216 or set(actual)!=expected:
        raise ValueError("Duplicate or missing episode(s)")
    if {w["world_index"] for w in worlds}!=set(range(24)):
        raise ValueError("World coverage incomplete")
    if any(w["roles_audit_only"]!=list(ROLE_MAPS[w["world_index"]])
           or (w["public_bit_audit_only"],w["hidden_bit_audit_only"])!=BIT_PAIRS[w["world_index"]]
           for w in worlds):
        raise ValueError("World identity or bit pairing altered")
    valid_model=all(
        sh["model"].get("real_model") is True
        and sh["model"].get("weights_sha256")==FIXED_MODEL_SHA256
        and sh["model"].get("backend_version")=="0.3.36"
        and sh["model"].get("temperature")==0.0
        for sh in shard_reports)
    prompt_integrity=True
    origin_integrity=True
    for row in outcomes:
        if row["actions_total"]!=len(row["events"]) or row["actions_total"]>HORIZON:
            origin_integrity=False
        if row["arm"] in RECOVERY_ARMS:
            if row["prefixed_actions"]!=3 or row["calls"]>4 or (
                    row["events"][2]["response"]!={"error":"DISABLED_TOOL"}):
                origin_integrity=False
        else:
            if row["prefixed_actions"]!=0:origin_integrity=False
        if row["arm"]!="T1" and row["supervisor_calls"]!=0:
            origin_integrity=False
        for i,event in enumerate(row["events"]):
            if event["step"]!=i or event["expected_action"] not in ACTIONS:
                origin_integrity=False
            if event["origin"]=="llm":
                p=event["exact_prompt"]
                if (not p or not event["raw_llm_response"]
                    or hashlib.sha256(p.encode()).hexdigest()!=event["prompt_sha256"]
                    or "roles_audit_only" in p or "hidden_bit_audit_only" in p):
                    prompt_integrity=False
                if event["action"]!=parse_action(event["raw_llm_response"]):
                    origin_integrity=False
            elif event["origin"] in ("supervisor","controlled_prefix"):
                if (event["exact_prompt"] is not None
                    or event["raw_llm_response"] is not None
                    or event["prompt_tokens"]!=0 or event["output_tokens"]!=0
                    or event["inference_seconds"]!=0
                    or event["prompt_sha256"] is not None):
                    origin_integrity=False
                if event["origin"]=="supervisor" and (
                        row["arm"]!="T1" or event["action"]!="verify"
                        or event["real_cursor_index_before"]!=3):
                    origin_integrity=False
            else:origin_integrity=False
    scores={c:{a:metrics([r for r in outcomes if r["condition"]==c and r["arm"]==a])
               for a in (SELECTION_ARMS if c=="stable" else ARMS)}
            for c in CONDITIONS}
    st,ch=scores["stable"],scores["changed"]
    ref=all(w["preflight"][c]["success"] and w["preflight"][c]["calls"]<=HORIZON
            for w in worlds for c in CONDITIONS)
    preflight_errors=all(w["controlled_error_prefix_responses"][-1]=={"error":"DISABLED_TOOL"}
                         for w in worlds)
    M2=(st["S1"]["first_llm_action_matches_cursor"]>=22
        and st["S1"]["first_llm_action_matches_cursor"]-
            st["S0"]["first_llm_action_matches_cursor"]>=4
        and st["S1"]["successes"]>=18
        and st["S1"]["successes"]-st["S0"]["successes"]>=4)
    M3=(any(scores[c]["T1"]["successes"]-scores[c]["S1"]["successes"]>=4
            for c in CONDITIONS)
        and sum(scores[c]["T1"]["supervisor_calls"] for c in CONDITIONS)>=4)
    R=ch
    M4=(R["R2"]["successes"]>=18
        and R["R2"]["successes"]-R["R0"]["successes"]>=6
        and R["R2"]["recovered_after_error"]>=18
        and all(r["calls"]<=4 and r["prefixed_actions"]==3
                for r in outcomes if r["arm"] in RECOVERY_ARMS))
    M0=(valid_model and len(worlds)==24 and len(outcomes)==216
        and Counter(BIT_PAIRS)=={(0,0):6,(0,1):6,(1,0):6,(1,1):6}
        and not set(ROLE_MAPS)&(set(USED_J_MAPS)|set(USED_K_MAPS)|set(USED_L_MAPS))
        and prompt_integrity)
    M1=ref and preflight_errors
    M5=origin_integrity and prompt_integrity and all(
        len(w["observed"])==10 and
        [r["source_id"] for r in w["observed"]]==[f"P{i}" for i in range(10)]
        for w in worlds)
    gates={
        "M0_model_worlds_216_complete":M0,
        "M1_reference_and_public_error_prefix":M1,
        "M2_named_step_improves_selection":M2,
        "M3_public_supervised_verify_improves_termination":M3,
        "M4_backup_cue_improves_forced_error_recovery":M4,
        "M5_traces_origins_budgets_and_no_hidden_leaks":M5,
        "M6_no_claim_of_autonomous_learning":True,
    }
    successes=sum(gates[k] for k in (
        "M2_named_step_improves_selection",
        "M3_public_supervised_verify_improves_termination",
        "M4_backup_cue_improves_forced_error_recovery"))
    if not valid_model:status="FAKE_TEST_ONLY"
    elif not(M0 and M1 and M5):status="INVALID_OR_INCOMPLETE"
    elif successes>=2:status="PILOT_MECHANISM_GO_TO_NEW_CONFIRM"
    elif successes==1:status="PILOT_MIXED"
    else:status="PILOT_NO_GO"
    report={
        "study":STUDY,"status":status,"model_sha256":FIXED_MODEL_SHA256,
        "gates":gates,"scores":scores,
        "paired":{c:{
            "selection_S1_vs_S0":paired_counts(outcomes,c,"S1","S0"),
            "termination_T1_vs_S1":paired_counts(outcomes,c,"T1","S1")}
            for c in CONDITIONS},
        "paired_recovery":{
            "backup_cue_R2_vs_primary_R0":paired_counts(outcomes,"changed","R2","R0"),
            "backup_cue_R2_vs_index_R1":paired_counts(outcomes,"changed","R2","R1"),
        },
        "failure_reasons":{c:{a:{
            reason:sum(reason in row["failure_reasons"] for row in outcomes
                       if row["condition"]==c and row["arm"]==a)
            for reason in ("wrong_first_action","never_reached_cursor_3",
                           "reached_cursor_3_no_verify","verify_false",
                           "repeated_disabled","other")}
            for a in scores[c]} for c in CONDITIONS},
        "shards":[{"shard":sh["shard"],"model":sh["model"],
                   "worlds":[w["world"] for w in sh["worlds"]]} for sh in shard_reports],
        "scientific_pass":False,"product_integrated":False,
        "scope":"post-hoc motivated prospective pilot of one synthetic binary family; external plan, forced prefix and verify controller; not autonomous learning",
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12m_summary.json").write_text(json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        (out/"f12m_full.json").write_text(json.dumps({**report,"worlds":worlds,"outcomes":outcomes},
                                                indent=2,sort_keys=True,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int,choices=range(SHARD_COUNT))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12m_artifacts"))
    args=p.parse_args()
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12m_shard*.json"))
        if len(files)!=SHARD_COUNT:raise ValueError("Missing F12-M shard(s)")
        reports=[json.loads(f.read_text()) for f in files]
        result=aggregate(reports,args.out)
        print(json.dumps({k:result[k] for k in
                          ("study","status","gates","scores","paired","paired_recovery")},
                          sort_keys=True))
        return 0
    if args.shard is None:p.error("Specify --shard 0..3 or --aggregate-dir")
    result=run_shard(FrozenGGUF(),args.shard,args.out)
    print(json.dumps({
        "study":result["study"],"shard":result["shard"],
        "status":"SHARD_ONLY_NO_SCIENTIFIC_CONCLUSION",
        "model":result["model"],"episodes":len(result["outcomes"]),
        "world_count":len(result["worlds"]),
    },sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
