"""F12-N: causal test of cursor action replacement after public tool failure.

Preregistered at 8d01be3a480f BEFORE this implementation. The LLM never
infers the plan; an evidence-only extractor supplies a mechanically valid
plan and documented alternative; recovery uses only authenticated public feedback.
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

STUDY="F12-N"

from observatory.f12j_policy_execution_revision import ROLE_MAPS as USED_J_MAPS
from observatory.f12k_cursor_causal_isolation import ROLE_MAPS as USED_K_MAPS
from observatory.f12lc_cursor_independent_confirmation import ROLE_MAPS as USED_L_MAPS
from observatory.f12m_execution_failure_diagnostic import ROLE_MAPS as USED_M_MAPS

ROLES=("LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD")
LOCK_WORLD="F12N_PUBLIC_CURSOR_2026-11-09:"
LOCK_BITS="F12N_BITS_2026-11-09:"

def hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def frozen_world_maps():
    excluded=set(USED_J_MAPS)|set(USED_K_MAPS)|set(USED_L_MAPS)|set(USED_M_MAPS)
    candidates=[tuple(p) for p in permutations(sorted(ROLES))
                if tuple(p) not in excluded]
    candidates.sort(key=lambda p:(hash_text(LOCK_WORLD+",".join(p)),p))
    result=tuple(candidates[:24])
    if len(result)!=24 or len(set(result))!=24 or set(result)&excluded:
        raise ValueError("F12-N world selection failed frozen uniqueness")
    return result

ROLE_MAPS=frozen_world_maps()
_PAIR_POOL=((0,0),(0,1),(1,0),(1,1))*6
_BIT_ORDER=sorted(range(24),key=lambda i:(hash_text(LOCK_BITS+str(i)),i))
BIT_PAIRS=tuple(_PAIR_POOL[i] for i in _BIT_ORDER)
SHARD_COUNT=4

TOOLS=tuple(f"q{i}" for i in range(6))
ACTIONS=(*TOOLS,"verify")
ARMS=("F0","F1","F2","F3")
CONDITIONS=("stable","changed","forced")
HORIZON=7
CONTEXT_LIMIT=3500
SYSTEM=(
    'You operate a simulated stateful tool world. Return exactly '
    '{"action":"q0"} replacing q0 by one allowed tool or verify. '
    'Read real tool feedback. You have a valid provisional plan built from '
    'public observations; no private hidden goal or roles are exposed.'
)
CATALOG=(
    "Opaque tools q0..q5 can change a binary register, enable operations, "
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


def has_public_disabled_primary(plan_info,events):
    """Only prior authentic public disabled error at the pending inversion."""
    plan=plan_info["plan"]
    if (expected_index(plan,plan_info["backup"],events)!=len(plan)-2
            or plan_info.get("backup") not in ACTIONS
            or not {"P4","P5","P6","P7"}.issubset(set(plan_info.get("sources",[])))):
        return False
    return any(
        e.get("origin") in ("llm","controlled_prefix")
        and e.get("action")==plan[-2]
        and e.get("response")=={"error":"DISABLED_TOOL"}
        for e in events
    )

def shown_action(plan_info,history,mode):
    plan=plan_info["plan"]
    index=expected_index(plan,plan_info["backup"],history)
    if mode=="F2":return None
    if mode=="F0":return plan[index]
    if mode in ("F1","F3"):
        if has_public_disabled_primary(plan_info,history):
            return plan_info["backup"]
        return plan[index]
    raise ValueError("Unknown preregistered F12-N arm")

def prompt_for(public_goal,observed,plan_info,history,mode):
    plan=plan_info["plan"]
    cursor=expected_index(plan,plan_info["backup"],history)
    visible=[{"action":e["action"],"response":e["response"]} for e in history]
    if mode=="F2":
        instructions=(f"CURRENT PLAN POSITION ({cursor+1}/{len(plan)}). "
                      "This cursor advances only after a successful public tool response.")
    elif mode in ("F0","F1"):
        cue=shown_action(plan_info,history,mode)
        instructions=(f"NEXT EXPECTED PLAN STEP ({cursor+1}/{len(plan)}): {cue}. "
                      "This cursor advances only after a successful public tool response.")
    elif mode=="F3":
        cue=shown_action(plan_info,history,mode)
        instructions=(f"PLAN POSITION ({cursor+1}/{len(plan)}). "
                      f"NEXT ADMISSIBLE ACTION: {cue}. "
                      "This cursor advances only after a successful public tool response.")
    else:raise ValueError("Unregistered F12-N arm")
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
    arm=origin.get("arm")
    cue=(shown_action(plan_info,events,arm) if origin["kind"]=="llm" else None)
    switched=bool(
        origin["kind"]=="llm" and arm in ("F1","F3")
        and has_public_disabled_primary(plan_info,events)
        and cue==plan_info["backup"])
    response=world.execute(action)
    tokens=delta or {"prompt_tokens":0,"output_tokens":0,"inference_seconds":0}
    row={
        "step":len(events),"action":action,"response":response,
        "origin":origin["kind"],"arm":arm,
        "real_cursor_index_before":idx,
        "displayed_action_cue":cue,
        "expected_action":plan_info["plan"][idx],
        "switched_after_public_error":switched,
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
    primary=plan_info["plan"][-2]
    first_error_step=next((i for i,e in enumerate(events)
                           if e["action"]==primary
                           and e["response"]=={"error":"DISABLED_TOOL"}),None)
    return {
        "success":world.verified,"events":events,
        "calls":sum(e["origin"]=="llm" for e in events),
        "prefixed_actions":sum(e["origin"]=="controlled_prefix" for e in events),
        "actions_total":len(events),
        "switched_any":any(e["switched_after_public_error"] for e in events),
        "prompt_tokens":sum(e["prompt_tokens"] for e in events),
        "output_tokens":sum(e["output_tokens"] for e in events),
        "first_public_disabled_index":first_error_step,
        "error_encountered":first_error_step is not None,
        "repeated_primary_posterror":any(
            e["action"]==primary for e in events[first_error_step+1:]
        ) if first_error_step is not None else False,
        "recovered_after_error":any(
            e["action"]==plan_info["backup"] and "value" in e["response"]
            and any(prev["action"]==primary
                    and prev["response"]=={"error":"DISABLED_TOOL"}
                    for prev in events[:i])
            for i,e in enumerate(events)),
        "preemptive_backup":any(
            e["action"]==plan_info["backup"] and "value" in e["response"]
            and not any(prev["action"]==primary
                        and prev["response"]=={"error":"DISABLED_TOOL"}
                        for prev in events[:i])
            for i,e in enumerate(events)),
    }

def run_episode(wid,condition,public_bit,hidden_bit,model,context,plan_info,arm):
    if arm not in ARMS or condition not in CONDITIONS:
        raise ValueError("Unknown preregistered arm/condition")
    world=World(wid,public_bit,hidden_bit,"changed" if condition=="forced" else condition)
    events=[]
    if condition=="forced":
        for action in plan_info["plan"][:3]:
            _event(world,plan_info,events,action,{"kind":"controlled_prefix","arm":arm})
        if (len(events)!=3 or events[-1]["response"]!={"error":"DISABLED_TOOL"}):
            raise ValueError("Controlled prefix did not produce authentic disabled response")
    for _ in range(HORIZON-len(events)):
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
    return {
        "episodes":len(rows),
        "successes":sum(r["success"] for r in rows),
        "error_encountered":sum(r["error_encountered"] for r in rows),
        "switched_after_public_error":sum(r["switched_any"] for r in rows),
        "recovered_after_error":sum(r["recovered_after_error"] for r in rows),
        "preemptive_backup":sum(r["preemptive_backup"] for r in rows),
        "repeated_primary_posterror":sum(r["repeated_primary_posterror"] for r in rows),
        "budget_exhausted_unsolved":sum(not r["success"] and r["actions_total"]==HORIZON for r in rows),
        "verified_true":sum(any(e["response"]=={"verified":True}
                                 for e in r["events"]) for r in rows),
        "calls":sum(r["calls"] for r in rows),
        "prefixed_actions":sum(r["prefixed_actions"] for r in rows),
        "actions_total":sum(r["actions_total"] for r in rows),
        "prompt_tokens":sum(r["prompt_tokens"] for r in rows),
        "output_tokens":sum(r["output_tokens"] for r in rows),
        "inference_seconds":sum(e["inference_seconds"] for r in rows for e in r["events"]),
    }


def frozen_preflight():
    worlds=[]
    used=set(USED_J_MAPS)|set(USED_K_MAPS)|set(USED_L_MAPS)|set(USED_M_MAPS)
    if len(ROLE_MAPS)!=24 or len(set(ROLE_MAPS))!=24 or set(ROLE_MAPS)&used:
        raise ValueError("F12-N worlds are not disjoint")
    if Counter(BIT_PAIRS)!={(0,0):6,(0,1):6,(1,0):6,(1,1):6}:
        raise ValueError("Unbalanced F12-N bit pairs")
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
            if not plan:raise ValueError(f"Ungrounded public plan O{wid}")
            ctx=public_context(observed,plan)
            pub,hid=BIT_PAIRS[wid]
            pre={c:deterministic_case(wid,c,pub,hid,plan)
                 for c in ("stable","changed")}
            if not all(r["success"] and r["calls"]<=HORIZON for r in pre.values()):
                raise ValueError("No deterministic public solution")
            world=World(wid,pub,hid,"changed")
            prefix=[world.execute(action) for action in plan["plan"][:3]]
            if prefix[-1]!={"error":"DISABLED_TOOL"}:
                raise ValueError("Seeded public error failed")
            worlds.append({
                "world":f"O{wid}","world_index":wid,
                "roles_audit_only":list(ROLE_MAPS[wid]),
                "public_bit_audit_only":pub,"hidden_bit_audit_only":hid,
                "observed":observed,"context":ctx,"plan":plan,
                "preflight":pre,"forced_prefix_responses":prefix,
            })
        finally:st.close()
    return worlds


def run_shard(model,shard,out=None):
    if shard not in range(SHARD_COUNT):raise ValueError("Unregistered shard")
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    all_worlds=frozen_preflight()  # Full external preflight before any model call.
    worlds=[w for w in all_worlds if w["world_index"]%SHARD_COUNT==shard]
    if len(worlds)!=6:raise ValueError("Expected six worlds per shard")
    outcomes=[]
    for w in worlds:
        wid=w["world_index"]
        pub,hid=w["public_bit_audit_only"],w["hidden_bit_audit_only"]
        for condition in CONDITIONS:
            rotation=(wid+(CONDITIONS.index(condition)))%len(ARMS)
            for arm in ARMS[rotation:]+ARMS[:rotation]:
                result=run_episode(
                    wid,condition,pub,hid,model,w["context"],w["plan"],arm)
                outcomes.append({
                    **result,"world":w["world"],"world_index":wid,
                    "condition":condition,"arm":arm,"sources":w["plan"]["sources"],
                })
    metadata=model.metadata() if hasattr(model,"metadata") else {}
    report={
        "study":STUDY,"shard":shard,"shard_count":SHARD_COUNT,
        "study_status":"SHARD_ONLY_NO_SCIENTIFIC_CONCLUSION",
        "model":metadata,"worlds":worlds,"outcomes":outcomes,
        "scientific_pass":False,"product_integrated":False,
    }
    if out is not None:
        (out/f"f12n_shard{shard}.json").write_text(
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
        raise ValueError("Four different shards required")
    worlds=[w for sh in shard_reports for w in sh["worlds"]]
    outcomes=[r for sh in shard_reports for r in sh["outcomes"]]
    expected={(wid,c,a) for wid in range(24) for c in CONDITIONS for a in ARMS}
    actual=[(r["world_index"],r["condition"],r["arm"]) for r in outcomes]
    if len(worlds)!=24 or len(outcomes)!=288 or len(set(actual))!=288 or set(actual)!=expected:
        raise ValueError("Missing or duplicate F12-N episodes")
    if {w["world_index"] for w in worlds}!=set(range(24)):
        raise ValueError("Missing or duplicate F12-N world")
    if any(w["roles_audit_only"]!=list(ROLE_MAPS[w["world_index"]])
           or (w["public_bit_audit_only"],w["hidden_bit_audit_only"])!=BIT_PAIRS[w["world_index"]]
           for w in worlds):
        raise ValueError("World/bit mapping mismatch")
    real=all(
        sh["model"].get("real_model") is True
        and sh["model"].get("weights_sha256")==FIXED_MODEL_SHA256
        and sh["model"].get("backend_version")=="0.3.36"
        and sh["model"].get("temperature")==0.0
        and sh["model"].get("revision")=="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
        for sh in shard_reports)
    trace_ok=True
    leak_ok=True
    switch_ok=True
    for row in outcomes:
        if row["actions_total"]!=len(row["events"]) or row["actions_total"]>HORIZON:
            trace_ok=False
        if row["condition"]=="forced":
            if (row["prefixed_actions"]!=3 or row["calls"]>4 or len(row["events"])<3
                or [e["origin"] for e in row["events"][:3]]!=["controlled_prefix"]*3
                or row["events"][2]["response"]!={"error":"DISABLED_TOOL"}):
                trace_ok=False
        elif row["prefixed_actions"]!=0:trace_ok=False
        plan=next((w["plan"] for w in worlds if w["world_index"]==row["world_index"]),None)
        if plan is None:raise ValueError("Missing world plan")
        history=[]
        for idx,e in enumerate(row["events"]):
            if e["step"]!=idx or e["expected_action"] not in ACTIONS:
                trace_ok=False
            if e["real_cursor_index_before"]!=expected_index(plan["plan"],plan["backup"],history):
                trace_ok=False
            if e["origin"]=="llm":
                p=e["exact_prompt"]
                if (not p or not e["raw_llm_response"] or
                    hashlib.sha256(p.encode()).hexdigest()!=e["prompt_sha256"] or
                    e["action"]!=parse_action(e["raw_llm_response"])):
                    trace_ok=False
                if ("roles_audit_only" in p or "hidden_bit_audit_only" in p or
                    "ROLE_MAPS" in p or "_hidden" in p):
                    leak_ok=False
                expected_cue=shown_action(plan,history,row["arm"])
                if e["displayed_action_cue"]!=expected_cue:trace_ok=False
                expected_switch=(row["arm"] in ("F1","F3")
                                 and has_public_disabled_primary(plan,history))
                if e["switched_after_public_error"]!=expected_switch:switch_ok=False
                if e["switched_after_public_error"] and (
                        e["displayed_action_cue"]!=plan["backup"] or
                        e["real_cursor_index_before"]!=2):
                    switch_ok=False
            elif e["origin"]=="controlled_prefix":
                if (e["exact_prompt"] is not None or
                    e["raw_llm_response"] is not None or
                    e["prompt_sha256"] is not None or
                    e["prompt_tokens"]!=0 or e["output_tokens"]!=0 or
                    e["inference_seconds"]!=0 or e["displayed_action_cue"] is not None or
                    e["switched_after_public_error"]):
                    trace_ok=False
            else:trace_ok=False
            history.append(e)
    scores={c:{a:metrics([r for r in outcomes if r["condition"]==c and r["arm"]==a])
               for a in ARMS} for c in CONDITIONS}
    matched_stable=True
    for wid in range(24):
        a=next(r for r in outcomes if r["world_index"]==wid and r["condition"]=="stable" and r["arm"]=="F0")
        b=next(r for r in outcomes if r["world_index"]==wid and r["condition"]=="stable" and r["arm"]=="F1")
        sig=lambda r:[(e["action"],e["response"],e["exact_prompt"]) for e in r["events"]]
        if sig(a)!=sig(b) or a["success"]!=b["success"]:
            matched_stable=False
    no_early_switch=all(
        not e["switched_after_public_error"]
        or (row["arm"] in ("F1","F3") and
            row["condition"] in ("changed","forced"))
        for row in outcomes for e in row["events"])
    world_disjoint=not set(ROLE_MAPS)&(
        set(USED_J_MAPS)|set(USED_K_MAPS)|set(USED_L_MAPS)|set(USED_M_MAPS))
    inputs_valid=Counter(BIT_PAIRS)=={(0,0):6,(0,1):6,(1,0):6,(1,1):6}
    source_ok=all(
        len(w["observed"])==10 and
        [r["source_id"] for r in w["observed"]]==[f"P{i}" for i in range(10)]
        for w in worlds)
    ref_ok=all(w["preflight"][c]["success"] and w["preflight"][c]["calls"]<=HORIZON
               for w in worlds for c in ("stable","changed"))
    prefix_ok=all(w["forced_prefix_responses"][-1]=={"error":"DISABLED_TOOL"}
                  for w in worlds)
    def rescue(c):
        pairs=paired_counts(outcomes,c,"F1","F0")
        return pairs["a_only"], pairs["a_only"]-pairs["b_only"]
    natural_rescue,natural_net=rescue("changed")
    forced_rescue,forced_net=rescue("forced")
    G={
        "N0_real_model_24_disjoint_worlds_288_complete":(
            real and len(worlds)==24 and len(outcomes)==288
            and world_disjoint and inputs_valid and source_ok and leak_ok),
        "N1_public_reference_and_authentic_forced_errors":ref_ok and prefix_ok,
        "N2_stable_noop_and_no_premature_switch":(
            matched_stable and no_early_switch and switch_ok
            and scores["stable"]["F1"]["switched_after_public_error"]==0),
        "N3_natural_error_improvement":(
            scores["changed"]["F1"]["successes"]>=20 and
            natural_net>=4 and natural_rescue>=4 and
            scores["changed"]["F1"]["switched_after_public_error"]>=4),
        "N4_forced_error_improvement":(
            scores["forced"]["F1"]["successes"]>=22 and
            forced_net>=4 and forced_rescue>=4 and
            scores["forced"]["F1"]["recovered_after_error"]>=20 and
            scores["forced"]["F1"]["repeated_primary_posterror"]<=2),
        "N5_complete_provenance_and_budget":trace_ok and leak_ok and switch_ok,
        "N6_no_product_or_autonomous_learning_claim":True,
    }
    if not real:
        status="FAKE_TEST_ONLY"
    elif not (G["N0_real_model_24_disjoint_worlds_288_complete"] and
              G["N1_public_reference_and_authentic_forced_errors"] and
              G["N2_stable_noop_and_no_premature_switch"] and
              G["N5_complete_provenance_and_budget"]):
        status="INVALID_PROTOCOL"
    elif G["N3_natural_error_improvement"] and G["N4_forced_error_improvement"]:
        status="PILOT_GO_TO_NEW_CONFIRM"
    elif G["N3_natural_error_improvement"] or G["N4_forced_error_improvement"]:
        status="PILOT_MIXED"
    else:status="PILOT_NO_GO"
    report={
        "study":STUDY,"status":status,"model_sha256":FIXED_MODEL_SHA256,
        "gates":G,"scores":scores,
        "paired":{c:{
            "adaptive_F1_vs_fixed_F0":paired_counts(outcomes,c,"F1","F0"),
            "adaptive_F1_vs_index_F2":paired_counts(outcomes,c,"F1","F2"),
            "structured_F3_vs_inline_F1":paired_counts(outcomes,c,"F3","F1")}
            for c in CONDITIONS},
        "worlds_disjoint":world_disjoint,
        "stable_prompt_traces_identical_F0_F1":matched_stable,
        "shards":[{"id":sh["shard"],"model":sh["model"],
                   "world_ids":[w["world"] for w in sh["worlds"]]}
                  for sh in shard_reports],
        "scientific_pass":False,"product_integrated":False,
        "scope":"24 new permutations of one synthetic binary-state task; external plan and public feedback controller, no autonomous learning or product validation",
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12n_summary.json").write_text(
            json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True))
        (out/"f12n_full.json").write_text(
            json.dumps({**report,"worlds":worlds,"outcomes":outcomes},
                       indent=2,ensure_ascii=False,sort_keys=True))
    return report

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--shard",type=int,choices=range(SHARD_COUNT))
    parser.add_argument("--aggregate-dir",type=Path)
    parser.add_argument("--out",type=Path,default=Path("f12n_artifacts"))
    args=parser.parse_args()
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12n_shard*.json"))
        if len(files)!=SHARD_COUNT:raise ValueError("Four F12-N shard files required")
        data=[json.loads(f.read_text()) for f in files]
        report=aggregate(data,args.out)
        print(json.dumps({k:report[k] for k in ("study","status","gates","scores","paired")},
                         sort_keys=True))
        return 0
    if args.shard is None:parser.error("Specify --shard 0..3 or --aggregate-dir")
    result=run_shard(FrozenGGUF(),args.shard,args.out)
    print(json.dumps({"study":result["study"],"shard":result["shard"],
                      "status":result["study_status"],"episodes":len(result["outcomes"]),
                      "model":result["model"]},sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
