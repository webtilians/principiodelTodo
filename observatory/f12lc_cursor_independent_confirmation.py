"""F12-L-C: independent 24-world confirmation of cursor progress and false cue hazards.

Preregistered at 7c3dab2424b1 BEFORE this implementation. The LLM never
infers the plan; an evidence-only extractor supplies a mechanically valid
plan and documented alternative. This tests F12-K K3/K5 on new worlds.
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

STUDY="F12-L-C"

from observatory.f12j_policy_execution_revision import ROLE_MAPS as USED_J_MAPS
from observatory.f12k_cursor_causal_isolation import ROLE_MAPS as USED_K_MAPS

ROLES=("LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD")
LOCK_WORLD="F12L_CONFIRM_LOCK_2026-11-08:"
LOCK_BITS="F12L_BIT_LOCK_2026-11-08:"

def hash_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()

def frozen_world_maps():
    excluded=set(USED_J_MAPS)|set(USED_K_MAPS)
    candidates=[tuple(p) for p in permutations(sorted(ROLES))
                if tuple(p) not in excluded]
    candidates.sort(key=lambda p:(hash_text(LOCK_WORLD+",".join(p)),p))
    result=tuple(candidates[:24])
    if len(result)!=24 or len(set(result))!=24 or set(result)&excluded:
        raise ValueError("F12-L world selection failed frozen uniqueness")
    return result

ROLE_MAPS=frozen_world_maps()
_PAIR_POOL=((0,0),(0,1),(1,0),(1,1))*6
_BIT_ORDER=sorted(range(24),key=lambda i:(hash_text(LOCK_BITS+str(i)),i))
BIT_PAIRS=tuple(_PAIR_POOL[i] for i in _BIT_ORDER)
SHARD_COUNT=4

TOOLS=tuple(f"w{i}" for i in range(6))
ACTIONS=(*TOOLS,"verify")
ARMS=("C0","C1","C2","C5")
CONDITIONS=("stable","changed")
HORIZON=7
CONTEXT_LIMIT=3500
SYSTEM=(
    'You operate a simulated stateful tool world. Return exactly '
    '{"action":"w0"} replacing u0 by one allowed tool or verify. '
    'Read real tool feedback. You have a valid provisional plan built from '
    'public observations; no private hidden goal or roles are exposed.'
)
CATALOG=(
    "Opaque tools w0..w5 can change a binary register, enable operations, "
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

def shown_action(plan, cursor, mode):
    """Rendered action cue, not an executor; C5 is deliberately stale."""
    if mode in ("C2","C3","C4"):
        return plan[cursor]
    if mode=="C5":
        return plan[min(cursor+1,len(plan)-1)]
    return None

def prompt_for(public_goal,observed,plan_info,history,mode):
    plan=plan_info["plan"]
    cursor=expected_index(plan,plan_info["backup"],history)
    visible=[{"action":e["action"],"response":e["response"]} for e in history]
    if mode=="C0":
        instructions="Use the visible evidence and valid provisional plan; choose the next tool."
    elif mode=="C1":
        instructions=(f"CURRENT PLAN POSITION ({cursor+1}/{len(plan)}). "
                      "This cursor advances only after a successful public tool response.")
    elif mode in ("C2","C3","C4","C5"):
        instructions=(f"NEXT EXPECTED PLAN STEP ({cursor+1}/{len(plan)}): "
                      f"{shown_action(plan,cursor,mode)}. "
                      "This cursor advances only after a successful public tool response.")
        if mode=="C3":instructions+=" "+NEUTRAL
        if mode=="C4":instructions+=" "+REPAIR
    else:
        raise ValueError("Unknown preregistered treatment")
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

def run_llm_case(wid,condition,public_bit,hidden_bit,model,context,plan_info,mode):
    world=World(wid,public_bit,hidden_bit,condition)
    goal="Use the hidden bit, invert it, and verify success."
    events=[]
    for t in range(HORIZON):
        prompt=prompt_for(goal,context,plan_info,events,mode)
        before=model.metadata() if hasattr(model,"metadata") else {}
        raw=model.chat(SYSTEM,prompt)
        after=model.metadata() if hasattr(model,"metadata") else {}
        action=parse_action(raw)
        before_cursor=expected_index(plan_info["plan"],plan_info["backup"],events)
        cue=shown_action(plan_info["plan"],before_cursor,mode)
        response=world.execute(action)
        events.append({
            "step":t,"action":action,"response":response,
            "real_cursor_index_before":before_cursor,
            "displayed_action_cue":cue,
            "expected_action":plan_info["plan"][before_cursor],
            "cue_guard_passed":(cue is None or cue==plan_info["plan"][before_cursor]),
            "raw_llm_response":raw,"exact_prompt":prompt,
            "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens":after.get("prompt_tokens",0)-before.get("prompt_tokens",0),
            "output_tokens":after.get("completion_tokens",0)-before.get("completion_tokens",0),
            "inference_seconds":after.get("inference_seconds",0)-before.get("inference_seconds",0),
        })
        if response=={"verified":True}:break
    return {
        "success":world.verified,"events":events,"calls":len(events),
        "prompt_tokens":sum(e["prompt_tokens"] for e in events),
        "output_tokens":sum(e["output_tokens"] for e in events),
        "recovered_after_error":any(
            e["action"]==plan_info["backup"] and "value" in e["response"]
            and any(prev["response"]=={"error":"DISABLED_TOOL"}
                    for prev in events[:i])
            for i,e in enumerate(events)
        ),
    }

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
    wrong_follow=0
    for row in rows:
        if row["events"]:
            first=row["events"][0]
            if (first["displayed_action_cue"] is not None
                    and first["displayed_action_cue"]!=first["expected_action"]
                    and first["action"]==first["displayed_action_cue"]):
                wrong_follow+=1
    return {
        "episodes":len(rows),
        "successes":sum(r["success"] for r in rows),
        "recovered_after_error":sum(r["recovered_after_error"] for r in rows),
        "wrong_first_cue_followed":wrong_follow,
        "calls":sum(r["calls"] for r in rows),
        "prompt_tokens":sum(r["prompt_tokens"] for r in rows),
        "output_tokens":sum(r["output_tokens"] for r in rows),
        "prompt_characters":sum(len(e["exact_prompt"]) for r in rows for e in r["events"]),
        "inference_seconds":sum(e["inference_seconds"] for r in rows for e in r["events"]),
    }

def frozen_preflight():
    worlds=[]
    if len(ROLE_MAPS)!=24 or len(set(ROLE_MAPS))!=24:
        raise ValueError("Wrong number of novel world maps")
    if set(ROLE_MAPS)&(set(USED_J_MAPS)|set(USED_K_MAPS)):
        raise ValueError("Reused J/K worlds")
    if Counter(BIT_PAIRS)!={(0,0):6,(0,1):6,(1,0):6,(1,1):6}:
        raise ValueError("Bit pair stratum imbalance")
    for wid in range(24):
        st=new_store()
        try:
            original=collect_probes(wid)
            # No probe contains target full four-step program.
            if len(original)!=10 or any(
                    any(ev["action"]=="verify" for ev in x["events"])
                    for x in original):
                raise ValueError(f"Invalid probing in L{wid}")
            save_probes(st,original)
            observed=from_store(st)
            plan=grounded_plan(observed)
            if not plan:raise ValueError(f"Ungrounded public policy L{wid}")
            ctx=public_context(observed,plan)
            pub,hid=BIT_PAIRS[wid]
            stable=deterministic_case(wid,"stable",pub,hid,plan)
            changed=deterministic_case(wid,"changed",pub,hid,plan)
            if not(stable["success"] and changed["success"]
                   and stable["calls"]<=HORIZON and changed["calls"]<=HORIZON):
                raise ValueError("Invalid reference policy")
            # Static guard: names must match the publicly indexed plan.
            correct=shown_action(plan["plan"],0,"C2")
            wrong=shown_action(plan["plan"],0,"C5")
            initial_guard={"correct_accepted":correct==plan["plan"][0],
                           "false_rejected":wrong!=plan["plan"][0],
                           "reported_bad_name":wrong,
                           "reported_good_name":correct}
            if not all(initial_guard[k] for k in ("correct_accepted","false_rejected")):
                raise ValueError("Security guard preflight failed")
            worlds.append({
                "world":f"L{wid}","world_index":wid,
                "roles_audit_only":list(ROLE_MAPS[wid]),
                "public_bit_audit_only":pub,"hidden_bit_audit_only":hid,
                "observed":observed,"plan":plan,"context":ctx,
                "preflight":{"stable":stable,"changed":changed},
                "guard_initial":initial_guard,
            })
        finally:st.close()
    return worlds

def run_shard(model,shard,out=None):
    if shard not in range(SHARD_COUNT):
        raise ValueError("Unregistered shard")
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    worlds_all=frozen_preflight()  # before first LLM inference
    worlds=[w for w in worlds_all if w["world_index"]%SHARD_COUNT==shard]
    if len(worlds)!=6:raise ValueError("Shard must contain exactly six worlds")
    outcomes=[]
    for w in worlds:
        wid=w["world_index"]
        for condition in CONDITIONS:
            rotation=(wid+(condition=="changed"))%len(ARMS)
            for arm in ARMS[rotation:]+ARMS[:rotation]:
                result=run_llm_case(
                    wid,condition,w["public_bit_audit_only"],
                    w["hidden_bit_audit_only"],model,w["context"],w["plan"],arm)
                outcomes.append({
                    **result,"world":w["world"],"world_index":wid,
                    "condition":condition,"arm":arm,"sources":w["plan"]["sources"],
                })
    metadata=model.metadata() if hasattr(model,"metadata") else {}
    shard_result={
        "study":STUDY,"shard":shard,"shard_count":SHARD_COUNT,
        "study_status":"SHARD_ONLY_NOT_SCIENTIFIC_RESULT",
        "model":metadata,"worlds":worlds,"outcomes":outcomes,
        "scientific_pass":False,"confirm_complete":False,
        "product_integrated":False,
    }
    if out is not None:
        (out/f"f12lc_shard{shard}.json").write_text(
            json.dumps(shard_result,indent=2,ensure_ascii=False,sort_keys=True))
    return shard_result

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
    if len(shard_reports)!=SHARD_COUNT:
        raise ValueError("Missing sharded confirmations")
    if {r["shard"] for r in shard_reports}!=set(range(SHARD_COUNT)):
        raise ValueError("Missing or duplicate shard IDs")
    worlds=[w for shard in shard_reports for w in shard["worlds"]]
    outcomes=[r for shard in shard_reports for r in shard["outcomes"]]
    expected={(wid,c,a) for wid in range(24) for c in CONDITIONS for a in ARMS}
    actual=[(r["world_index"],r["condition"],r["arm"]) for r in outcomes]
    if (len(worlds)!=24 or len(outcomes)!=192 or
            len(set(actual))!=192 or set(actual)!=expected or
            {w["world_index"] for w in worlds}!=set(range(24))):
        raise ValueError("Malformed or incomplete aggregate results")
    if any(w["roles_audit_only"]!=list(ROLE_MAPS[w["world_index"]])
           or (w["public_bit_audit_only"],w["hidden_bit_audit_only"])!=BIT_PAIRS[w["world_index"]]
           for w in worlds):
        raise ValueError("Shard world mapping or bits mismatch")
    if any(r["study"]!=STUDY or r["shard_count"]!=SHARD_COUNT
           for r in shard_reports):
        raise ValueError("Wrong shard metadata")
    scores={c:{a:metrics([r for r in outcomes if r["condition"]==c and r["arm"]==a])
               for a in ARMS} for c in CONDITIONS}
    s,h=scores["stable"],scores["changed"]
    wrong_first=s["C5"]["wrong_first_cue_followed"]
    secure=all(
        w["guard_initial"]["correct_accepted"] and
        w["guard_initial"]["false_rejected"]
        for w in worlds)
    prompt_integrity=all(
        e["exact_prompt"] and
        hashlib.sha256(e["exact_prompt"].encode()).hexdigest()==e["prompt_sha256"]
        and "_hidden" not in e["exact_prompt"] and "ROLE_MAPS" not in e["exact_prompt"]
        and e["expected_action"] in ACTIONS
        and e["real_cursor_index_before"]>=0
        and (e["cue_guard_passed"]==
             (e["displayed_action_cue"] is None or
              e["displayed_action_cue"]==e["expected_action"]))
        for r in outcomes for e in r["events"])
    real=all(r["model"].get("weights_sha256")==FIXED_MODEL_SHA256
             and r["model"].get("real_model") is True
             and r["model"].get("backend_version")=="0.3.36"
             and r["model"].get("temperature")==0.0
             for r in shard_reports)
    maps_valid=(len(set(ROLE_MAPS))==24 and
                not set(ROLE_MAPS)&(set(USED_J_MAPS)|set(USED_K_MAPS)))
    if not real and any(r["model"].get("real_model") for r in shard_reports):
        raise ValueError("Mixed model validity, abort")
    gates={
        "L0_reproducible_real_model_24_new_worlds_192_complete_episodes":(
            real and maps_valid and Counter(BIT_PAIRS)==
                {(0,0):6,(0,1):6,(1,0):6,(1,1):6}
            and len(outcomes)==192 and prompt_integrity),
        "L1_public_only_reference_solves_24_each":all(
            w["preflight"][c]["success"] and w["preflight"][c]["calls"]<=HORIZON
            for w in worlds for c in CONDITIONS),
        "L2_confirm_numeric_cursor_both_conditions":(
            s["C1"]["successes"]>=20
            and s["C1"]["successes"]-s["C0"]["successes"]>=12
            and h["C1"]["successes"]>=16
            and h["C1"]["successes"]-h["C0"]["successes"]>=8),
        "L3_confirm_bad_action_cue_hazard":(
            s["C2"]["successes"]>=20
            and s["C2"]["successes"]-s["C5"]["successes"]>=12
            and wrong_first>=18 and secure),
        "L4_changed_condition_contextual_robustness":(
            h["C1"]["successes"]>=16 and h["C2"]["successes"]>=16),
        "L5_complete_shard_trace_cost_and_guard_audit":(
            len(shard_reports)==4 and len(outcomes)==192 and
            len(worlds)==24 and prompt_integrity and secure and
            all(len(w["observed"])==10
                and [e["source_id"] for e in w["observed"]]==
                    [f"P{i}" for i in range(10)]
                for w in worlds)),
        "L6_no_autonomous_learning_or_product_claim":True,
    }
    if not real:
        status="FAKE_TEST_ONLY"
    elif not(all(gates[k] for k in (
        "L0_reproducible_real_model_24_new_worlds_192_complete_episodes",
        "L1_public_only_reference_solves_24_each",
        "L5_complete_shard_trace_cost_and_guard_audit"))):
        status="INVALID_OR_INCOMPLETE"
    elif not(gates["L2_confirm_numeric_cursor_both_conditions"] and
             gates["L3_confirm_bad_action_cue_hazard"]):
        status="CONFIRM_FAIL"
    elif not gates["L4_changed_condition_contextual_robustness"]:
        status="CONFIRM_PARTIAL_NO_ROBUSTNESS"
    else:
        status="CONFIRM_PASS_SYNTHETIC_CURSOR"
    report={
        "study":STUDY,"status":status,"gates":gates,"scores":scores,
        "paired":{c:{
            "C1_vs_C0":paired_counts(outcomes,c,"C1","C0"),
            "C2_vs_C5":paired_counts(outcomes,c,"C2","C5")}
            for c in CONDITIONS},
        "security_guard":{"false_initial_cues_rejected":sum(w["guard_initial"]["false_rejected"] for w in worlds),
                          "correct_initial_cues_accepted":sum(w["guard_initial"]["correct_accepted"] for w in worlds)},
        "shards":[{"shard":r["shard"],"model":r["model"],
                   "worlds":[w["world"] for w in r["worlds"]]}
                  for r in shard_reports],
        "scope_limitations":"24 correlated synthetic worlds of one binary task family; externally engineered policy and cursor; no autonomous learning, real MCP or product evidence",
        "scientific_pass_synthetic_only":status=="CONFIRM_PASS_SYNTHETIC_CURSOR",
        "product_integrated":False,
        "model_sha256":FIXED_MODEL_SHA256,
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12lc_confirm_summary.json").write_text(
            json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True))
        (out/"f12lc_confirm_full.json").write_text(
            json.dumps({**report,"worlds":worlds,"outcomes":outcomes},
                       indent=2,ensure_ascii=False,sort_keys=True))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int,choices=range(SHARD_COUNT))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12lc_artifacts"))
    args=p.parse_args()
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12lc_shard*.json"))
        if len(files)!=SHARD_COUNT:
            raise ValueError(f"Aggregate requires 4 shard files, found {len(files)}")
        reports=[json.loads(f.read_text()) for f in files]
        result=aggregate(reports,args.out)
        print(json.dumps({k:result[k] for k in ("study","status","gates","scores","security_guard","paired")},sort_keys=True))
        return 0
    if args.shard is None:
        p.error("Specify --shard 0..3 or --aggregate-dir")
    result=run_shard(FrozenGGUF(),args.shard,args.out)
    print(json.dumps({
        "study":result["study"],"shard":result["shard"],
        "status":"SHARD_ONLY_NOT_SCIENTIFIC_RESULT",
        "model":result["model"],
        "episodes":len(result["outcomes"]),
        "world_count":len(result["worlds"]),
    },sort_keys=True))
    return 0

if __name__=="__main__":
    raise SystemExit(main())
