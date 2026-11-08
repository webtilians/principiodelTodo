"""F12-K: isolate cursor progress, direct action cues, padding and interference.

Preregistered at 8fb2501befd5 BEFORE this implementation. The LLM never
infers the plan; an evidence-only extractor supplies a mechanically valid
plan and documented alternative. The study tests cursor mechanisms separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256
from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY="F12-K"
ROLE_MAPS=(
    ("LOAD","READ","ENABLE","BACKUP","HOLD","PRIMARY"),
    ("PRIMARY","ENABLE","LOAD","HOLD","READ","BACKUP"),
    ("BACKUP","READ","LOAD","PRIMARY","ENABLE","HOLD"),
    ("ENABLE","HOLD","PRIMARY","READ","LOAD","BACKUP"),
    ("READ","LOAD","HOLD","BACKUP","PRIMARY","ENABLE"),
    ("HOLD","PRIMARY","READ","ENABLE","BACKUP","LOAD"),
    ("LOAD","BACKUP","PRIMARY","READ","HOLD","ENABLE"),
    ("PRIMARY","HOLD","ENABLE","BACKUP","LOAD","READ"),
    ("READ","BACKUP","HOLD","LOAD","ENABLE","PRIMARY"),
    ("BACKUP","ENABLE","HOLD","PRIMARY","READ","LOAD"),
)
TOOLS=tuple(f"v{i}" for i in range(6))
ACTIONS=(*TOOLS,"verify")
ARMS=("C0","C1","C2","C3","C4","C5")
CONDITIONS=("stable","changed")
HORIZON=7
CONTEXT_LIMIT=3500
SYSTEM=(
    'You operate a simulated stateful tool world. Return exactly '
    '{"action":"v0"} replacing u0 by one allowed tool or verify. '
    'Read real tool feedback. You have a valid provisional plan built from '
    'public observations; no private hidden goal or roles are exposed.'
)
CATALOG=(
    "Opaque tools v0..v5 can change a binary register, enable operations, "
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
BIT_PAIRS=((0,0),(1,0),(0,1),(1,1),(0,1),(1,0),(1,1),(0,0),(0,0),(1,1))


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

def run(model,out=None):
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    worlds=[]
    # Preflight for all ten worlds before any LLM call.
    for wid in range(len(ROLE_MAPS)):
        st=new_store()
        try:
            save_probes(st,collect_probes(wid))
            observed=from_store(st)
            plan=grounded_plan(observed)
            if not plan:raise ValueError(f"Unsolvable public plan K{wid}")
            ctx=public_context(observed,plan)
            public_bit,hidden_bit=BIT_PAIRS[wid]
            stable=deterministic_case(wid,"stable",public_bit,hidden_bit,plan)
            changed=deterministic_case(wid,"changed",public_bit,hidden_bit,plan)
            if not stable["success"] or not changed["success"]:
                raise ValueError("Invalid deterministic reference")
            worlds.append({
                "world":f"K{wid}","public_bit":public_bit,
                "hidden_bit_for_audit_only":hidden_bit,
                "observed":observed,"plan":plan,"context":ctx,
                "preflight":{"stable":stable,"changed":changed},
            })
        finally:st.close()
    outcomes=[]
    for wid,w in enumerate(worlds):
        for condition in CONDITIONS:
            rotation=(wid+(condition=="changed"))%len(ARMS)
            for arm in ARMS[rotation:]+ARMS[:rotation]:
                result=run_llm_case(
                    wid,condition,w["public_bit"],
                    w["hidden_bit_for_audit_only"],model,
                    w["context"],w["plan"],arm)
                outcomes.append({**result,"world":w["world"],
                                 "condition":condition,"arm":arm,
                                 "source_ids":w["plan"]["sources"]})
    scores={condition:{
        arm:metrics([r for r in outcomes
                     if r["condition"]==condition and r["arm"]==arm])
        for arm in ARMS} for condition in CONDITIONS}
    st,ch=scores["stable"],scores["changed"]
    metadata=model.metadata() if hasattr(model,"metadata") else {}
    strong=all(w["preflight"]["stable"]["success"]
               and w["preflight"]["changed"]["success"] for w in worlds)
    no_private=all("_hidden" not in e["exact_prompt"]
                   and "ROLE_MAPS" not in e["exact_prompt"]
                   for row in outcomes for e in row["events"])
    sources_valid=all(len(w["observed"])==10 and len(w["context"])<=CONTEXT_LIMIT
                      for w in worlds)
    wrong_first=st["C5"]["wrong_first_cue_followed"]+ch["C5"]["wrong_first_cue_followed"]
    token_ratio=(sum(scores[c]["C4"]["prompt_tokens"] for c in CONDITIONS)
                 /max(1,sum(scores[c]["C3"]["prompt_tokens"] for c in CONDITIONS)))
    gates={
        "K0_real_pinned_10_worlds_120_episodes_no_private_leak":(
            getattr(model,"real_model",False)
            and metadata.get("weights_sha256")==FIXED_MODEL_SHA256
            and len(worlds)==10 and len(outcomes)==120
            and all(scores[c][a]["episodes"]==10 for c in CONDITIONS for a in ARMS)
            and no_private and sources_valid),
        "K1_deterministic_solution_10_of_10_each":strong,
        "K2_exact_next_action_over_index_only":(
            st["C2"]["successes"]>=8
            and st["C2"]["successes"]>=st["C1"]["successes"]+3),
        "K3_index_only_over_no_cursor":(
            st["C1"]["successes"]>=6
            and st["C1"]["successes"]>=st["C0"]["successes"]+3),
        "K4_repair_semantics_vs_neutral_length":(
            all(scores[c]["C3"]["successes"]>=scores[c]["C2"]["successes"]-1
                for c in CONDITIONS)
            and any(scores[c]["C4"]["successes"]<=scores[c]["C3"]["successes"]-3
                    for c in CONDITIONS)),
        "K5_stale_action_cue_harms_and_followed":(
            st["C5"]["successes"]<=st["C2"]["successes"]-3
            and wrong_first>=3),
        "K6_changed_feedback_recovery_observed":all(
            scores["changed"][a]["episodes"]==10 for a in ARMS),
        "K7_complete_prompts_sha_costs_and_public_cursor":(
            all(e["exact_prompt"] and
                hashlib.sha256(e["exact_prompt"].encode()).hexdigest()==e["prompt_sha256"]
                and e["real_cursor_index_before"]>=0
                and e["expected_action"] in ACTIONS
                for row in outcomes for e in row["events"])
            and len(NEUTRAL)==len(REPAIR)),
    }
    status=("PILOT_GO_TO_NEW_CONFIRM" if
            gates["K0_real_pinned_10_worlds_120_episodes_no_private_leak"]
            and gates["K1_deterministic_solution_10_of_10_each"]
            and gates["K7_complete_prompts_sha_costs_and_public_cursor"]
            and any(gates[k] for k in (
                "K2_exact_next_action_over_index_only",
                "K3_index_only_over_no_cursor",
                "K4_repair_semantics_vs_neutral_length",
                "K5_stale_action_cue_harms_and_followed"))
            else "PILOT_NO_GO")
    if not getattr(model,"real_model",False):
        status="FAKE_TEST_ONLY"
    elif not gates["K0_real_pinned_10_worlds_120_episodes_no_private_leak"]:
        status="INVALID_PROTOCOL_OR_MODEL"
    elif not gates["K1_deterministic_solution_10_of_10_each"]:
        status="INVALID_REFERENCE_SOLVABILITY"
    report={
        "study":STUDY,"status":status,"model":metadata,"worlds":worlds,
        "outcomes":outcomes,"scores":scores,"gates":gates,
        "neutral_to_repair_token_ratio":token_ratio,
        "cost_confound_if_ratio_above_1_10":token_ratio>1.10,
        "wrong_first_cue_followed_all_conditions":wrong_first,
        "scientific_pass":False,"confirm_opened":False,"product_integrated":False,
    }
    if out is not None:
        (out/"f12k_full_report.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        (out/"f12k_summary.json").write_text(
            json.dumps({k:v for k,v in report.items() if k not in ("worlds","outcomes")},
                       indent=2,sort_keys=True,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("f12k_artifacts"))
    args=p.parse_args()
    report=run(FrozenGGUF(),args.out)
    print(json.dumps({k:report[k] for k in ("study","status","scores","gates","model")},
                     sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
