"""F12-J: can frozen LLM execute and revise a VALID evidence-derived policy?

Preregistered at ac1346f BEFORE this implementation. The LLM never
infers the plan; an evidence-only extractor supplies a mechanically valid
plan and documented alternative. The study tests execution separately.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256
from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY="F12-J"
ROLE_MAPS=(
    ("LOAD","ENABLE","READ","PRIMARY","BACKUP","HOLD"),
    ("ENABLE","READ","PRIMARY","BACKUP","HOLD","LOAD"),
    ("READ","PRIMARY","BACKUP","HOLD","LOAD","ENABLE"),
    ("PRIMARY","BACKUP","HOLD","LOAD","ENABLE","READ"),
    ("BACKUP","HOLD","LOAD","ENABLE","READ","PRIMARY"),
    ("HOLD","LOAD","ENABLE","READ","PRIMARY","BACKUP"),
    ("ENABLE","BACKUP","LOAD","PRIMARY","HOLD","READ"),
    ("READ","HOLD","PRIMARY","ENABLE","BACKUP","LOAD"),
)
TOOLS=tuple(f"u{i}" for i in range(6))
ACTIONS=(*TOOLS,"verify")
ARMS=("B0","B1","B2","B3")
CONDITIONS=("stable","changed")
HORIZON=7
CONTEXT_LIMIT=3500
SYSTEM=(
    'You operate a simulated stateful tool world. Return exactly '
    '{"action":"u0"} replacing u0 by one allowed tool or verify. '
    'Read real tool feedback. You have a valid provisional plan built from '
    'public observations; no private hidden goal or roles are exposed.'
)
CATALOG=(
    "Opaque tools u0..u5 can change a binary register, enable operations, "
    "or leave it unchanged. They return public value 0/1, ack, or error. "
    "verify returns true/false. The public goal is to set the complement "
    "of a HIDDEN bit and verify. You may rely only on the observed history."
)
REPAIR=(
    "If the current inversion step returns error DISABLED_TOOL, select "
    "the alternative inversion tool supported by the public episodes, "
    "then verify. Do not retry a permanently unavailable operation."
)

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
    matches=st.search("public observed tool events partial probe feedback",top_k=20)
    out=[json.loads(r.content) for r in matches if r.kind==MemoryKind.EPISODIC]
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

def prompt_for(public_goal,observed,plan_info,history,mode):
    plan=plan_info["plan"]
    cursor=expected_index(plan,plan_info["backup"],history)
    visible=[{"action":e["action"],"response":e["response"]} for e in history]
    instructions=(
        "Use the visible evidence and valid provisional plan; choose the next tool."
        if mode=="B0" else
        f"NEXT EXPECTED PLAN STEP ({cursor+1}/{len(plan)}): {plan[cursor]}. "
        "This cursor advances only after a successful public tool response."
    )
    if mode=="B2":
        instructions+=" "+REPAIR
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
        response=world.execute(action)
        events.append({
            "step":t,"action":action,"response":response,
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
            e["response"]=={"error":"DISABLED_TOOL"} for e in events[:i])
            and e["action"]==plan_info["backup"] and "value" in e["response"]
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

def run(model,out=None):
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    worlds=[]
    for wid in range(8):
        st=new_store()
        try:
            save_probes(st,collect_probes(wid))
            observed=from_store(st)
            plan=grounded_plan(observed)
            if not plan:raise ValueError(f"Unsolvable grounded plan for J{wid}")
            ctx=public_context(observed,plan)
            public_bit=wid%2
            hidden_bit=(wid//2)%2
            # Preflight BEFORE calling any LLM: this should not be confused
            # with learned plan execution or evaluator feedback to the LLM.
            stable=deterministic_case(wid,"stable",public_bit,hidden_bit,plan)
            changed=deterministic_case(wid,"changed",public_bit,hidden_bit,plan)
            if not stable["success"] or not changed["success"]:
                raise ValueError("Reference plan is not valid in this world")
            worlds.append({"world":f"J{wid}","public_bit":public_bit,
                           "hidden_bit_for_audit_only":hidden_bit,
                           "observed":observed,"plan":plan,"context":ctx,
                           "preflight":{"stable":stable,"changed":changed}})
        finally:st.close()
    outcomes=[]
    # Deterministic rotated arm order, worlds fresh for every condition+arm.
    for wid,w in enumerate(worlds):
        for condition in CONDITIONS:
            rotation=(wid+(condition=="changed"))%len(ARMS)
            for arm in ARMS[rotation:]+ARMS[:rotation]:
                if arm=="B3":
                    result=deterministic_case(
                        wid,condition,w["public_bit"],
                        w["hidden_bit_for_audit_only"],w["plan"])
                else:
                    result=run_llm_case(
                        wid,condition,w["public_bit"],
                        w["hidden_bit_for_audit_only"],model,
                        w["context"],w["plan"],arm)
                outcomes.append({**result,"world":w["world"],
                                 "condition":condition,"arm":arm,
                                 "source_ids":w["plan"]["sources"]})
    scores={c:{a:{
        "episodes":len(rows:=[r for r in outcomes if r["condition"]==c and r["arm"]==a]),
        "successes":sum(r["success"] for r in rows),
        "recovered_after_error":sum(r["recovered_after_error"] for r in rows),
        "calls":sum(r["calls"] for r in rows),
        "prompt_tokens":sum(r["prompt_tokens"] for r in rows),
        "output_tokens":sum(r["output_tokens"] for r in rows),
    } for a in ARMS} for c in CONDITIONS}
    s,ch=scores["stable"],scores["changed"]
    metadata=model.metadata() if hasattr(model,"metadata") else {}
    gates={
        "J0_real_model_and_64_clean_paired_episodes":(
            getattr(model,"real_model",False)
            and metadata.get("weights_sha256")==FIXED_MODEL_SHA256
            and len(outcomes)==64 and all(
                scores[c][a]["episodes"]==8 for c in CONDITIONS for a in ARMS)
            and all(len(w["observed"])==10 and len(w["context"])<=CONTEXT_LIMIT
                    for w in worlds)),
        "J1_strong_deterministic_upper_bound":(
            s["B3"]["successes"]==8 and ch["B3"]["successes"]==8),
        "J2_plan_cursor_execution_gain_stable":(
            s["B1"]["successes"]>=6
            and s["B1"]["successes"]>=s["B0"]["successes"]+3),
        "J3_revision_after_public_disabled_error":(
            ch["B2"]["successes"]>=6
            and ch["B2"]["successes"]>=ch["B1"]["successes"]+3
            and ch["B2"]["recovered_after_error"]>=6),
        "J4_no_stable_regression_or_25pct_cost_excess":(
            s["B2"]["successes"]>=s["B1"]["successes"]
            and s["B2"]["calls"]<=1.25*max(1,s["B1"]["calls"])
            and s["B2"]["prompt_tokens"]<=1.25*max(1,s["B1"]["prompt_tokens"])),
        "J5_provenance_and_exact_prompts":all(
            bool(e["exact_prompt"]) and "role" not in e["exact_prompt"]
            for r in outcomes if r["arm"]!="B3" for e in r["events"]),
    }
    status="PILOT_GO_TO_NEW_CONFIRM" if all(gates.values()) else "PILOT_NO_GO"
    if not getattr(model,"real_model",False):status="FAKE_TEST_ONLY"
    elif not gates["J0_real_model_and_64_clean_paired_episodes"]:
        status="INVALID_MODEL_OR_PROTOCOL"
    elif not gates["J1_strong_deterministic_upper_bound"]:
        status="INVALID_REFERENCE_PLAN"
    report={
        "study":STUDY,"status":status,"model":metadata,"worlds":worlds,
        "outcomes":outcomes,"scores":scores,"gates":gates,
        "scientific_pass":False,"confirm_opened":False,"product_integrated":False,
    }
    if out is not None:
        (out/"f12j_full_report.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        (out/"f12j_summary.json").write_text(
            json.dumps({k:v for k,v in report.items() if k not in ("worlds","outcomes")},
                       indent=2,sort_keys=True,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("f12j_artifacts"))
    args=p.parse_args()
    report=run(FrozenGGUF(),args.out)
    print(json.dumps({k:report[k] for k in ("study","status","scores","gates","model")},
                     sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
