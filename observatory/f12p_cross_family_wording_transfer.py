"""F12-P-T: transfer admissible wording to two new tool-state-machine families.

Prospectively preregistered at e7fb7ab1a87a. This module must run only
after F12-P-H passes its strict history/carryover gate.
External plan is derived from public SQLite episodes, never from private world.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from collections import Counter
from itertools import permutations
from pathlib import Path
from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256
from observatory.f12n_public_error_cursor import new_store
from src.infinito3.types import MemoryKind,MemoryRecord

STUDY="F12-P-T"
FAMILIES=("auth","deps")
AUTH_ROLES=("OPEN","AUTHORIZE","SUBMIT","SUBMIT_BACKUP","AUDIT","RESET","DUMMY")
DEPS_ROLES=("SCAN","RESOLVE","LOCK","APPLY","APPLY_BACKUP","NOOP","WIPE")
ROLES_BY_FAMILY={"auth":AUTH_ROLES,"deps":DEPS_ROLES}
TOOLS=tuple(f"r{i}" for i in range(7))
ACTIONS=(*TOOLS,"verify")
ARMS=("E","A","N")
CONDITIONS=("stable","changed","forced")
SHARD_COUNT=4
HORIZON=7
GOAL_BY_FAMILY={
    "auth":"Complete an authorized operation and verify its receipt.",
    "deps":"Deploy a locked dependency bundle and verify the deployment.",
}
FAILURE={"auth":"PROVIDER_DOWN","deps":"CHANNEL_OFFLINE"}
SYSTEM=(
    'You operate a simulated tool service. Return exactly '
    '{"action":"r0"} replacing r0 by one allowed opaque tool or verify. '
    "Read tool feedback and the public, evidence-derived valid plan. "
    "No private role mapping or evaluator state is available."
)
CATALOG=(
    "Opaque tools r0..r6 can have prerequisites or side effects. "
    "Some tools are distractors or reset operations. "
    "Each returns public JSON feedback. "
    "verify returns whether the public goal was achieved. "
    "Use only observed public evidence."
)

def sha(value):return hashlib.sha256(value.encode()).hexdigest()

def selected_maps(family):
    roles=ROLES_BY_FAMILY[family]
    options=[tuple(p) for p in permutations(sorted(roles))]
    options.sort(key=lambda p:(
        sha("F12P_TRANSFER_2026-11-09:"+family+":"+",".join(p)),p))
    selected=tuple(options[:12])
    if len(selected)!=12 or len(set(selected))!=12:
        raise ValueError("F12-P map selection malformed")
    return selected

MAPS={family:selected_maps(family) for family in FAMILIES}

def ident(wid):
    if wid not in range(24):raise ValueError("World not preregistered")
    family=FAMILIES[wid//12]
    index=wid%12
    return family,index,MAPS[family][index]

class World:
    """Private state. Never expose _roles, phase booleans or target to actor."""
    def __init__(self,wid,condition):
        self.family,self.local_index,roles=ident(wid)
        self._roles=dict(zip(TOOLS,roles))
        self.condition=condition
        self.opened=False
        self.authorized=False
        self.submitted=False
        self.scanned=False
        self.resolved=False
        self.locked=False
        self.applied=False
        self.verified=False
    def execute(self,action):
        role=self._roles.get(action,action)
        if role=="verify":
            self.verified=self.submitted if self.family=="auth" else self.applied
            return {"verified":self.verified}
        if self.family=="auth":
            if role=="OPEN":
                self.opened=True
                return {"session":"open"}
            if role=="AUTHORIZE":
                if not self.opened:return {"error":"NO_SESSION"}
                self.authorized=True
                return {"authorized":True}
            if role in ("SUBMIT","SUBMIT_BACKUP"):
                if not self.authorized:return {"error":"UNAUTHORIZED"}
                if role=="SUBMIT" and self.condition=="changed":
                    return {"error":"PROVIDER_DOWN"}
                self.submitted=True
                return {"submitted":True}
            if role=="AUDIT":
                return {"audit":"ready"} if self.opened else {"error":"NO_SESSION"}
            if role=="RESET":
                self.opened=self.authorized=self.submitted=False
                return {"reset":True}
            if role=="DUMMY":return {"noop":True}
        else:
            if role=="SCAN":
                self.scanned=True
                return {"manifest":"seen"}
            if role=="RESOLVE":
                if not self.scanned:return {"error":"NO_MANIFEST"}
                self.resolved=True
                return {"dependencies":"resolved"}
            if role=="LOCK":
                if not self.resolved:return {"error":"NO_DEPENDENCIES"}
                self.locked=True
                return {"lock":"ok"}
            if role in ("APPLY","APPLY_BACKUP"):
                if not self.locked:return {"error":"NOT_LOCKED"}
                if role=="APPLY" and self.condition=="changed":
                    return {"error":"CHANNEL_OFFLINE"}
                self.applied=True
                return {"applied":True}
            if role=="NOOP":return {"noop":True}
            if role=="WIPE":
                self.scanned=self.resolved=self.locked=self.applied=False
                return {"wipe":True}
        return {"error":"INVALID_ACTION"}

PROBES={
    "auth":(
        ("OPEN",),
        ("OPEN","AUTHORIZE"),
        ("OPEN","AUTHORIZE","SUBMIT"),
        ("OPEN","AUTHORIZE","SUBMIT_BACKUP"),
        ("AUTHORIZE",),
        ("AUDIT",),
    ),
    "deps":(
        ("SCAN",),
        ("SCAN","RESOLVE"),
        ("SCAN","RESOLVE","LOCK"),
        ("SCAN","RESOLVE","LOCK","APPLY"),
        ("SCAN","RESOLVE","LOCK","APPLY_BACKUP"),
        ("APPLY",),
    ),
}

def collect_probes(wid):
    family,_,roles=ident(wid)
    by_role={role:tool for tool,role in zip(TOOLS,roles)}
    result=[]
    for i,steps in enumerate(PROBES[family]):
        world=World(wid,"stable")
        events=[]
        for role in steps:
            tool=by_role[role]
            events.append({"action":tool,"response":world.execute(tool)})
        result.append({"source_id":f"P{i}","events":events})
    return result

def store_public(st,records):
    for r in records:
        st.add(MemoryRecord(
            content=json.dumps(r,separators=(",",":"),sort_keys=True),
            kind=MemoryKind.EPISODIC,importance=.9,
            metadata={"study":STUDY,"source_id":r["source_id"],"observed_only":True}
        ))

def from_store(st):
    result=[]
    for i in range(6):
        sid=f"P{i}"
        matches=st.search(sid,top_k=20)
        rec=next((x for x in matches if x.kind==MemoryKind.EPISODIC
                  and x.metadata.get("source_id")==sid),None)
        if rec is None:raise ValueError("Missing SQLite public source "+sid)
        parsed=json.loads(rec.content)
        if set(parsed)!={"source_id","events"}:raise ValueError("Private record leak")
        result.append(parsed)
    if [x["source_id"] for x in result]!=[f"P{i}" for i in range(6)]:
        raise ValueError("Public evidence identity corruption")
    return result

def grounded_plan(family,observed):
    if [x["source_id"] for x in observed]!=[f"P{i}" for i in range(6)]:
        return None
    if any("verified" in e["response"] for r in observed for e in r["events"]):
        return None
    if family=="auth":
        p=observed[2]["events"]
        alt=observed[3]["events"]
        plan=[p[0]["action"],p[1]["action"],p[2]["action"],"verify"]
        backup=alt[-1]["action"]
        valid=(
            len(p)==len(alt)==3 and
            p[0]["response"]=={"session":"open"} and
            p[1]["response"]=={"authorized":True} and
            p[2]["response"]=={"submitted":True} and
            alt[0:2]==p[0:2] and
            alt[2]["response"]=={"submitted":True} and
            observed[4]["events"][-1]["response"]=={"error":"NO_SESSION"}
        )
        sources=["P0","P1","P2","P3","P4"]
    else:
        p=observed[3]["events"]
        alt=observed[4]["events"]
        if len(p)!=4 or len(alt)!=4:return None
        plan=[e["action"] for e in p]+["verify"]
        backup=alt[-1]["action"]
        valid=(
            p[0]["response"]=={"manifest":"seen"} and
            p[1]["response"]=={"dependencies":"resolved"} and
            p[2]["response"]=={"lock":"ok"} and
            p[3]["response"]=={"applied":True} and
            alt[:3]==p[:3] and
            alt[3]["response"]=={"applied":True} and
            observed[5]["events"][-1]["response"]=={"error":"NOT_LOCKED"}
        )
        sources=["P0","P1","P2","P3","P4","P5"]
    if not valid or len(set(plan))!=len(plan) or backup in plan:return None
    return {"plan":plan,"backup":backup,"sources":sources}

def cursor(plan,backup,events):
    n=0
    for e in events:
        if n>=len(plan):break
        good="error" not in e["response"] and e["response"]!={"verified":False}
        if good and (e["action"]==plan[n] or
                     (n==len(plan)-2 and e["action"]==backup)):
            n+=1
    return min(n,len(plan)-1)

def has_public_error(family,plan_info,events):
    plan=plan_info["plan"]
    if (cursor(plan,plan_info["backup"],events)!=len(plan)-2 or
        plan_info["backup"] not in ACTIONS):
        return False
    needed={"P2","P3"} if family=="auth" else {"P3","P4"}
    if not needed.issubset(set(plan_info["sources"])):return False
    return any(e["origin"] in ("llm","controlled_prefix")
               and e["action"]==plan[-2]
               and e["response"]=={"error":FAILURE[family]}
               for e in events)

def action_cue(family,plan_info,events):
    pos=cursor(plan_info["plan"],plan_info["backup"],events)
    return (plan_info["backup"] if has_public_error(family,plan_info,events)
            else plan_info["plan"][pos])

def prompt_for(family,context,plan_info,events,arm):
    if arm not in ARMS:raise ValueError("Unknown arm")
    plan=plan_info["plan"]
    i=cursor(plan,plan_info["backup"],events)
    cue=action_cue(family,plan_info,events)
    if arm=="E":label="NEXT EXPECTED PLAN STEP"
    elif arm=="A":label="NEXT ADMISSIBLE ACTION"
    else:label="NEXT ACTION"
    instruction=(f"{label} ({i+1}/{len(plan)}): {cue}. "
                 "This cursor advances only after a successful public tool response.")
    public=[{"action":e["action"],"response":e["response"]} for e in events]
    return (
        f"PUBLIC GOAL: {GOAL_BY_FAMILY[family]}\nCATALOG: {CATALOG}\n"
        f"VALID PLAN (derived from PUBLIC TOOL evidence): {json.dumps(plan)}\n"
        f"ALTERNATIVE observed tool for the primary step: {plan_info['backup']}\n"
        f"PUBLIC EXPERIENCE REFERENCES:\n{context}\n"
        f"OBSERVED EVENTS IN CURRENT EPISODE: "
        f"{json.dumps(public,separators=(',',':'))}\n"
        f"INSTRUCTION: {instruction}\nReturn one strict JSON action."
    )

def parse_action(raw):
    try:v=json.loads(raw)
    except (TypeError,ValueError):return "INVALID_LLM_OUTPUT"
    return v["action"] if type(v) is dict and set(v)=={"action"} and v["action"] in ACTIONS else "INVALID_LLM_OUTPUT"

def reference_case(wid,condition,plan_info):
    w=World(wid,condition)
    ev=[]
    for action in plan_info["plan"]:
        response=w.execute(action)
        ev.append({"action":action,"response":response})
        if "error" in response and response["error"]==FAILURE[w.family]:
            back=plan_info["backup"]
            ev.append({"action":back,"response":w.execute(back)})
    return {"success":w.verified,"actions":len(ev),"events":ev}

def preflight():
    worlds=[]
    for wid in range(24):
        family,local,roles=ident(wid)
        st=new_store()
        try:
            probes=collect_probes(wid)
            if len(probes)!=6:raise ValueError("Six public probes required")
            store_public(st,probes)
            observed=from_store(st)
            plan=grounded_plan(family,observed)
            if not plan:raise ValueError(f"Public plan ungrounded {wid}")
            context="\n".join(json.dumps(r,sort_keys=True,separators=(",",":"))
                              for r in observed)
            if len(context)>3600:raise ValueError("Public context unexpectedly long")
            controls={cond:reference_case(wid,cond,plan) for cond in ("stable","changed")}
            if not all(c["success"] and c["actions"]<=HORIZON for c in controls.values()):
                raise ValueError("Unsolvable public control")
            w=World(wid,"changed")
            forced=[w.execute(action) for action in plan["plan"][:-1]]
            if forced[-1]!={"error":FAILURE[family]}:
                raise ValueError("Forced prefix not authentic")
            worlds.append({
                "world_index":wid,"world":f"{family.upper()}{local}",
                "family":family,"roles_audit_only":list(roles),
                "observed":observed,"plan":plan,"context":context,
                "preflight":controls,"forced_prefix_feedback":forced,
            })
        finally:st.close()
    if len(worlds)!=24 or len({(w["family"],tuple(w["roles_audit_only"])) for w in worlds})!=24:
        raise ValueError("World uniqueness failed")
    return worlds

def run_episode(wid,condition,context,plan_info,arm,model):
    family,_,_=ident(wid)
    w=World(wid,"changed" if condition=="forced" else condition)
    ev=[]
    if condition=="forced":
        for tool in plan_info["plan"][:-1]:
            idx=cursor(plan_info["plan"],plan_info["backup"],ev)
            ev.append({
                "step":len(ev),"action":tool,"response":w.execute(tool),
                "origin":"controlled_prefix",
                "cursor_before":idx,"recommended":None,"switch":False,
                "prompt":None,"prompt_sha256":None,"raw":None,
                "tokens_in":0,"tokens_out":0,"seconds":0.0,
            })
        if ev[-1]["response"]!={"error":FAILURE[family]}:
            raise ValueError("Forced case missing authentic error")
    for _ in range(HORIZON-len(ev)):
        p=prompt_for(family,context,plan_info,ev,arm)
        before=model.metadata()
        raw=model.chat(SYSTEM,p)
        after=model.metadata()
        action=parse_action(raw)
        idx=cursor(plan_info["plan"],plan_info["backup"],ev)
        recommended=action_cue(family,plan_info,ev)
        switched=has_public_error(family,plan_info,ev)
        response=w.execute(action)
        ev.append({
            "step":len(ev),"action":action,"response":response,
            "origin":"llm","cursor_before":idx,
            "recommended":recommended,"switch":switched,
            "prompt":p,"prompt_sha256":sha(p),"raw":raw,
            "tokens_in":after["prompt_tokens"]-before["prompt_tokens"],
            "tokens_out":after["completion_tokens"]-before["completion_tokens"],
            "seconds":after["inference_seconds"]-before["inference_seconds"],
        })
        if response=={"verified":True}:break
    failure_idx=next((i for i,e in enumerate(ev)
                      if e["action"]==plan_info["plan"][-2]
                      and e["response"]=={"error":FAILURE[family]}),None)
    return {
        "success":w.verified,"events":ev,"actions_total":len(ev),
        "llm_calls":sum(e["origin"]=="llm" for e in ev),
        "prefix_actions":sum(e["origin"]=="controlled_prefix" for e in ev),
        "had_primary_error":failure_idx is not None,
        "switched":any(e["switch"] for e in ev),
        "recovered_after_error":any(e["action"]==plan_info["backup"]
                                    and failure_idx is not None and i>failure_idx
                                    and "error" not in e["response"]
                                    for i,e in enumerate(ev)),
        "preemptive_backup":any(e["action"]==plan_info["backup"]
                                and (failure_idx is None or i<failure_idx)
                                for i,e in enumerate(ev)),
        "repeated_primary_posterror":any(e["action"]==plan_info["plan"][-2]
                                         and failure_idx is not None and i>failure_idx
                                         for i,e in enumerate(ev)),
        "tokens_in":sum(e["tokens_in"] for e in ev),
        "tokens_out":sum(e["tokens_out"] for e in ev),
    }

def metrics(rows):
    return {
        "episodes":len(rows),"successes":sum(r["success"] for r in rows),
        "llm_calls":sum(r["llm_calls"] for r in rows),
        "tokens_in":sum(r["tokens_in"] for r in rows),
        "tokens_out":sum(r["tokens_out"] for r in rows),
        "errors_encountered":sum(r["had_primary_error"] for r in rows),
        "recoveries":sum(r["recovered_after_error"] for r in rows),
        "preemptive_backup":sum(r["preemptive_backup"] for r in rows),
        "repeated_primary":sum(r["repeated_primary_posterror"] for r in rows),
        "budget_exhausted":sum(not r["success"] and r["actions_total"]==HORIZON for r in rows),
    }

def run_shard(model,shard,out=None):
    if shard not in range(4):raise ValueError("Shard out of range")
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    worlds=[w for w in preflight() if w["world_index"]%4==shard]
    if len(worlds)!=6:raise ValueError("Expected 6 worlds per shard")
    outcomes=[]
    for w in worlds:
        wid=w["world_index"]
        for condition in CONDITIONS:
            turn=(wid+CONDITIONS.index(condition))%3
            for arm in ARMS[turn:]+ARMS[:turn]:
                r=run_episode(wid,condition,w["context"],w["plan"],arm,model)
                outcomes.append({**r,"world_index":wid,"family":w["family"],
                                 "condition":condition,"arm":arm})
    report={
        "study":STUDY,"shard":shard,"model":model.metadata(),
        "worlds":worlds,"outcomes":outcomes,
        "status":"SHARD_ONLY_NO_SCIENTIFIC_CONCLUSION",
        "product_integrated":False,
    }
    if out is not None:
        (out/f"f12p_transfer_shard{shard}.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
    return report

def aggregate(shards,out=None):
    if len(shards)!=4 or {x["shard"] for x in shards}!={0,1,2,3}:
        raise ValueError("Four unique transfer shards required")
    worlds=[w for sh in shards for w in sh["worlds"]]
    outcomes=[r for sh in shards for r in sh["outcomes"]]
    actual=[(r["world_index"],r["condition"],r["arm"]) for r in outcomes]
    expected={(i,c,a) for i in range(24) for c in CONDITIONS for a in ARMS}
    if (len(worlds)!=24 or len(outcomes)!=216 or len(set(actual))!=216
        or set(actual)!=expected):raise ValueError("Incomplete transfer data")
    wby={w["world_index"]:w for w in worlds}
    if set(wby)!=set(range(24)):raise ValueError("Missing/duplicate world")
    model_ok=all(
        sh["model"].get("real_model") is True
        and sh["model"].get("weights_sha256")==FIXED_MODEL_SHA256
        and sh["model"].get("revision")=="8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
        and sh["model"].get("backend_version")=="0.3.36"
        and sh["model"].get("temperature")==0.0
        and sh["model"].get("n_threads")==4 for sh in shards)
    world_ok=all(
        w["family"]==ident(w["world_index"])[0]
        and w["roles_audit_only"]==list(ident(w["world_index"])[2])
        and len(w["observed"])==6
        and [r["source_id"] for r in w["observed"]]==[f"P{i}" for i in range(6)]
        and not any(e["action"]=="verify" for r in w["observed"] for e in r["events"])
        for w in worlds)
    reference_ok=all(w["preflight"][c]["success"] and w["preflight"][c]["actions"]<=7
                     for w in worlds for c in ("stable","changed"))
    forced_ok=all(w["forced_prefix_feedback"][-1]=={"error":FAILURE[w["family"]]}
                  for w in worlds)
    valid=True
    switch_ok=True
    stable_no_switch=True
    for row in outcomes:
        w=wby[row["world_index"]]
        ev=row["events"]
        if len(ev)>HORIZON or row["actions_total"]!=len(ev):
            valid=False
        plen=len(w["plan"]["plan"])-1
        if row["condition"]=="forced":
            if len(ev)<plen or row["prefix_actions"]!=plen or row["llm_calls"]>HORIZON-plen:
                valid=False
            if (ev[plen-1]["response"]!={"error":FAILURE[w["family"]]} or
                [x["origin"] for x in ev[:plen]]!=["controlled_prefix"]*plen):
                valid=False
        elif row["prefix_actions"]!=0:valid=False
        if row["condition"]=="stable" and row["switched"]:
            stable_no_switch=False
        history=[]
        for i,e in enumerate(ev):
            if e["step"]!=i or e["cursor_before"]!=cursor(
                w["plan"]["plan"],w["plan"]["backup"],history):
                valid=False
            if e["origin"]=="llm":
                p=prompt_for(w["family"],w["context"],w["plan"],history,row["arm"])
                if (e["prompt"]!=p or e["prompt_sha256"]!=sha(p)
                    or e["action"]!=parse_action(e["raw"])
                    or e["recommended"]!=action_cue(w["family"],w["plan"],history)):
                    valid=False
                if "roles_audit_only" in p or "_roles" in p or "World(" in p:
                    valid=False
                if e["switch"]!=has_public_error(w["family"],w["plan"],history):
                    switch_ok=False
            elif e["origin"]=="controlled_prefix":
                if any(e[k] is not None for k in ("prompt","prompt_sha256","raw","recommended")):
                    valid=False
                if e["switch"] or e["tokens_in"] or e["tokens_out"] or e["seconds"]:
                    valid=False
            else:valid=False
            history.append(e)
    scores={
        fam:{c:{a:metrics([r for r in outcomes if r["family"]==fam
                           and r["condition"]==c and r["arm"]==a])
                for a in ARMS} for c in CONDITIONS}
        for fam in FAMILIES
    }
    pairwise={}
    for fam in FAMILIES:
        pairwise[fam]={}
        for c in CONDITIONS:
            a={r["world_index"]:r["success"] for r in outcomes
               if r["family"]==fam and r["condition"]==c and r["arm"]=="A"}
            e={r["world_index"]:r["success"] for r in outcomes
               if r["family"]==fam and r["condition"]==c and r["arm"]=="E"}
            if len(a)!=12 or set(a)!=set(e):raise ValueError("Incomplete paired family")
            pairwise[fam][c]={
                "A_only":sum(a[i] and not e[i] for i in a),
                "E_only":sum(e[i] and not a[i] for i in a),
                "both":sum(a[i] and e[i] for i in a),
                "neither":sum(not a[i] and not e[i] for i in a),
            }
    T3=all(
        scores[f]["stable"]["A"]["successes"]>=10
        and scores[f]["changed"]["A"]["successes"]>=9
        and scores[f]["forced"]["A"]["successes"]>=10 for f in FAMILIES)
    T4=all(
        (scores[f]["changed"]["A"]["successes"]
         -scores[f]["changed"]["E"]["successes"])>=3
        and (scores[f]["forced"]["A"]["successes"]
             -scores[f]["forced"]["E"]["successes"])>=1
        and scores[f]["stable"]["A"]["successes"]>=scores[f]["stable"]["E"]["successes"]
        and pairwise[f]["changed"]["A_only"]>=2
        for f in FAMILIES)
    avg_tokens={f:{c:{a:scores[f][c][a]["tokens_in"]/max(scores[f][c][a]["llm_calls"],1)
                     for a in ARMS} for c in CONDITIONS} for f in FAMILIES}
    ae_ratio={f:{c:max(avg_tokens[f][c]["A"],avg_tokens[f][c]["E"])/min(
                       avg_tokens[f][c]["A"],avg_tokens[f][c]["E"])
                 if min(avg_tokens[f][c]["A"],avg_tokens[f][c]["E"])>0 else float("inf")
                 for c in CONDITIONS} for f in FAMILIES}
    cost_confound=any(x>1.10 for x_by_c in ae_ratio.values() for x in x_by_c.values())
    gates={
        "T0_real_24_new_family_worlds_216_complete":model_ok and world_ok and valid,
        "T1_public_reference_and_forced_error":reference_ok and forced_ok,
        "T2_only_public_errors_switch_no_stable_switch":switch_ok and stable_no_switch,
        "T3_admissible_robust_each_family":T3,
        "T4_admissible_beats_expected_each_family":T4,
        "T5_trace_costs_and_no_private_leaks":valid and switch_ok,
    }
    if not model_ok:status="FAKE_TEST_ONLY"
    elif not all(gates[k] for k in (
        "T0_real_24_new_family_worlds_216_complete",
        "T1_public_reference_and_forced_error",
        "T2_only_public_errors_switch_no_stable_switch",
        "T5_trace_costs_and_no_private_leaks")):
        status="INVALID_TRANSFER"
    elif T3 and T4 and not cost_confound:
        status="TRANSFER_PILOT_GO_TO_NEW_CONFIRM"
    else:status="TRANSFER_MIXED"
    report={
        "study":STUDY,"status":status,"gates":gates,
        "scores":scores,"paired":pairwise,"average_tokens_per_call":avg_tokens,
        "admissible_expected_cost_ratio":ae_ratio,"prompt_cost_confound":cost_confound,
        "scoped_limitations":"Two synthetic service families; externally authored plan from public probes; no learning, no real MCP or product",
        "product_integrated":False,"scientific_pass":False,
    }
    if out is not None:
        out.mkdir(parents=True,exist_ok=True)
        (out/"f12p_transfer_summary.json").write_text(
            json.dumps(report,indent=2,sort_keys=True,ensure_ascii=False))
        (out/"f12p_transfer_full.json").write_text(
            json.dumps({**report,"worlds":worlds,"outcomes":outcomes},
                       indent=2,sort_keys=True,ensure_ascii=False))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--shard",type=int,choices=(0,1,2,3))
    p.add_argument("--aggregate-dir",type=Path)
    p.add_argument("--out",type=Path,default=Path("f12p_transfer"))
    args=p.parse_args()
    if args.aggregate_dir is not None:
        files=sorted(args.aggregate_dir.glob("f12p_transfer_shard*.json"))
        if len(files)!=4:raise ValueError("Four transfer shards required")
        result=aggregate([json.loads(f.read_text()) for f in files],args.out)
        print(json.dumps({k:result[k] for k in ("study","status","gates","scores","paired","prompt_cost_confound")},
                         sort_keys=True))
        return 0
    if args.shard is None:p.error("Specify --shard or --aggregate-dir")
    report=run_shard(FrozenGGUF(),args.shard,args.out)
    print(json.dumps({"study":STUDY,"shard":args.shard,
                      "episodes":len(report["outcomes"]),"model":report["model"]},sort_keys=True))
    return 0
if __name__=="__main__":raise SystemExit(main())
