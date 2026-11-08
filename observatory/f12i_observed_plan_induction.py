"""F12-I: prospective grounded effect-plan induction, not plan execution by LLM.

Prereg committed as 94e36a4 BEFORE this experiment code or any result.
The researcher schedules ONLY disjoint primitive probes. The frozen real LLM
can see their public results, never the evaluator's hidden role mapping.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from observatory.f12f_evidence_validator import FrozenGGUF, FIXED_MODEL_SHA256
from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY = "F12-I"
ROLE_MAPS = (
    ("LOAD","ENABLE","READ","FLIP","HOLD"),
    ("ENABLE","READ","FLIP","HOLD","LOAD"),
    ("READ","FLIP","HOLD","LOAD","ENABLE"),
    ("FLIP","HOLD","LOAD","ENABLE","READ"),
    ("HOLD","LOAD","ENABLE","READ","FLIP"),
    ("ENABLE","FLIP","LOAD","HOLD","READ"),
    ("FLIP","READ","ENABLE","LOAD","HOLD"),
    ("READ","HOLD","FLIP","ENABLE","LOAD"),
)
TOOLS = tuple(f"t{i}" for i in range(5))
ALLOWED = (*TOOLS, "verify")
ARMS = ("A0","A1","A2","A3")
CONTEXT_LIMIT = 4500
SYSTEM = (
    'Induce one executable tool plan from public evidence, not from assumed tool names. '
    'Return ONLY strict JSON {"plan":["t0",...,"verify"],"sources":["ID",...]}. '
    'The plan must use at most 5 actions, finish with verify, and work for '
    'BOTH possible hidden bits and public input values. Citations must refer '
    'to actual public episodes supplied; use [] if none exist. '
    'No natural language, no extra fields or oracle assumptions.'
)
CATALOG = (
    "Opaque tools t0,t1,t2,t3,t4 may change a binary working register, "
    "enable an operation, or leave it unchanged. Only public tool responses "
    "(value=0/1, ack, error) reveal their effects. verify checks the target "
    "but is not available in probe histories. Goal: store complement of "
    "the HIDDEN bit and verify. Tool names have NO semantic meaning."
)

class World:
    """Evaluator-private semantics; no hidden data in public event records."""
    def __init__(self, world_id, public_bit, hidden_bit):
        self.roles = dict(zip(TOOLS, ROLE_MAPS[world_id]))
        self.public = public_bit
        self.hidden = hidden_bit
        self.selected = False
        self.working = None

    def execute(self, action):
        role = self.roles.get(action, action)
        if role == "LOAD":
            self.working = self.public
            return {"value": self.working}
        if role == "ENABLE":
            self.selected = True
            return {"ack": True}
        if role == "READ":
            if not self.selected:
                return {"error": "precondition"}
            self.working = self.hidden
            return {"value": self.working}
        if role == "FLIP":
            if self.working is None:
                return {"error": "uninitialized"}
            self.working = 1 - self.working
            return {"value": self.working}
        if role == "HOLD":
            if self.working is None:
                return {"error": "uninitialized"}
            return {"value": self.working}
        if role == "verify":
            return {"verified": self.working is not None
                    and self.working == 1 - self.hidden}
        return {"error": "invalid_action"}

# Each probe fully resets the world. Hidden inputs are scheduler-only.
PROBES = (
    ("LOAD",), ("LOAD",),
    ("LOAD","FLIP"), ("LOAD","FLIP"),
    ("ENABLE","READ"), ("ENABLE","READ"),
    ("READ",),
    ("LOAD","HOLD"), ("LOAD","HOLD"),
    ("ENABLE",),
)
PROBE_PUBLIC = (0,1,0,1,0,1,0,0,1,0)
PROBE_HIDDEN = (0,1,0,1,0,1,0,0,1,0)

def collect_probes(world_id):
    inverse = {role: tool for tool, role in zip(TOOLS, ROLE_MAPS[world_id])}
    records = []
    for i, roles in enumerate(PROBES):
        world = World(world_id, PROBE_PUBLIC[i], PROBE_HIDDEN[i])
        events = []
        for role in roles:
            action = inverse[role]
            events.append({"action": action, "response": world.execute(action)})
        records.append({
            "source_id": f"OBS{i}", "public_input": PROBE_PUBLIC[i],
            "events": events,
        })
    return records

def new_store():
    return SQLiteCognitiveMemoryStore(path=":memory:", embedding_provider=HashEmbeddingProvider())

def save_probes(store, public_records):
    for r in public_records:
        store.add(MemoryRecord(
            content=json.dumps(r, sort_keys=True, separators=(",", ":")),
            kind=MemoryKind.EPISODIC, importance=.9,
            metadata={"study": STUDY,"source_id":r["source_id"],"observed_only":True}))

def from_memory(store):
    # Read from real SQLite search and confirm coverage; do not substitute
    # evaluator-private map or fabricate any missing episode.
    hits = store.search("observed actions tool output probe public input", top_k=20)
    parsed = []
    for r in hits:
        if r.kind != MemoryKind.EPISODIC:
            continue
        x = json.loads(r.content)
        if set(x) != {"source_id","public_input","events"}:
            raise ValueError("unexpected or private evidence field")
        parsed.append(x)
    parsed.sort(key=lambda x: int(x["source_id"][3:]))
    if [x["source_id"] for x in parsed] != [f"OBS{i}" for i in range(10)]:
        raise ValueError("F12-I invalid incomplete INFINITO retrieval")
    return parsed

def display_record(x):
    return json.dumps(x,sort_keys=True,separators=(",", ":"),ensure_ascii=False)

def masked_flip_public(records):
    # Remove ONLY the observed second tool response for the two FLIP probes,
    # preserving their IDs and already observed LOAD response.
    return [{**r,"events":r["events"][:1] if i in (2,3) else r["events"]}
            for i,r in enumerate(records)]

def render(records):
    result = "\n".join(display_record(x) for x in records)
    if len(result)>CONTEXT_LIMIT:
        raise ValueError("Public probe context truncated: protocol invalid")
    return result

def parse_plan(raw, allowed_sources, require_citations):
    try:
        x=json.loads(raw)
    except (ValueError,TypeError):
        return {"plan":[],"sources":[],"status":"INVALID_JSON"}
    if type(x) is not dict or set(x)!={"plan","sources"}:
        return {"plan":[],"sources":[],"status":"INVALID_SCHEMA"}
    p,s=x["plan"],x["sources"]
    if (type(p) is not list or type(s) is not list or not (1<=len(p)<=5)
            or p[-1]!="verify" or any(type(t) is not str or t not in ALLOWED for t in p)
            or any(type(t) is not str or t not in allowed_sources for t in s)
            or (require_citations and not s)):
        return {"plan":[],"sources":[],"status":"INVALID_PLAN"}
    return {"plan":p,"sources":s,"status":"OK"}

def generate(model, records, arm):
    evidence = render(records) if arm!="A0" else ""
    prompt = f"GOAL/CATALOG: {CATALOG}\nOBSERVED PUBLIC EPISODES:\n{evidence or '[none]'}\nReturn JSON."
    before=model.metadata() if hasattr(model,"metadata") else {}
    raw=model.chat(SYSTEM,prompt,max_tokens=192)
    after=model.metadata() if hasattr(model,"metadata") else {}
    parsed=parse_plan(raw,{x["source_id"] for x in records},
                      require_citations=arm!="A0")
    return {**parsed, "arm":arm,"raw":raw,"context_text":evidence,
            "exact_prompt":prompt,"prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens":after.get("prompt_tokens",0)-before.get("prompt_tokens",0),
            "output_tokens":after.get("completion_tokens",0)-before.get("completion_tokens",0),
            "inference_seconds":after.get("inference_seconds",0)-before.get("inference_seconds",0)}

def public_effect_planner(records):
    """Researcher-coded role inference; ONLY reads event actions/responses."""
    e={int(r["source_id"][3:]):r for r in records}
    def step(i,j):
        return e[i]["events"][j]
    load=step(0,0)["action"]
    if not (step(0,0)["response"]=={"value":0}
            and step(1,0)=={"action":load,"response":{"value":1}}):
        return {"plan":[],"sources":[],"status":"ABSTAIN_LOAD"}
    flip=step(2,1)["action"]
    if not (step(2,0)["action"]==load and step(3,0)["action"]==load
            and step(2,1)["response"]=={"value":1}
            and step(3,1)=={"action":flip,"response":{"value":0}}):
        return {"plan":[],"sources":[],"status":"ABSTAIN_FLIP"}
    enable=step(4,0)["action"]
    read=step(4,1)["action"]
    if not (step(4,0)["response"]=={"ack":True}
            and step(5,0)=={"action":enable,"response":{"ack":True}}
            and step(4,1)["response"]=={"value":0}
            and step(5,1)=={"action":read,"response":{"value":1}}
            and step(6,0)=={"action":read,"response":{"error":"precondition"}}):
        return {"plan":[],"sources":[],"status":"ABSTAIN_READ"}
    if len({load,flip,enable,read})!=4:
        return {"plan":[],"sources":[],"status":"AMBIGUOUS"}
    return {"plan":[enable,read,flip,"verify"],
            "sources":["OBS2","OBS3","OBS4","OBS5","OBS6"],
            "status":"ENGINEERED_EFFECT_INFERENCE"}

def eval_offline(world_id, plan):
    evaluations=[]
    for public_bit in (0,1):
        for hidden_bit in (0,1):
            world=World(world_id,public_bit,hidden_bit)
            events=[]
            for action in plan:
                response=world.execute(action)
                events.append({"action":action,"response":response})
                if action=="verify":
                    break
            succeeded=(bool(events) and events[-1]["action"]=="verify"
                       and events[-1]["response"]=={"verified":True})
            evaluations.append({"public_bit":public_bit,"hidden_bit":hidden_bit,
                                "success":succeeded,"events":events})
    return evaluations

def run(model,out=None):
    if out is not None:out.mkdir(parents=True,exist_ok=True)
    worlds=[]
    for world_id in range(len(ROLE_MAPS)):
        st=new_store()
        try:
            probes=collect_probes(world_id)
            save_probes(st,probes)
            raw=from_memory(st)
            masked=masked_flip_public(raw)
            proposed={
                "A0":generate(model,[],"A0"),
                "A1":generate(model,raw,"A1"),
                "A2":generate(model,masked,"A2"),
                "A3":{**public_effect_planner(raw),"arm":"A3",
                      "raw":None,"context_text":render(raw),"prompt_tokens":0,
                      "output_tokens":0,"inference_seconds":0},
            }
            results={}
            for arm,proposal in proposed.items():
                trials=eval_offline(world_id,proposal["plan"])
                results[arm]={"full_world":sum(x["success"] for x in trials)==4,
                              "successes":sum(x["success"] for x in trials),
                              "trials":trials}
            worlds.append({
                "world_id":f"I{world_id}","probes":raw,
                "masked_probes":masked,"proposals":proposed,
                "scores":results,"memory_sources":len(st.all()),
            })
        finally:st.close()
    summaries={a:{
        "valid_plans":sum(w["proposals"][a]["status"] in ("OK","ENGINEERED_EFFECT_INFERENCE")
                          for w in worlds),
        "full_worlds":sum(w["scores"][a]["full_world"] for w in worlds),
        "successful_trials":sum(w["scores"][a]["successes"] for w in worlds),
        "prompt_tokens":sum(w["proposals"][a]["prompt_tokens"] for w in worlds)
    } for a in ARMS}
    a1,a0,a2,a3=(summaries[a]["full_worlds"] for a in ("A1","A0","A2","A3"))
    metadata=model.metadata() if hasattr(model,"metadata") else {}
    complete=all(w["memory_sources"]==10 for w in worlds)
    untrained=all(not any("verify" in [e["action"] for e in p["events"]]
                          for p in w["probes"]) for w in worlds)
    support=all(w["proposals"]["A3"]["status"]=="ENGINEERED_EFFECT_INFERENCE"
                for w in worlds)
    citation_support=all(
        set(w["proposals"]["A1"]["sources"]).issubset(
            {r["source_id"] for r in w["probes"]})
        for w in worlds)
    gates={
        "G0_real_pinned_8_worlds_32_plans_128_eval":(
            getattr(model,"real_model",False)
            and metadata.get("weights_sha256")==FIXED_MODEL_SHA256
            and len(worlds)==8 and all(len(w["probes"])==10 for w in worlds)
            and all(len(w["scores"][a]["trials"])==4 for w in worlds for a in ARMS)),
        "G1_genuine_public_probe_coverage_without_target":complete and untrained,
        "G2_LLM_experience_gain":a1>=6 and a1>=a0+3 and a1>=a2+3,
        "G3_engineered_planner_positive_control":a3>=7 and support,
        "G4_exact_observed_source_ids":citation_support,
        "G5_costs_and_raw_provenance_logged":all(
            len(w["proposals"][a]["exact_prompt"])>0
            for w in worlds for a in ("A0","A1","A2")),
    }
    status=("PILOT_GO_TO_NEW_CONFIRM" if all(gates.values()) else "PILOT_NO_GO")
    if not getattr(model,"real_model",False):status="FAKE_TEST_ONLY"
    elif not gates["G0_real_pinned_8_worlds_32_plans_128_eval"]:
        status="INVALID_MODEL_OR_PROTOCOL"
    elif not gates["G1_genuine_public_probe_coverage_without_target"]:
        status="INVALID_PUBLIC_EVIDENCE"
    report={"study":STUDY,"status":status,"model":metadata,
            "worlds":worlds,"summary":summaries,"gates":gates,
            "scientific_pass":False,"confirm_opened":False,"product_integrated":False}
    if out is not None:
        (out/"f12i_full_report.json").write_text(
            json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True))
        (out/"f12i_summary.json").write_text(
            json.dumps({k:v for k,v in report.items() if k!="worlds"},
                       indent=2,ensure_ascii=False,sort_keys=True))
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",type=Path,default=Path("f12i_artifacts"))
    args=p.parse_args()
    result=run(FrozenGGUF(),args.out)
    print(json.dumps({k:result[k] for k in ("study","status","summary","gates","model")},
                     sort_keys=True))
    return 0

if __name__=="__main__":raise SystemExit(main())
