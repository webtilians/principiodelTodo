"""E10-D: ONE predeclared general declarative capture + bilingual plan-intent fix.

All modifications are injected EXPERIMENT-LOCALLY. src/infinito3 remains frozen.
No learned gate, no domain-specific capture rule, no historical E10-B data reuse.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
import re
import statistics
import tempfile
from collections import Counter
from pathlib import Path

from src.infinito3.cognitive_loop import CognitiveLoop
from src.infinito3.context_builder import ApproximateTokenEstimator
from src.infinito3.engine import CognitiveEngine
from src.infinito3.generalized_context_builder import GeneralizedContextBuilder
from src.infinito3.goals import SimpleGoalEngine
from src.infinito3.llm_adapter import OpenAIResponsesAdapter
from src.infinito3.memory import RuleBasedMemoryGate
from src.infinito3.persistent_memory import SQLiteCognitiveMemoryStore
from src.infinito3.types import GateDecision,MemoryKind
from observatory.e10b_semantic_longitudinal_live import (
    ACTIONS, NAMES, NOISE, ARMS as ORIGINAL_ARMS, MODEL,
    REASONING_EFFORT, EMBEDDING_MODEL, CONTEXT_BUDGET, TOP_K,OUTPUT_TOKENS,
    _RESPONSE_INSTRUCTIONS,ExperimentArm,CachedSemanticEmbeddings,parse_action,
)
from observatory.e10b_semantic_longitudinal_live import World as BaseWorld

DEV_BASE=2100000
CONFIRM_BASE=2200000
FAMILIES=("owner","backup","audience","retention","favorite_tool","schedule_plan")
ARMS=("no_memory","semantic_rag","infinito3","infinito3_gate_open","e10d_intervention")
OPTIONS={
    "owner":("equipo Ámbar","equipo Marino","equipo Cedro","equipo Lince","equipo Sur","equipo Delta"),
    "backup":("almacén Faro","almacén Nube","almacén Olivo","almacén Cedro","almacén Brisa","almacén Río"),
    "audience":("personal de soporte","equipo de producto","personal de calidad","equipo jurídico","equipo comercial","usuarios internos"),
    "retention":("45 días","90 días","120 días","180 días","365 días","dos años"),
    "favorite_tool":("Visor A","Editor B","Panel C","Consola D","Tablero E","Portal F"),
    "schedule_plan":("a las nueve","a las once","a las quince","al mediodía","al anochecer","a las siete"),
}
FACET={
    "owner":"el equipo que aprueba el documento",
    "backup":"el depósito de las copias de seguridad",
    "audience":"el destinatario del resumen operativo",
    "retention":"el plazo de conservación de los archivos",
    "favorite_tool":"la herramienta que acordamos utilizar",
    "schedule_plan":"el horario de revisión del plan",
}

class BroadDeclarativeGate:
    """Experimental nonlearning baseline: store safe sufficiently long statements.

    Original RuleBasedMemoryGate is kept for all of its prior positive
    classifications. This one extension is DOMAIN-AGNOSTIC and costs storage.
    """
    def __init__(self):
        self.original=RuleBasedMemoryGate()
    def evaluate(self,text:str)->GateDecision:
        first=self.original.evaluate(text)
        if first.should_store:
            return first
        tokens=re.findall(r"[\wáéíóúüñ]+",text.lower(),flags=re.I)
        if len(tokens)>=5 and "?" not in text and "¿" not in text:
            return GateDecision(True,0.60,MemoryKind.EPISODIC,
                                ["general_declarative_five_tokens"])
        return first

class FixedPlanIntentBuilder(GeneralizedContextBuilder):
    """Only the Spanish 'plan' / English 'plans?' intent collision is changed."""
    @classmethod
    def _asks_for_goals(cls,query:str)->bool:
        normalized=cls._normalized(query)
        if re.search(r"\b(?:my|our|the|current|pending|upcoming|future)\s+plan\b",normalized):
            return True
        # Leave 'plans' plural (and all other task/goal markers) untouched.
        # Bare 'plan' in Spanish must not trigger English-goal retrieval mode.
        without_bare_plan=re.sub(r"\bplan\b"," ",normalized)
        return super()._asks_for_goals(without_bare_plan)


def seed_for(split:str,index:int,j:int)->int:
    if split=="dev" and index==0 and j in range(6):
        return DEV_BASE+j
    if split=="confirm" and index in range(4) and j in range(6):
        return CONFIRM_BASE+1000*index+j
    raise ValueError("E10-D: only preregistered fresh splits")


def world_for(seed:int)->BaseWorld:
    if seed not in {seed_for("dev",0,j) for j in range(6)} and seed not in {
        seed_for("confirm",i,j) for i in range(4) for j in range(6)}:
        raise ValueError("Cannot use older E10 or F9 seeds")
    rng=random.Random(seed)
    family=FAMILIES[seed%1000]
    subject=f"{rng.choice(NAMES)}-{rng.randrange(100,999)}"
    return BaseWorld(
        seed=seed,family=family,subject=subject,
        options=tuple(rng.sample(OPTIONS[family],2)),
        actions=tuple(rng.sample(ACTIONS,2)),
        distractors_before=tuple(rng.choice(NOISE) for _ in range(12)),
        distractors_after=tuple(rng.choice(NOISE) for _ in range(12)),
    )

def fact(w:BaseWorld,value:str,updated:bool)->str:
    if w.family=="owner":
        return (f"La aprobación del documento {w.subject} corresponde ahora al {value}; este acuerdo reemplaza el anterior."
                if updated else f"Para el documento {w.subject} acordamos que el {value} se encargará de aprobarlo.")
    if w.family=="backup":
        return (f"El depósito vigente de las copias del sistema {w.subject} se cambia a {value}; el anterior deja de utilizarse."
                if updated else f"Las copias de seguridad del sistema {w.subject} se depositarán en {value} según lo acordado.")
    if w.family=="audience":
        return (f"Se ha actualizado el destino del resumen {w.subject}: a partir de hoy se envía al {value}, no al anterior."
                if updated else f"El resumen operativo {w.subject} se prepara específicamente para {value}.")
    if w.family=="retention":
        return (f"Rectificamos la política para archivos {w.subject}. Ahora deben conservarse {value}, sustituyendo el plazo previo."
                if updated else f"Para los archivos {w.subject} fijamos una conservación de {value}.")
    if w.family=="favorite_tool":
        return (f"Al revisar el procedimiento {w.subject} cambiamos la herramienta elegida y desde ahora usaremos {value}."
                if updated else f"Para el procedimiento {w.subject} hemos acordado utilizar {value} como herramienta.")
    if w.family=="schedule_plan":
        return (f"Hemos cambiado la revisión del plan {w.subject}: a partir de ahora la realizaremos {value}, anulando el horario antiguo."
                if updated else f"La revisión del plan {w.subject} se hará {value}, conforme al acuerdo inicial.")
    raise ValueError(w.family)

def histories(w:BaseWorld,reverse:bool):
    old=w.options[1 if reverse else 0]
    new=w.options[0 if reverse else 1]
    return ((fact(w,old,False),*w.distractors_before),
            (fact(w,new,True),*w.distractors_after))

def question(w:BaseWorld)->str:
    a,b=w.options; x,y=w.actions
    return (f"¿Qué opción vigente acordamos para {FACET[w.family]} del proyecto {w.subject}? "
            f"Si corresponde «{a}», responde solo {x}; si corresponde «{b}», responde solo {y}. "
            "La última actualización sustituye la anterior. Si el dato no consta, responde ABSTENER.")

def expected(w:BaseWorld,value:str)->str:
    return w.actions[w.options.index(value)]

class InterventionArm(ExperimentArm):
    """Production pipeline composition with the 2 experimental injections."""
    def __init__(self,client,embeddings,tempdir:Path,adapter_factory=None):
        self.name="e10d_intervention"
        self.path=tempdir/"memory.sqlite"
        self.embeddings=embeddings
        self.adapter=adapter_factory() if adapter_factory else OpenAIResponsesAdapter(
            client,model=MODEL,reasoning_effort=REASONING_EFFORT)
        self._create_engine()
        self.loop=CognitiveLoop(self.engine,self.adapter,
                                developer_instructions=_RESPONSE_INSTRUCTIONS,
                                history_limit=0)

    def _create_engine(self):
        store=SQLiteCognitiveMemoryStore(path=str(self.path),
                                         embedding_provider=self.embeddings)
        goals=SimpleGoalEngine()
        builder=FixedPlanIntentBuilder(memory_store=store,goal_engine=goals)
        self.engine=CognitiveEngine(memory_store=store,goal_engine=goals,
                                    memory_gate=BroadDeclarativeGate(),
                                    context_builder=builder)
    def restart(self):
        self.loop.reset_history()
        self.engine.memory_store.close()
        self._create_engine()
        self.loop=CognitiveLoop(self.engine,self.adapter,
                                developer_instructions=_RESPONSE_INSTRUCTIONS,
                                history_limit=0)
    def close(self):
        self.engine.memory_store.close()


def create_arm(name,client,embeddings,p,adapter_factory):
    return (InterventionArm(client,embeddings,p,adapter_factory)
            if name=="e10d_intervention" else
            ExperimentArm(name,client,embeddings,p,adapter_factory))

def evaluate_world(w,client,embeddings,adapter_factory=None):
    prompt=question(w); variants={}
    for inverted in (False,True):
        before,after=histories(w,inverted)
        label="reversed" if inverted else "original"
        with tempfile.TemporaryDirectory(prefix="e10d-") as tmp:
            arms={name:create_arm(name,client,embeddings,Path(tmp)/name,
                                  adapter_factory) for name in ARMS}
            try:
                for line in before:
                    for arm in arms.values():arm.ingest(line)
                first={a:arm.decide(prompt) for a,arm in arms.items()}
                stored_first={a:len(arm.engine.memory_store.all()) for a,arm in arms.items()}
                for arm in arms.values():arm.restart()
                for line in after:
                    for arm in arms.values():arm.ingest(line)
                second={a:arm.decide(prompt) for a,arm in arms.items()}
                stored_second={a:len(arm.engine.memory_store.all()) for a,arm in arms.items()}
                variants[label]={
                    "expected_first":expected(w,w.options[1 if inverted else 0]),
                    "expected_second":expected(w,w.options[0 if inverted else 1]),
                    "first":first,"second":second,
                    "stored_first":stored_first,"stored_second":stored_second,
                }
            finally:
                for arm in arms.values():arm.close()
    orig=variants["original"];opp=variants["reversed"];stats={}
    for a in ARMS:
        outcomes=[int(v[t][a]["action"]==v["expected_"+t])
                 for v in (orig,opp) for t in ("first","second")]
        flips=all(orig[t][a]["action"]!=opp[t][a]["action"]
                  and orig[t][a]["action"] in ACTIONS
                  and opp[t][a]["action"] in ACTIONS for t in ("first","second"))
        stats[a]={
            "four_correct":int(all(outcomes)),
            "individual_correct":sum(outcomes),
            "first_correct":outcomes[0]+outcomes[2],
            "second_correct":outcomes[1]+outcomes[3],
            "both_flip":int(flips),
            "avg_context_tokens":statistics.mean(v[t][a]["estimated_context_tokens"]
                             for v in (orig,opp) for t in ("first","second")),
            "avg_stored_after":statistics.mean(v["stored_second"][a] for v in (orig,opp)),
            "provider_input_tokens":sum(int(v[t][a]["provider_tokens"].get("input_tokens",0))
                             for v in (orig,opp) for t in ("first","second")),
            "provider_output_tokens":sum(int(v[t][a]["provider_tokens"].get("output_tokens",0))
                             for v in (orig,opp) for t in ("first","second")),
        }
    return {"seed":w.seed,"family":w.family,"subject":w.subject,"question":prompt,
            "options":list(w.options),"actions":list(w.actions),
            "variants":variants,"arms":stats}


def negative_controls(client,embeddings,adapter_factory=None):
    results=[]
    for j in range(6):
        w=world_for(seed_for("dev",0,j))
        with tempfile.TemporaryDirectory(prefix="e10d-nodata-") as tmp:
            arms={a:create_arm(a,client,embeddings,Path(tmp)/a,adapter_factory)
                  for a in ARMS}
            try:
                outcomes={a:arm.decide(question(w)) for a,arm in arms.items()}
                results.append({"family":w.family,
                                "actions":{a:{"action":v["action"],
                                             "abstained":int(v["action"]=="ABSTENER"),
                                             "raw":v["raw"]} for a,v in outcomes.items()}})
            finally:
                for arm in arms.values():arm.close()
    return results

def summary(rows,negatives):
    if not rows:raise ValueError("No cases")
    out={"world_pairs":len(rows),"by_arm":{},"by_family":{},"by_index":{}}
    for a in ARMS:
        data=[r["arms"][a] for r in rows]
        out["by_arm"][a]={
            "four_correct":sum(x["four_correct"] for x in data),
            "four_rate":statistics.mean(x["four_correct"] for x in data),
            "individual_correct":sum(x["individual_correct"] for x in data),
            "first_correct":sum(x["first_correct"] for x in data),
            "second_correct":sum(x["second_correct"] for x in data),
            "both_flip":sum(x["both_flip"] for x in data),
            "mean_context_tokens":statistics.mean(x["avg_context_tokens"] for x in data),
            "mean_stored_after":statistics.mean(x["avg_stored_after"] for x in data),
            "input_tokens":sum(x["provider_input_tokens"] for x in data),
            "output_tokens":sum(x["provider_output_tokens"] for x in data),
            "negatives_abstained":sum(x["actions"][a]["abstained"] for x in negatives),
        }
    for family in FAMILIES:
        d=[r for r in rows if r["family"]==family]
        out["by_family"][family]={a:statistics.mean(r["arms"][a]["four_correct"] for r in d)
                                  for a in ARMS}
    for index in sorted(set(r["index"] for r in rows)):
        d=[r for r in rows if r["index"]==index]
        out["by_index"][str(index)]={a:statistics.mean(r["arms"][a]["four_correct"] for r in d)
                                    for a in ARMS}
    return out

def primary_criterion(stats):
    differences=[stats["by_family"][f]["e10d_intervention"]-
                 stats["by_family"][f]["semantic_rag"] for f in FAMILIES]
    rng=random.Random(1781)
    vals=sorted(statistics.mean(rng.choices(differences,k=6)) for _ in range(2000))
    abst_i=stats["by_arm"]["e10d_intervention"]["negatives_abstained"]
    abst_r=stats["by_arm"]["semantic_rag"]["negatives_abstained"]
    return {"difference_by_family":dict(zip(FAMILIES,differences)),
            "mean_difference":statistics.mean(differences),
            "positive_families":sum(x>0 for x in differences),
            "bootstrap_95_family_cluster":[vals[50],vals[1950]],
            "passes":statistics.mean(differences)>=.10 and
                     sum(x>0 for x in differences)>=4 and vals[50]>0 and abst_i>=abst_r,
            "confirm_consumed":True}

def run(split,output:Path,client,adapter_factory=None):
    if split not in ("dev","confirm"):raise ValueError(split)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh output directory required")
    output.mkdir(parents=True,exist_ok=True)
    emb=CachedSemanticEmbeddings(client)
    rows=[]
    for i in (range(1) if split=="dev" else range(4)):
        for j in range(6):
            w=world_for(seed_for(split,i,j))
            row=evaluate_world(w,client,emb,adapter_factory)
            row["index"]=i
            rows.append(row)
            with (output/"progress.jsonl").open("a",encoding="utf8") as fp:
                fp.write(json.dumps(row,ensure_ascii=False)+"\n")
            print("E10D_PAIR_FINISHED",split,i,j,flush=True)
    neg=negative_controls(client,emb,adapter_factory)
    stats=summary(rows,neg)
    result={
        "experiment":"INFINITO E10-D one preregistered intervention, REAL LLM",
        "split":split,"model":MODEL,"reasoning_effort":REASONING_EFFORT,
        "embedding_model":EMBEDDING_MODEL,"semantic_embedding_api_calls":emb.calls,
        "real_llm_expected":adapter_factory is None,"source_sha":os.getenv("GITHUB_SHA","local"),
        "protocol_hash":hashlib.sha256(Path("docs/INFINITO_E10D_SINGLE_INTERVENTION_PROTOCOL.md").read_bytes()).hexdigest(),
        "script_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "core_modified":False,"historical_splits_reused":False,
        "pairs":len(rows),"summary":stats,
        "criterion":primary_criterion(stats) if split=="confirm" else None,
        "negative_controls":neg,"cases":rows,
    }
    (output/"report.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf8")
    print("E10D_RESULT",split,json.dumps({"arms":stats["by_arm"],"criterion":result["criterion"],
                 "families":stats["by_family"],"embedding_calls":emb.calls},ensure_ascii=False),flush=True)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--split",choices=("dev","confirm"),required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("E10-D REAL API key missing; no offline fallback")
    if args.split=="confirm" and os.getenv("E10D_DEV_GATE_PASSED")!="1":
        raise SystemExit("E10-D confirmation requires successful live DEV gate on same SHA")
    from openai import OpenAI
    client=OpenAI(api_key=os.environ["OPENAI_API_KEY"],timeout=60,max_retries=2)
    run(args.split,args.output,client)
