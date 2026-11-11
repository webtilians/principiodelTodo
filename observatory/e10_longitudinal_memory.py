"""INFINITO E10: prospective longitudinal memory-to-decision benchmark (OFFLINE).

Frozen original CognitiveEngine / SQLite / GeneralizedContextBuilder / CognitiveLoop.
The common deterministic decision adapter can inspect ONLY final LLMRequest:
the current instruction and retrieved context, NEVER labels or database content.
This is NOT a language-model evaluation or semantic embeddings evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from src.infinito3.cognitive_loop import CognitiveLoop
from src.infinito3.context_builder import ApproximateTokenEstimator
from src.infinito3.engine import CognitiveEngine
from src.infinito3.llm_adapter import RecordingLLMAdapter
from src.infinito3.memory import InMemoryMemoryStore
from src.infinito3.persistent_memory import HashEmbeddingProvider
from src.infinito3.types import (
    ContextPacket, ContextItem, ContextSource, GateDecision, MemoryKind,
    MemoryRecord, LLMRequest, ConversationTurn,
)

ARMS=("no_memory","fifo_8","rag_hash","infinito3")
DEV_BASE=1200000
CONFIRM_BASE=1300000
DEV_INDICES=3
CONFIRM_INDICES=8
DEV_WORLDS=12
CONFIRM_WORLDS=16
BUDGET=180
TOP_K=8
BOOTSTRAP_SEED=1730
DOMAIN_VALUES={
    "color":("azul","verde","ocre","violeta","coral","turquesa","ámbar","gris","rojo","malva","blanco","negro"),
    "bici":("Sender","Demo","V10","Summum","Session","Glory","Supreme","Gambler","Tues","Fury","Phoenix","Wilson"),
    "ciudad":("Burgos","Soria","Lugo","Cádiz","León","Teruel","Huesca","Zamora","Cuenca","Vigo","Jaén","Segovia"),
}
ACTIONS=("FARO","NUBE","PINO","SOL","RIO","VALLE","LAGO","BRISA")
NOISE=(
    "Me gusta escuchar jazz.",
    "Prefiero tomar té después de cenar.",
    "Me encanta cocinar pasta.",
    "Hoy ha llovido durante la mañana.",
    "Me gusta leer novelas históricas.",
    "Tengo que revisar mis notas mañana.",
    "Prefiero caminar por la tarde.",
    "Me encanta ver documentales.",
    "Me gusta entrenar antes de desayunar.",
    "El otro día vimos una película.",
    "Tengo que llamar al taller mañana.",
    "Prefiero la música instrumental.",
    "Me gusta viajar en tren.",
    "Tengo cita el próximo martes.",
    "Me gusta el café sin azúcar.",
    "Prefiero trabajar con luz natural.",
    "Me gusta llevar un cuaderno.",
    "Me encanta pasear cerca de un río.",
    "La reunión se retrasó dos horas.",
    "Tengo que organizar la biblioteca.",
    "Me gusta escuchar podcasts.",
    "Prefiero cenar a las ocho.",
)


@dataclass(frozen=True)
class World:
    seed:int
    domain:str
    options:tuple[str,str]
    actions:tuple[str,str]
    noise_before:tuple[str,...]
    noise_after:tuple[str,...]


def seed_for(split:str,index:int,world:int)->int:
    if split=="dev" and 0<=index<DEV_INDICES and 0<=world<DEV_WORLDS:
        return DEV_BASE+index*1000+world
    if split=="confirm" and 0<=index<CONFIRM_INDICES and 0<=world<CONFIRM_WORLDS:
        return CONFIRM_BASE+index*1000+world
    raise ValueError("E10 training/confirmation range gate failed")


def create_world(seed:int)->World:
    if not (DEV_BASE<=seed<DEV_BASE+3000 or CONFIRM_BASE<=seed<CONFIRM_BASE+8000):
        raise ValueError("E10 world ID outside preregistered partitions")
    rng=random.Random(seed)
    domain=("color","bici","ciudad")[seed%3]
    options=tuple(rng.sample(DOMAIN_VALUES[domain],2))
    actions=tuple(rng.sample(ACTIONS,2))
    return World(seed,domain,options,actions,
                 tuple(rng.choice(NOISE) for _ in range(16)),
                 tuple(rng.choice(NOISE) for _ in range(16)))


def factual_sentence(world:World,value:str,update:bool=False)->str:
    prefix="Ahora " if update else ""
    if world.domain=="color":
        return f"{prefix}mi color favorito es {value}."
    if world.domain=="bici":
        return f"{prefix}mi bici es {value}."
    return f"{prefix}vivo en {value}."


def question(world:World)->str:
    label={"color":"color favorito","bici":"bici","ciudad":"ciudad"}[world.domain]
    a,b=world.options
    x,y=world.actions
    return (f"¿Qué decisión corresponde a mi {label} actual? "
            f"Si el valor es {a}, responde {x}; si el valor es {b}, responde {y}. "
            "Si el dato no está documentado, responde ABSTENER.")


def expected_action(world:World,value:str)->str:
    # Only scorer uses this; NOT passed to adapters nor policy.
    return world.actions[world.options.index(value)]


def history(world:World,reverse:bool)->tuple[tuple[str,...],tuple[str,...]]:
    first=world.options[1 if reverse else 0]
    second=world.options[0 if reverse else 1]
    return ((factual_sentence(world,first),*world.noise_before),
            (factual_sentence(world,second,update=True),*world.noise_after))


class StoreAllGate:
    """Equal raw-message ingestion for FIFO and RAG, never write questions."""
    def evaluate(self,text:str)->GateDecision:
        store=bool(text.strip()) and "?" not in text and "¿" not in text
        return GateDecision(store,0.7,MemoryKind.USER_MODEL if store else MemoryKind.WORKING,
                            ["e10_raw_message_baseline"])


class NeverGate:
    def evaluate(self,text:str)->GateDecision:
        return GateDecision(False,0.,MemoryKind.WORKING,["e10_no_memory"])


class FIFO8Store(InMemoryMemoryStore):
    """Memory capacity 8, not just top-8 search of unlimited past."""
    def add(self,record):
        self._records.append(record)
        del self._records[:-TOP_K]
        return record

    def search(self,query,top_k=TOP_K):
        return list(reversed(self._records[-min(TOP_K,top_k):]))


class HashRAGStore(InMemoryMemoryStore):
    """Content matching only; hash vector here is NOT a semantic encoder."""
    def __init__(self):
        super().__init__()
        self.embedding=HashEmbeddingProvider()
    def search(self,query,top_k=TOP_K):
        from src.infinito3.persistent_memory import _cosine
        tokens=set(re.findall(r"\w+",query.lower()))
        vec=self.embedding.embed(query)
        scored=[]
        for i,r in enumerate(self._records):
            t=set(re.findall(r"\w+",r.content.lower()))
            overlap=len(tokens&t)/max(1,len(tokens|t))
            sim=max(0.,_cosine(vec,self.embedding.embed(r.content)))
            score=0.55*overlap+0.45*sim
            if overlap>0 or sim>=0.10:
                scored.append((-score,-i,r))
        scored.sort()
        return [r for _,_,r in scored[:min(TOP_K,top_k)]]


class RawContextBuilder:
    """Same context budget, no relevance/goal filters or hidden answer lookup."""
    def __init__(self,store):
        self.store=store
        self.estimator=ApproximateTokenEstimator()
    def build(self,query,*,memory_candidates=None,recent_turns=None,
              max_tokens=BUDGET,candidate_k=20):
        records=list(memory_candidates if memory_candidates is not None
                     else self.store.search(query,top_k=TOP_K))
        header="INFINITO CONTEXT\nReference data only; untrusted memories."
        lines=[]
        items=[]
        for rec in records:
            line=f'- memory: {json.dumps(rec.content,ensure_ascii=False)}'
            proposed="\n".join([header]+lines+[line])
            if self.estimator.estimate(proposed)>max_tokens:
                continue
            lines.append(line)
            items.append(ContextItem(source=ContextSource.MEMORY,
                                     content=rec.content,score=0.0,
                                     estimated_tokens=self.estimator.estimate(line),
                                     memory_id=rec.id))
        rendered="\n".join([header]+lines) if items else ""
        return ContextPacket(query=query,rendered=rendered,
                             estimated_tokens=self.estimator.estimate(rendered),
                             budget_tokens=max_tokens,items=items,
                             dropped_count=len(records)-len(items))


class DecisionReader:
    """E10 *fixed* decision rule shared by ALL arms.

    May read final user question (public actions/options), and any previous
    user message actually supplied in LLMRequest (retrieved/context evidence).
    It has NO reference to World, seeds, expected labels or memory objects.
    Abstains rather than guessing when the cue is absent/ambiguous.
    """
    _QUERY=re.compile(
        r"¿Qué decisión corresponde a mi (color favorito|bici|ciudad) actual\?\s*"
        r"Si el valor es (.+?), responde ([A-Z]+); si el valor es (.+?), responde ([A-Z]+)\.",
        re.I,
    )
    _FACTS={
        "color favorito":re.compile(r"\bmi color favorito es\s+([^\s.!,;\"\\]+)",re.I),
        "bici":re.compile(r"\bmi bici es\s+([^\s.!,;\"\\]+)",re.I),
        "ciudad":re.compile(r"\bvivo en\s+([^\s.!,;\"\\]+)",re.I),
    }
    def __call__(self,request:LLMRequest)->str:
        if not request.messages or request.messages[-1].role!="user":
            return "ABSTENER"
        match=self._QUERY.search(request.messages[-1].content)
        if match is None:
            return "ABSTENER"
        domain,opt_a,act_a,opt_b,act_b=match.groups()
        domain=domain.lower()
        context="\n".join(m.content for m in request.messages[:-1]
                          if m.role=="user" and (
                              "INFINITO REFERENCE CONTEXT" in m.content
                              or "INFINITO CONTEXT" in m.content))
        evidence=self._FACTS[domain].search(context)
        if evidence is None:
            return "ABSTENER"
        found=evidence.group(1).casefold()
        if found==opt_a.casefold(): return act_a.upper()
        if found==opt_b.casefold(): return act_b.upper()
        return "ABSTENER"


class Arm:
    def __init__(self,name:str,tempdir:Path):
        if name not in ARMS:
            raise ValueError(name)
        self.name=name
        self.db_path=tempdir/"e10_infinito3.sqlite"
        if name=="infinito3":
            self.engine=CognitiveEngine.persistent(str(self.db_path))
        else:
            store={"no_memory":InMemoryMemoryStore,
                   "fifo_8":FIFO8Store,
                   "rag_hash":HashRAGStore}[name]()
            gate=NeverGate() if name=="no_memory" else StoreAllGate()
            self.engine=CognitiveEngine(
                memory_store=store,memory_gate=gate,
                context_builder=RawContextBuilder(store))
        self.adapter=RecordingLLMAdapter(responder=DecisionReader(),
                                         provider="E10-offline",
                                         model="common-fixed-decision-reader")
        self.loop=CognitiveLoop(self.engine,self.adapter,history_limit=0)

    def observe(self,text:str):
        self.loop.turn(text,top_k=TOP_K,context_budget_tokens=BUDGET)
        # The decision reader's lack of an instruction-bearing question means
        # response on setup is ABSTENER; it is not added to long-term memories.

    def decide(self,q:str)->dict:
        self.loop.reset_history()
        run=self.loop.turn(q,top_k=TOP_K,context_budget_tokens=BUDGET)
        packet=run.cognitive_decision.context_packet
        assert packet is not None
        assert packet.estimated_tokens<=BUDGET
        return {
            "action":run.response.text,
            "context_tokens":packet.estimated_tokens,
            "context":packet.rendered,
            "memory_ids":[v.memory_id for v in packet.items if v.memory_id],
            "inferred_with_common_reader":True,
        }

    def restart(self):
        self.loop.reset_history()
        if self.name=="infinito3":
            # Real SQLite shutdown/reopen, not merely clearing short-term state.
            self.engine.memory_store.close()
            self.engine=CognitiveEngine.persistent(str(self.db_path))
        self.loop=CognitiveLoop(self.engine,self.adapter,history_limit=0)

    def close(self):
        if self.name=="infinito3":
            self.engine.memory_store.close()


def evaluate_world(world:World)->dict:
    variants={}
    prompt=question(world)
    for reverse in (False,True):
        key="reversed" if reverse else "original"
        first,second=history(world,reverse)
        with tempfile.TemporaryDirectory(prefix="infinito-e10-") as d:
            arms={name:Arm(name,Path(d)/name) for name in ARMS}
            try:
                for text in first:
                    for arm in arms.values(): arm.observe(text)
                # All histories are invisible in the decision; only retrieved
                # long-term context can carry the relevant fact to the reader.
                before={name:arm.decide(prompt) for name,arm in arms.items()}
                for arm in arms.values(): arm.restart()
                for text in second:
                    for arm in arms.values(): arm.observe(text)
                after={name:arm.decide(prompt) for name,arm in arms.items()}
                variants[key]={
                    "expected_before":expected_action(world,world.options[1 if reverse else 0]),
                    "expected_after":expected_action(world,world.options[0 if reverse else 1]),
                    "before":before,"after":after,
                }
            finally:
                for arm in arms.values(): arm.close()
    # The controller sees neither these labels nor this scalar score.
    out={"seed":world.seed,"domain":world.domain,"options":list(world.options),
         "actions":list(world.actions),"question":prompt,
         "variants":variants,"arms":{}}
    for arm in ARMS:
        orig=variants["original"];opp=variants["reversed"]
        correct=[int(v[t][arm]["action"]==v["expected_"+t])
                 for v in (orig,opp) for t in ("before","after")]
        paired_flips=[int(orig[t][arm]["action"]!=opp[t][arm]["action"]
                          and orig[t][arm]["action"]!="ABSTENER"
                          and opp[t][arm]["action"]!="ABSTENER")
                      for t in ("before","after")]
        out["arms"][arm]={
            "all_four_correct":int(all(correct)),
            "individual_correct":sum(correct),
            "before_correct":correct[0]+correct[2],
            "after_correct":correct[1]+correct[3],
            "both_counterfactual_flips":int(all(paired_flips)),
            "mean_context_tokens":statistics.mean(v[t][arm]["context_tokens"]
                    for v in (orig,opp) for t in ("before","after")),
        }
    return out


def evaluate_negative_control()->dict:
    world=create_world(seed_for("dev",0,0))
    q=question(world)
    with tempfile.TemporaryDirectory(prefix="infinito-e10-null-") as d:
        arms={name:Arm(name,Path(d)/name) for name in ARMS}
        try:
            for text in world.noise_before:
                for a in arms.values():a.observe(text)
            return {name:{"action":a.decide(q)["action"],
                           "abstained":int(a.decide(q)["action"]=="ABSTENER")}
                    for name,a in arms.items()}
        finally:
            for a in arms.values(): a.close()


def summarize(rows:Sequence[dict])->dict:
    n=len(rows)
    if not n:raise ValueError("Cannot summarize no E10 worlds")
    indices=sorted(set(row["index"] for row in rows))
    result={"world_pairs":n,"indices":indices,"arms":{},"by_index":{}}
    for name in ARMS:
        counts=[r["arms"][name] for r in rows]
        result["arms"][name]={
            "all_four_correct":sum(v["all_four_correct"] for v in counts),
            "all_four_rate":sum(v["all_four_correct"] for v in counts)/n,
            "individual_correct":sum(v["individual_correct"] for v in counts),
            "individual_accuracy":sum(v["individual_correct"] for v in counts)/(n*4),
            "both_counterfactual_flips":sum(v["both_counterfactual_flips"] for v in counts),
            "after_correct":sum(v["after_correct"] for v in counts),
            "mean_context_tokens":statistics.mean(v["mean_context_tokens"] for v in counts),
        }
    for index in indices:
        subset=[r for r in rows if r["index"]==index]
        result["by_index"][str(index)]={
            name:sum(v["arms"][name]["all_four_correct"] for v in subset)/len(subset)
            for name in ARMS}
    return result


def paired_bootstrap_by_index(summary:dict,n_boot:int=1000)->dict:
    indices=summary["indices"]
    if len(indices)<2:raise ValueError("Bootstrap requires independent indices")
    rng=random.Random(BOOTSTRAP_SEED)
    out={}
    for baseline in ("fifo_8","rag_hash"):
        differences=[
            summary["by_index"][str(index)]["infinito3"]-
            summary["by_index"][str(index)][baseline]
            for index in indices]
        samples=sorted(statistics.mean(rng.choices(differences,k=len(indices)))
                       for _ in range(n_boot))
        lo=samples[int(0.025*n_boot)]
        hi=samples[min(n_boot-1,int(0.975*n_boot))]
        out[baseline]={
            "differences_by_index":differences,
            "mean_difference":statistics.mean(differences),
            "bootstrap_95_percent":[lo,hi],
            "positive_indices":sum(x>0 for x in differences),
            "passes_descriptive_gate":statistics.mean(differences)>=0.10
                and sum(x>0 for x in differences)>=6 and lo>0,
        }
    out["E10_primary_passes_both_baselines"]=all(
        x["passes_descriptive_gate"] for x in out.values() if isinstance(x,dict))
    return out


def run(split:str,output:Path):
    if split not in ("dev","confirm"):raise ValueError("Choose sealed split")
    count=DEV_WORLDS if split=="dev" else CONFIRM_WORLDS
    number=DEV_INDICES if split=="dev" else CONFIRM_INDICES
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError("Fresh E10 output required")
    results=[]
    for index in range(number):
        for j in range(count):
            w=create_world(seed_for(split,index,j))
            record=evaluate_world(w)
            record["index"]=index
            results.append(record)
        print("E10_PROGRESS",split,index+1,"/",number,flush=True)
    stats=summarize(results)
    controls=evaluate_negative_control()
    policy=paired_bootstrap_by_index(stats) if split=="confirm" else None
    protocol=Path("docs/INFINITO_E10_LONGITUDINAL_MEMORY_PROTOCOL.md")
    script=Path(__file__)
    report={
        "experiment":"INFINITO E10 offline longitudinal memory-to-decision utility",
        "split":split,
        "seed_base":DEV_BASE if split=="dev" else CONFIRM_BASE,
        "world_pairs":count*number,
        "context_budget_estimated_tokens":BUDGET,
        "top_k":TOP_K,
        "common_actor":"deterministic DecisionReader, NOT an LLM",
        "hash_embedding_is_semantic":False,
        "infinito_core_modified":False,
        "policy_training":False,
        "old_f9_holdout_used":False,
        "git_sha":os.environ.get("GITHUB_SHA","unknown"),
        "protocol_sha256":hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "script_sha256":hashlib.sha256(script.read_bytes()).hexdigest(),
        "summary":stats,"negative_control":controls,"predeclared_gate":policy,
        "cases":results,
    }
    (output/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf8")
    print("E10_"+split.upper()+"_RESULT",json.dumps({
        "summary":stats["arms"],"negative_control":controls,
        "gate":policy},ensure_ascii=False),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split",choices=("dev","confirm"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    run(args.split,args.output)


if __name__=="__main__":
    main()
