"""E11-A: learned FUTURE RELEVANCE for a capacity-bounded symbolic memory.

Not F4: IDs are new in every episode; model must transfer relevance features,
not memorize demand per-key. Retrospective feedback on actually queried key is
shown ONLY after the bounded-memory decisions have finished. No LLM, no
semantic understanding, no query oracle inside selector.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

MEMORY_CAPACITY=4
EVENT_COUNT=16
REPLICAS=5
TRAIN_PER_REP=2500
DEV_PER_REP=160
CONFIRM_PER_REP=640
TRAIN_BASE=3100000
DEV_BASE=3200000
CONFIRM_BASE=3300000
ORACLE_COEFFICIENTS=(0.70,0.80,1.20,0.20,-0.30,0.35,1.60)
FEATURE_NAMES=("urgent","pending","previously_queried",
               "customer_source","automated_source","recency",
               "pending_x_previously_queried")
SCHEMES=("predictive","uniform","reversed")
PRACTICAL=("fifo","reservoir","fixed_priority",
           "recency_priority","learned_shuffled")
ALL_POLICIES=(*PRACTICAL,"learned","distribution_oracle")
SEED_BOOTSTRAP=1711


@dataclass(frozen=True)
class Event:
    key:str
    payload:int
    urgent:int
    pending:int
    previously_queried:int
    source:int  # 0 generic, 1 customer, 2 automatic
    position:int

    def features(self)->tuple[float,...]:
        return (float(self.urgent),float(self.pending),
                float(self.previously_queried),float(self.source==1),
                float(self.source==2),self.position/(EVENT_COUNT-1),
                float(self.pending*self.previously_queried))


@dataclass(frozen=True)
class Episode:
    events:tuple[Event,...]
    # Held by harness. NEVER given to a policy during observations.
    future_query_key:str
    regime:str
    seed:int


def seed_for(phase:str,replica:int,episode:int)->int:
    bounds={"train":(TRAIN_BASE,TRAIN_PER_REP),
            "dev":(DEV_BASE,DEV_PER_REP),
            "confirm":(CONFIRM_BASE,CONFIRM_PER_REP)}
    if phase not in bounds or not 0<=replica<REPLICAS:
        raise ValueError("Unknown split or replica")
    base,count=bounds[phase]
    if not 0<=episode<count:
        raise ValueError("Out-of-range heldout episode")
    return base+10000*replica+episode


def dot(a:Sequence[float],b:Sequence[float])->float:
    return sum(x*y for x,y in zip(a,b))


def softmax(scores:Sequence[float])->list[float]:
    m=max(scores)
    ex=[math.exp(v-m) for v in scores]
    denom=sum(ex)
    return [v/denom for v in ex]


def episode_for(phase:str,replica:int,i:int,regime:str)->Episode:
    if regime not in SCHEMES:
        raise ValueError("Invalid regime")
    seed=seed_for(phase,replica,i)
    rng=random.Random(seed)
    events=[]
    for j in range(EVENT_COUNT):
        # All features sampled INDEPENDENT of position and key, except recency.
        urgent=rng.randrange(2)
        pending=rng.randrange(2)
        previously=rng.randrange(2)
        source=rng.randrange(3)
        payload=rng.randrange(2**32)
        # Split-prefixed identities prevent transfer by ID memorization.
        events.append(Event(f"{phase}:r{replica}:s{seed}:record{j}",
                    payload,urgent,pending,previously,source,j))
    coeff=(ORACLE_COEFFICIENTS if regime=="predictive"
           else tuple(-w for w in ORACLE_COEFFICIENTS))
    if regime=="uniform":
        target=rng.randrange(EVENT_COUNT)
    else:
        probabilities=softmax([dot(coeff,e.features()) for e in events])
        value=rng.random()
        acc=0.0
        target=EVENT_COUNT-1
        for j,p in enumerate(probabilities):
            acc+=p
            if value<=acc:
                target=j
                break
    return Episode(tuple(events),events[target].key,regime,seed)


class BoundedMemory:
    """Only 4 actual Event objects. No hidden cache or evicted-key index."""
    def __init__(self):
        self.saved:dict[str,Event]={}
        self.order:list[str]=[]
        self.maximum_seen=0

    def put(self,event:Event,evict_key:str|None=None)->None:
        if event.key in self.saved:
            self.saved[event.key]=event
            return
        if len(self.saved)>=MEMORY_CAPACITY:
            if evict_key is None or evict_key not in self.saved:
                raise ValueError("Memory full: explicit key to evict required")
            del self.saved[evict_key]
            self.order.remove(evict_key)
        self.saved[event.key]=event
        self.order.append(event.key)
        self.maximum_seen=max(self.maximum_seen,len(self.saved))
        assert len(self.saved)<=MEMORY_CAPACITY

    def recall(self,key:str)->int|None:
        e=self.saved.get(key)
        return e.payload if e else None


class BaseSelector:
    def __init__(self,seed:int):
        self.rng=random.Random(seed)
    def begin(self)->BoundedMemory:
        return BoundedMemory()
    def consider(self,memory:BoundedMemory,event:Event)->None:
        raise NotImplementedError

class FIFO(BaseSelector):
    def consider(self,memory,event):
        memory.put(event,memory.order[0] if len(memory.saved)==MEMORY_CAPACITY else None)

class Reservoir(BaseSelector):
    def begin(self):
        self.count=0
        return super().begin()
    def consider(self,memory,event):
        self.count+=1
        if len(memory.saved)<MEMORY_CAPACITY:
            memory.put(event)
        elif self.rng.randrange(self.count)<MEMORY_CAPACITY:
            memory.put(event,self.rng.choice(memory.order))

class RankSelector(BaseSelector):
    """Streaming top-4 by a score computed from only each candidate record.

    Observe the current event and bounded memory contents; never the target.
    """
    def __init__(self,seed:int,weights:Sequence[float]):
        super().__init__(seed)
        if len(weights)!=len(FEATURE_NAMES):
            raise ValueError("Invalid learned feature vector")
        self.weights=tuple(float(x) for x in weights)

    def score(self,e:Event)->float:
        return dot(self.weights,e.features())

    def consider(self,memory,event):
        if len(memory.saved)<MEMORY_CAPACITY:
            memory.put(event)
            return
        weakest=min(memory.order,key=lambda k:self.score(memory.saved[k]))
        if self.score(event)>self.score(memory.saved[weakest]):
            memory.put(event,weakest)

class FeedbackLearner:
    """Log-linear prospective demand predictor trained from DELAYED query feedback.

    Trainer has a full *historical* episode transcript AFTER the memory writes
    finish; the live selection policy does NOT. Never uses oracle coefficients.
    """
    def __init__(self):
        self.weights=[0.0]*len(FEATURE_NAMES)
        self.examples=0
        self.logloss_total=0.0

    def feedback(self,history:tuple[Event,...],future_query_key:str):
        assert self.examples>=0
        index=next(j for j,e in enumerate(history) if e.key==future_query_key)
        xs=[e.features() for e in history]
        logits=[dot(self.weights,x) for x in xs]
        probabilities=softmax(logits)
        self.logloss_total-=math.log(max(1e-15,probabilities[index]))
        expected=[sum(p*x[d] for p,x in zip(probabilities,xs))
                  for d in range(len(FEATURE_NAMES))]
        # Online feedback from past queries only, step fixed before any holdout.
        lr=0.08/math.sqrt(1.0+self.examples/1700.0)
        for d in range(len(FEATURE_NAMES)):
            gradient=xs[index][d]-expected[d]-0.0008*self.weights[d]
            self.weights[d]+=lr*gradient
        self.examples+=1

def fixed_priority()->tuple[float,...]:
    # Simple, strong and transparent generic heuristic. NOT tuned after test.
    return (0.9,0.9,0.9,0.1,0.0,0.18,0.0)

def recency_priority()->tuple[float,...]:
    return (0.35,0.35,0.35,0.05,0.0,2.0,0.0)

def training_models()->dict:
    out={}
    for replica in range(REPLICAS):
        actual=FeedbackLearner()
        shuffled=FeedbackLearner()
        for i in range(TRAIN_PER_REP):
            episode=episode_for("train",replica,i,"predictive")
            # Hindsight feedback shown AFTER the hypothetical memory has been
            # selected. This model never sees future_query_key in consider().
            actual.feedback(episode.events,episode.future_query_key)
            bogus_index=random.Random(7400000+replica*10000+i).randrange(EVENT_COUNT)
            shuffled.feedback(episode.events,episode.events[bogus_index].key)
        out[str(replica)]={
            "learned_weights":actual.weights,
            "shuffled_weights":shuffled.weights,
            "train_examples":actual.examples,
            "train_cross_entropy":actual.logloss_total/actual.examples,
            "shuffled_cross_entropy":shuffled.logloss_total/shuffled.examples,
        }
    return out

def policy_for(name:str,seed:int,model:dict,regime:str):
    if name=="fifo":return FIFO(seed+9)
    if name=="reservoir":return Reservoir(seed+9)
    if name=="fixed_priority":return RankSelector(seed+9,fixed_priority())
    if name=="recency_priority":return RankSelector(seed+9,recency_priority())
    if name=="learned_shuffled":return RankSelector(seed+9,model["shuffled_weights"])
    if name=="learned":return RankSelector(seed+9,model["learned_weights"])
    if name=="distribution_oracle":
        return RankSelector(seed+9,ORACLE_COEFFICIENTS if regime!="reversed"
                            else tuple(-x for x in ORACLE_COEFFICIENTS))
    raise ValueError(name)

def run_trial(episode:Episode,model:dict)->dict:
    out={}
    for name in ALL_POLICIES:
        selector=policy_for(name,episode.seed,model,episode.regime)
        memory=selector.begin()
        # NEVER pass the future query key into begin() or consider().
        for e in episode.events:
            selector.consider(memory,e)
        assert len(memory.saved)==MEMORY_CAPACITY and memory.maximum_seen==MEMORY_CAPACITY
        recalled=memory.recall(episode.future_query_key)
        actual=next(e.payload for e in episode.events if e.key==episode.future_query_key)
        out[name]={
            "hit":int(recalled is not None),
            "correct":int(recalled==actual),
            "saved_count":len(memory.saved),
            "memory_capacity_violations":int(memory.maximum_seen>MEMORY_CAPACITY),
        }
    return {"seed":episode.seed,"regime":episode.regime,"policies":out}

def summarize(rows:list[dict])->dict:
    if not rows:raise ValueError("Empty evaluation")
    indexes=sorted({r["replica"] for r in rows})
    result={"episodes":len(rows),"replicas":indexes,"policies":{},"by_replica":{}}
    for p in ALL_POLICIES:
        hit=sum(r["policies"][p]["hit"] for r in rows)
        result["policies"][p]={"hit":hit,"rate":hit/len(rows)}
    for i in indexes:
        sub=[r for r in rows if r["replica"]==i]
        result["by_replica"][str(i)]={
            p:sum(r["policies"][p]["hit"] for r in sub)/len(sub)
            for p in ALL_POLICIES}
    return result

def bootstrap(summary:dict):
    rng=random.Random(SEED_BOOTSTRAP)
    out={}
    for baseline in PRACTICAL:
        diffs=[summary["by_replica"][str(i)]["learned"]-
               summary["by_replica"][str(i)][baseline] for i in summary["replicas"]]
        vals=sorted(statistics.mean(rng.choices(diffs,k=len(diffs)))
                    for _ in range(10000))
        out[baseline]={
            "mean_difference":statistics.mean(diffs),
            "by_replica":diffs,
            "positive_replicas":sum(x>0 for x in diffs),
            "bootstrap_95":[vals[250],vals[9750]],
        }
    strongest=max(PRACTICAL,key=lambda p:summary["policies"][p]["rate"])
    strongest_diff=out[strongest]
    out["strongest_practical_baseline"]=strongest
    out["preregistered_gate"]=(
        strongest_diff["mean_difference"]>=0.05
        and strongest_diff["positive_replicas"]>=4
        and strongest_diff["bootstrap_95"][0]>0
        and all(v["mean_difference"]>0 for p,v in out.items() if p in PRACTICAL)
    )
    return out

def eval_cohort(phase:str,models:dict,regime:str):
    if phase not in ("dev","confirm"):
        raise ValueError("Only validation cohorts allowed")
    total=DEV_PER_REP if phase=="dev" else CONFIRM_PER_REP
    rows=[]
    for rep in range(REPLICAS):
        for j in range(total):
            ep=episode_for(phase,rep,j,regime)
            row=run_trial(ep,models[str(rep)])
            row["replica"]=rep
            rows.append(row)
    return rows,summarize(rows)

def train_and_dev(output:Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh output required")
    output.mkdir(parents=True,exist_ok=True)
    models=training_models()
    model_data={
        "experiment":"E11-A training only",
        "training_split":"3100000+replica*10000+index",
        "train_episodes_per_replica":TRAIN_PER_REP,
        "replicas":REPLICAS,
        "feedback":"observed future query AFTER memory selection, no prospective target leakage",
        "feature_names":FEATURE_NAMES,
        "weights":models,
        "script_hash":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_hash":hashlib.sha256(Path("docs/INFINITO_E11A_LEARNED_RELEVANCE_PROTOCOL.md").read_bytes()).hexdigest(),
        "git_sha":os.environ.get("GITHUB_SHA","unknown"),
    }
    (output/"frozen_models.json").write_text(json.dumps(model_data,indent=2),encoding="utf8")
    rows,summary=eval_cohort("dev",models,"predictive")
    report={"split":"dev","summary":summary,"rows":rows}
    (output/"dev_report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("E11A_TRAIN_DEV_RESULT",json.dumps({
        "weights":models,"dev_hits":summary["policies"]},ensure_ascii=False),flush=True)
    return report

def confirm(output:Path,model_path:Path):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Fresh confirmation output required")
    model_data=json.loads(model_path.read_text(encoding="utf8"))
    if model_data["script_hash"]!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise RuntimeError("Learner implementation changed after weights were frozen")
    if model_data["protocol_hash"]!=hashlib.sha256(Path("docs/INFINITO_E11A_LEARNED_RELEVANCE_PROTOCOL.md").read_bytes()).hexdigest():
        raise RuntimeError("E11 protocol changed after weights were frozen")
    if model_data["replicas"]!=REPLICAS or set(model_data["weights"])!=set(str(i) for i in range(REPLICAS)):
        raise ValueError("Missing independent learned replicas")
    output.mkdir(parents=True,exist_ok=True)
    print("E11A_CONFIRM_3300000_OPENED_AND_CONSUMED",flush=True)
    results={}
    for regime in SCHEMES:
        rows,stats=eval_cohort("confirm",model_data["weights"],regime)
        results[regime]={"summary":stats,"paired_rows":rows,
                         "contrasts":bootstrap(stats) if regime=="predictive" else None}
        (output/f"{regime}_paired_worlds.json").write_text(json.dumps(rows),encoding="utf8")
        print("E11A_CONFIRM_REGIME",regime,json.dumps(stats["policies"]),flush=True)
    negative=results["uniform"]["summary"]
    p=negative["policies"]
    null_gap=p["learned"]["rate"]-p["reservoir"]["rate"]
    gate=results["predictive"]["contrasts"]["preregistered_gate"] and null_gap<=0.05
    report={
        "experiment":"E11-A learned future relevance / F4 extension to unseen keys",
        "split":"confirm",
        "model_source_sha":model_data["git_sha"],
        "script_hash":model_data["script_hash"],
        "protocol_hash":model_data["protocol_hash"],
        "memory_capacity":MEMORY_CAPACITY,
        "events_per_episode":EVENT_COUNT,
        "confirm_episodes_per_regime":REPLICAS*CONFIRM_PER_REP,
        "independent_training_replicas":REPLICAS,
        "oracle_demand_distribution_is_not_given_to_learner":True,
        "learned_identity_transfer":True,
        "no_llm":True,
        "no_infinito3_core_modified":True,
        "no_e10_splits_reused":True,
        "uniform_expected_capacity_rate":MEMORY_CAPACITY/EVENT_COUNT,
        "uniform_learned_minus_reservoir":null_gap,
        "results":{name:{"summary":v["summary"],"contrasts":v["contrasts"]}
                   for name,v in results.items()},
        "progress_gate":gate,
    }
    (output/"report.json").write_text(json.dumps(report,indent=2),encoding="utf8")
    print("E11A_FINAL_RESULT",json.dumps({
        "predictive":report["results"]["predictive"],
        "uniform":report["results"]["uniform"]["summary"]["policies"],
        "reversed":report["results"]["reversed"]["summary"]["policies"],
        "progress_gate":gate},ensure_ascii=False),flush=True)
    return report

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase",choices=("train-dev","confirm"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--models",type=Path)
    args=p.parse_args()
    if args.phase=="train-dev":
        train_and_dev(args.output)
    else:
        if not args.models:
            raise SystemExit("Confirmation requires frozen models from previous job")
        confirm(args.output,args.models)
if __name__=="__main__":
    main()
