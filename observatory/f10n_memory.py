"""F10-N: explicit experience memory, learned operation transitions, new-program synthesis.

NO operator transition is programmed into EffectMemory / planner. The world
executes real steps; the learner sees before/after work register, selection
flag, precondition failures, and READ content only AFTER READ executes.
A latent memory bit is never supplied before the real read. Planning runs
two *hypotheses* (F4=0,1) and scores predicted goal satisfaction on both.
"""
from __future__ import annotations
from collections import defaultdict,Counter
import json
import math
import numpy as np

from observatory.f10n_worlds import PROGRAMS,TRAIN_PROGRAMS

def execute(row,seq,slot,xor_lut,mode="normal"):
    """Environment owns operator semantics, labels and reward; not planner."""
    work=-1
    selected=0
    invalid=0
    trace=[]
    mask=np.flatnonzero(row["mask"]>.5)
    memory_contents=row["contents"]
    if mode=="shuffle":
        memory_contents=row["contents"].copy()
        memory_contents[mask]=row["contents"][mask[::-1]]
    for op in seq:
        prev=(selected,work)
        read_obs=-1
        delta=0
        if op=="P":
            b=int(row["public_bit"])
            if mode=="public_randomized" and row["task"]=="MEMORY":
                b=1-b
            work=b
        elif op=="S":
            selected=1
        elif op=="R":
            if not selected:
                delta=1
            else:
                read_obs=int(memory_contents[slot]>0)
                work=read_obs
        elif op=="X":
            if work==-1:
                delta=1
            elif mode!="x_disabled":
                work=int(xor_lut[work])
        else:
            raise AssertionError(op)
        invalid+=delta
        trace.append({
            "op":op,"before":list(prev),"after":[selected,work],
            "invalid_delta":delta,"read_observed":read_obs,
        })
    answer=0 if work==-1 else work
    success=int(answer==row["truth"])
    reward=(2*success-1)-0.01*len(seq)-0.10*invalid
    return {
        "success":success,"answer":answer,"reward":float(reward),
        "invalid":int(invalid),"operations":len(seq),
        "trace":trace,
    }

def transition_key(op,selected,work,public_bit,observed_read):
    """Input signature fixed prospectively; transition outcome NOT supplied."""
    return (str(op),int(selected),int(work),
            int(public_bit) if op=="P" else -1,
            int(observed_read) if op=="R" else -1)

def encode_key(key):
    return ",".join(map(str,key))

def encode_value(value):
    return ",".join(map(str,value))

class EffectMemory:
    """Sufficient statistics of ACTUALLY OBSERVED operation effects."""
    def __init__(self):
        self.counts=defaultdict(Counter)
        self.episodes=0
        self.steps=0
        self.invalid=0
        self.transition_fingerprints=set()

    def observe(self,episode,public_bit):
        self.episodes+=1
        for step in episode["trace"]:
            before=step["before"]
            key=encode_key(transition_key(
                step["op"],before[0],before[1],public_bit,
                step["read_observed"]))
            post=encode_value((step["after"][0],step["after"][1],
                               step["invalid_delta"]))
            self.counts[key][post]+=1
            self.transition_fingerprints.add(key)
            self.steps+=1
            self.invalid+=step["invalid_delta"]

    def outcome(self,op,selected,work,public_bit,read_bit):
        key=encode_key(transition_key(
            op,selected,work,public_bit,
            read_bit if op=="R" and selected else -1))
        observations=self.counts.get(key)
        if not observations:
            return None,0
        # Maximum-likelihood transition; deterministic observed operators.
        result_str,count=max(observations.items(),key=lambda x:(x[1],x[0]))
        result=tuple(map(int,result_str.split(",")))
        return result,int(sum(observations.values()))

    def simulate(self,seq,public_bit,hypothetical_readbit):
        selected=0
        work=-1
        invalid=0
        uncertainty=0.
        for op in seq:
            outcome,n=self.outcome(op,selected,work,public_bit,hypothetical_readbit)
            if outcome is None:
                return None,True
            selected,work,delta=outcome
            invalid+=delta
            uncertainty+=1/math.sqrt(n)
        ans=0 if work==-1 else work
        return (ans,invalid,uncertainty),False

    def plan(self,source,flip,public_bit,include_heldout=True):
        """Choose a NEW plan using predicted transitions, not reward labels."""
        best=None
        best_score=-float("inf")
        candidates=PROGRAMS if include_heldout else TRAIN_PROGRAMS
        for seq in candidates:
            predicted=[]
            unknown=False
            for hypothetical_f4 in (0,1):
                result,missing=self.simulate(seq,public_bit,hypothetical_f4)
                if missing:
                    unknown=True
                    break
                predicted.append((hypothetical_f4,result))
            if unknown:
                continue
            correct=0.
            penalty=0.
            for bit,(answer,invalid,uncertainty) in predicted:
                goal=(public_bit if source=="PUBLIC" else bit)^flip
                correct+=int(answer==goal)/2
                penalty+=(.10*invalid+.002*uncertainty)/2
            score=correct-.01*len(seq)-penalty
            if score>best_score+1e-12 or (
                abs(score-best_score)<=1e-12 and
                (best is None or len(seq)<len(best) or
                 (len(seq)==len(best) and seq<best))):
                best=seq
                best_score=score
        return (best if best is not None else ()),float(best_score)

    def novelty(self,seq,source,public_bit):
        """No oracle, no predicted rewards; propose weakly covered transitions."""
        if not seq:
            return 0.
        quality=0.
        for hypothetical in (0,1):
            selected,work=0,-1
            for op in seq:
                key=encode_key(transition_key(op,selected,work,public_bit,
                                             hypothetical if op=="R" and selected else -1))
                n=sum(self.counts.get(key,{}).values())
                quality+=1./math.sqrt(n+1)
                next_state,_=self.outcome(op,selected,work,public_bit,hypothetical)
                if next_state is not None:
                    selected,work,_=next_state
        return quality/(2*len(seq))

    def snapshot(self):
        return {
            "episodes":self.episodes,"steps":self.steps,"invalid":self.invalid,
            "transition_signatures":len(self.counts),
            "transitions":{
                k:dict(sorted(v.items())) for k,v in sorted(self.counts.items())
            },
        }

    @classmethod
    def from_snapshot(cls,record):
        o=cls()
        o.episodes=int(record["episodes"])
        o.steps=int(record["steps"])
        o.invalid=int(record["invalid"])
        for k,v in record["transitions"].items():
            o.counts[k]=Counter(v)
        return o

class RewardMemory:
    """Explicit per-context, per-program scoreboard, no operator effects."""
    def __init__(self):
        self.table=defaultdict(lambda:[0,0.,0])
        self.episodes=0

    def observe(self,row,program_id,episode):
        context=(row["task"],row["flip"],row["public_bit"],int(program_id))
        entry=self.table[context]
        entry[0]+=1
        entry[1]+=episode["reward"]
        entry[2]+=episode["success"]
        self.episodes+=1

    def mean(self,source,flip,public_bit,pid):
        x=self.table.get((source,flip,public_bit,pid))
        return None if x is None else x[1]/x[0]

    def rank(self,source,flip,public_bit,mode):
        best=None
        best_score=-1e30
        for pid,seq in enumerate(PROGRAMS):
            if seq not in TRAIN_PROGRAMS:
                continue
            if mode=="reward_table":
                score=self.mean(source,flip,public_bit,pid)
                if score is None:
                    means=[entry[1]/entry[0] for (src,fl,b,p),entry
                           in self.table.items() if p==pid]
                    score=sum(means)/len(means) if means else -2.
            elif mode=="additive_reward_table":
                x0=self.mean("PUBLIC",0,public_bit,pid)
                x1=self.mean("PUBLIC",1,public_bit,pid)
                x2=self.mean("MEMORY",0,public_bit,pid)
                if x0 is not None and x1 is not None and x2 is not None:
                    score=x2+x1-x0 if source=="MEMORY" and flip==1 else (
                        self.mean(source,flip,public_bit,pid) or 0.)
                else:
                    score=-2.
            else:
                raise ValueError(mode)
            if score>best_score+1e-12 or (abs(score-best_score)<=1e-12 and
               (best is None or len(seq)<len(best))):
                best,best_score=seq,score
        return (best if best is not None else ()),float(best_score)

    def snapshot(self):
        return {
            "episodes":self.episodes,
            "scores":[
                {"source":a,"flip":b,"public_bit":c,"program_id":p,
                 "count":v[0],"total_reward":v[1],"success_count":v[2]}
                for (a,b,c,p),v in sorted(self.table.items())
            ]
        }

    @classmethod
    def from_snapshot(cls,record):
        o=cls()
        o.episodes=record["episodes"]
        for t in record["scores"]:
            o.table[(t["source"],t["flip"],t["public_bit"],
                     t["program_id"])]=[
                         t["count"],t["total_reward"],t["success_count"]]
        return o
