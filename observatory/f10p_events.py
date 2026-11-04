"""F10-P: infer executable effects from ONLY public tool-output messages.

No learner method accepts a world row, internal before/after, slot index,
read_observed, winner, truth or private F4. EventEffects receives a list
of (operation name, public output/missing) from ACTUALLY EXECUTED actions.
The tool-output schema and planner state abstraction are GIVEN by researcher,
while P/X value maps and S/R validity are inferred from real observations.
"""
from __future__ import annotations

from collections import Counter
import itertools
import math
from observatory.f10p_worlds import PROGRAMS

MIN_EXAMPLES=10
MIN_CONSISTENCY=.90
OPS=("P","S","R","X")
CHANNELS=("E100","E50","E0","E_noX","E_shuffleX")

def public_events(executed):
    """Serialize ONLY observable tool responses. Never return internal trace."""
    events=[]
    for step in executed["trace"]:
        op=step["op"]
        if op=="P":
            output={"value":int(step["after"][1])}
        elif op=="S":
            output={"ack":True}
        elif op=="R":
            output=({"value":int(step["read_observed"])}
                    if step["read_observed"] in (0,1) else {"error":True})
        elif op=="X":
            output=({"value":int(step["after"][1])}
                    if step["invalid_delta"]==0 else {"error":True})
        else:
            raise AssertionError("unknown tool")
        assert set(output) in ({"value"},{"ack"},{"error"})
        events.append({"op":op,"output":output})
    return events

def project(events,arm,mask_rng,shuffle_rng):
    """Observation-only intervention. Does NOT change environment executions."""
    transformed=[]
    for event in events:
        output=dict(event["output"])
        op=event["op"]
        if arm=="E0" or (arm=="E_noX" and op=="X") or (
            arm=="E50" and mask_rng.integers(2)==0):
            output=None
        elif arm=="E_shuffleX" and op=="X" and "value" in output:
            output={"value":int(shuffle_rng.integers(2))}
        transformed.append({"op":op,"output":output})
    return transformed

class EventEffects:
    def __init__(self):
        self.episodes=0
        self.events_seen=0
        self.p={0:Counter(),1:Counter()}
        self.x={0:Counter(),1:Counter()}
        self.s_ack=0
        self.r_valid_after_s=0
        self.r_error_after_s=0

    def observe(self,events,public_bit):
        """Only public-event payloads and public input; NO private trace."""
        assert public_bit in (0,1)
        self.episodes+=1
        last_visible=None
        saw_ack=False
        for item in events:
            assert set(item)=={"op","output"}
            op=item["op"]; output=item["output"]
            assert op in OPS
            if output is None:
                # A missing value is UNKNOWN, never 0. Missing S ACK means
                # we cannot know from these observations if R was enabled.
                if op=="S":saw_ack=False
                if op in ("P","R","X"):last_visible=None
                continue
            assert isinstance(output,dict)
            assert set(output) in ({"value"},{"ack"},{"error"})
            self.events_seen+=1
            if op=="S":
                assert output=={"ack":True}
                self.s_ack+=1
                saw_ack=True
            elif op=="P":
                assert set(output)=={"value"}
                val=int(output["value"])
                assert val in (0,1)
                self.p[public_bit][val]+=1
                last_visible=val
            elif op=="R":
                if "value" in output:
                    assert output["value"] in (0,1)
                    if saw_ack:self.r_valid_after_s+=1
                    last_visible=int(output["value"])
                else:
                    assert output=={"error":True}
                    if saw_ack:self.r_error_after_s+=1
                    last_visible=None
            elif op=="X":
                if "value" in output:
                    val=int(output["value"])
                    assert val in (0,1)
                    if last_visible in (0,1):
                        self.x[last_visible][val]+=1
                    last_visible=val
                else:
                    assert output=={"error":True}
                    last_visible=None

    @staticmethod
    def learned(counts):
        n=sum(counts.values())
        if n<MIN_EXAMPLES:
            return None
        value,count=max(sorted(counts.items()),key=lambda v:v[1])
        if count/n<MIN_CONSISTENCY:
            return None
        return int(value)

    def simulate(self,seq,public_bit,hypothetical_read):
        """Predict from learned public results; F4 value only hypothetical."""
        work=None
        selection=False
        for op in seq:
            if op=="P":
                work=self.learned(self.p[public_bit])
                if work is None:return None
            elif op=="S":
                if self.s_ack<MIN_EXAMPLES:return None
                selection=True
            elif op=="R":
                if not selection or self.r_valid_after_s<MIN_EXAMPLES:
                    return None
                work=hypothetical_read
            elif op=="X":
                if work is None:return None
                work=self.learned(self.x[work])
                if work is None:return None
            else:
                raise ValueError(op)
        return 0 if work is None else int(work)

    def plan(self,source,flip,public_bit):
        assert source in ("PUBLIC","MEMORY") and flip in (0,1)
        assert public_bit in (0,1)
        best=()
        best_score=-float("inf")
        for seq in PROGRAMS:
            predicted=(self.simulate(seq,public_bit,0),
                       self.simulate(seq,public_bit,1))
            if any(x is None for x in predicted):
                continue
            success=0.
            for q,answer in enumerate(predicted):
                target=(public_bit if source=="PUBLIC" else q)^flip
                success+=int(answer==target)/2
            score=success-.01*len(seq)
            if score>best_score+1e-12 or (
                abs(score-best_score)<=1e-12
                and (len(seq)<len(best) or
                     (len(seq)==len(best) and seq<best))):
                best=seq
                best_score=score
        return best,float(best_score)

    def snapshot(self):
        return {
            "schema":"F10P_TOOL_OUTPUT_EVENTS_V1",
            "episodes":self.episodes,
            "events_seen":self.events_seen,
            "minimum_observations":MIN_EXAMPLES,
            "consistency_threshold":MIN_CONSISTENCY,
            "P":{"0":dict(sorted(self.p[0].items())),
                 "1":dict(sorted(self.p[1].items()))},
            "X":{"0":dict(sorted(self.x[0].items())),
                 "1":dict(sorted(self.x[1].items()))},
            "S_ack":self.s_ack,
            "R_valid_after_S_ack":self.r_valid_after_s,
            "R_error_after_S_ack":self.r_error_after_s,
        }

    @classmethod
    def from_snapshot(cls,record):
        assert record["schema"]=="F10P_TOOL_OUTPUT_EVENTS_V1"
        assert record["minimum_observations"]==MIN_EXAMPLES
        assert record["consistency_threshold"]==MIN_CONSISTENCY
        obj=cls()
        obj.episodes=int(record["episodes"])
        obj.events_seen=int(record["events_seen"])
        for name,slot in (("P",obj.p),("X",obj.x)):
            for b in (0,1):
                slot[b]=Counter({int(k):int(v) for k,v in
                                 record[name][str(b)].items()})
        obj.s_ack=int(record["S_ack"])
        obj.r_valid_after_s=int(record["R_valid_after_S_ack"])
        obj.r_error_after_s=int(record["R_error_after_S_ack"])
        return obj
