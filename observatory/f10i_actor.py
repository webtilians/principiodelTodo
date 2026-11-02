"""F10-I recurrent episodic retrieval actor shared by A/B/C/D; no oracle in policy.

Policy factors p(READ|H,q) p(slot|READ,H,F4,q) p(action|read_value,H,q).
Only selected report's value enters action head; pointer never reads values.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.distributions import Bernoulli,Categorical
from observatory.f10i_worlds import KEYS,EVENTS,FEATURES

WIDTH=32

class ExperienceActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru=nn.GRU(FEATURES+1,WIDTH,batch_first=True)
        self.selector=nn.Sequential(nn.Linear(6,16),nn.Tanh(),nn.Linear(16,1))
        self.gate=nn.Sequential(nn.Linear(WIDTH+1,16),nn.Tanh(),nn.Linear(16,1))
        self.head=nn.Sequential(nn.Linear(WIDTH+3,32),nn.Tanh(),nn.Linear(32,1))
        # Identical extra head exists in EVERY arm, only C trains it.
        self.sensor_head=nn.Sequential(nn.Linear(WIDTH+3,16),nn.Tanh(),nn.Linear(16,1))

    def encode(self,hist,mem,mode="read"):
        _,state=self.gru(hist)
        h=state[-1]
        query=hist[:,-1,:KEYS]
        inv=hist[:,-1,-1]
        valid=(mem[:,:,:KEYS]*query[:,None,:]).sum(-1)>.5
        sign=mem[:,:,KEYS+1]
        time=mem[:,:,KEYS+2]
        src=mem[:,:,KEYS+3:KEYS+5]
        if mode=="erase_sign":sign=torch.zeros_like(sign)
        if mode=="erase_time":time=torch.zeros_like(time)
        features=torch.stack((sign,time,src[:,:,0],src[:,:,1],time**2,sign*time),dim=-1)
        score=self.selector(features).squeeze(-1)
        logits=(score*4).masked_fill(~valid,-1e9)
        gate=self.gate(torch.cat((h,inv[:,None]),dim=1)).flatten()
        return h,inv,valid,logits,gate

    def interpret(self,h,inv,value,available):
        features=torch.cat((h,value[:,None],available[:,None],inv[:,None]),dim=1)
        return self.head(features).flatten(),self.sensor_head(features).flatten()

    def sample(self,hist,mem):
        h,inv,valid,selection_logits,gate_logit=self.encode(hist,mem)
        gates=Bernoulli(logits=gate_logit)
        indices=Categorical(logits=selection_logits)
        g=gates.sample()
        idx=indices.sample()
        value=mem[torch.arange(len(mem)),idx,KEYS]*g
        act_logits,sensor_logits=self.interpret(h,inv,value,g)
        actions=Bernoulli(logits=act_logits)
        a=actions.sample()
        lp=(gates.log_prob(g) +g*indices.log_prob(idx) + actions.log_prob(a))
        entropy=(gates.entropy()+g*indices.entropy()+actions.entropy())
        return {
            "read":g,"index":idx,"action":a,"log_prob":lp,
            "entropy":entropy,"sensor_logits":sensor_logits,
        }

    def expert_objective(self,hist,mem,winner,truth):
        # Control D ONLY. Privileged targets are forbidden in A/B/C losses.
        h,inv,valid,selection_logits,gate_logit=self.encode(hist,mem)
        value=mem[torch.arange(len(mem)),winner,KEYS]
        act_logits,_=self.interpret(h,inv,value,torch.ones_like(value))
        return (nn.functional.binary_cross_entropy_with_logits(
                   gate_logit,torch.ones_like(gate_logit))+
                nn.functional.cross_entropy(selection_logits,winner)+
                nn.functional.binary_cross_entropy_with_logits(act_logits,truth))

    def predict(self,hist,mem,mode="read"):
        """Deterministic policy. READ can be forced to isolate F4 causality."""
        h,inv,valid,selection_logits,gate_logit=self.encode(hist,mem,mode)
        idx=selection_logits.argmax(-1)
        g=(gate_logit>=0).float()
        if mode=="force_read":g=torch.ones_like(g)
        if mode=="force_skip":g=torch.zeros_like(g)
        values=mem[:,:,KEYS].clone()
        if mode=="shuffle":
            for b in range(len(values)):
                matching=torch.where(valid[b])[0]
                values[b,matching]=values[b,matching.roll(1)]
        value=values[torch.arange(len(values)),idx]*g
        logits,_=self.interpret(h,inv,value,g)
        return {
            "action":(logits>=0).long(),
            "read":g.long(),
            "selected":idx,
            "read_probability":torch.sigmoid(gate_logit),
            "action_probability":torch.sigmoid(logits),
        }

def count_parameters(model):
    return sum(x.numel() for x in model.parameters())
