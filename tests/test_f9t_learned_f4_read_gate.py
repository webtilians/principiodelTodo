"""F9-T contract tests: real actor H/C gate, F4 causal boundary and disjoint seeds."""
import unittest
import numpy as np
import torch

from observatory.f9i_selective_memory import DIM
from observatory.f9t_learned_f4_read_gate import (
    ACTOR_DIM,STATE_DIM,GATE_IN,ReadHead,READ_COST,LEARNED_MODES,
    seed_for,gate_features,gated_actor_input,gate_decision,
    scheduled_read,public_object_visible,
)

class TestsF9TLearnedF4ReadGate(unittest.TestCase):
    def test_new_seed_sets_do_not_touch_historical_F9(self):
        train={seed_for("train",i,j) for i in range(4) for j in range(5)}
        dev={seed_for("dev",i,j) for i in range(4) for j in range(5)}
        confirm={seed_for("confirm",i,j) for i in range(4) for j in range(5)}
        self.assertEqual((len(train),len(dev),len(confirm)),(20,20,20))
        self.assertFalse(train&dev or train&confirm or dev&confirm)
        self.assertTrue(all(x>6_000_000 for x in train|dev|confirm))
        for params in (("train",4,0),("train",0,600),
                       ("dev",0,20),("confirm",0,40),
                       ("heldout",0,0),("confirm",-1,0)):
            with self.assertRaises(ValueError):seed_for(*params)

    def test_gate_sees_only_148_public_plus_PREVIOUS_hc(self):
        raw=np.linspace(0,1,ACTOR_DIM,dtype=np.float32)
        h=np.full((1,1,64),0.123,np.float32)
        c=np.full((1,1,64),-0.321,np.float32)
        a=gate_features(raw,(h,c),True)
        b=gate_features(raw,(h,c),False)
        self.assertEqual(tuple(a.shape),(GATE_IN,))
        self.assertEqual(GATE_IN,276)
        self.assertTrue(torch.equal(a[:ACTOR_DIM],b[:ACTOR_DIM]))
        self.assertTrue(torch.equal(b[ACTOR_DIM:],torch.zeros(STATE_DIM)))
        self.assertFalse(torch.equal(a[ACTOR_DIM:],b[ACTOR_DIM:]))
        self.assertEqual(tuple(gate_features(raw,None).shape),(276,))
        with self.assertRaises(AssertionError):
            gate_features(np.zeros(DIM,dtype=np.float32),None)
        with self.assertRaises(AssertionError):
            gate_features(np.zeros(149,dtype=np.float32),None)

    def test_gate_may_only_erase_F4_readout(self):
        raw=np.arange(ACTOR_DIM,dtype=np.float32)/ACTOR_DIM
        f4=np.arange(DIM-ACTOR_DIM,dtype=np.float32)
        encoded=np.concatenate((raw,f4))
        read=gated_actor_input(raw,encoded,True)
        skip=gated_actor_input(raw,encoded,False)
        self.assertTrue(np.array_equal(read,encoded))
        self.assertTrue(np.array_equal(skip[:ACTOR_DIM],read[:ACTOR_DIM]))
        self.assertTrue(np.all(skip[ACTOR_DIM:]==0))
        self.assertFalse(np.array_equal(skip,read))
        corrupt=encoded.copy()
        corrupt[10]=99
        with self.assertRaises(AssertionError):
            gated_actor_input(raw,corrupt,False)

    def test_binary_head_produces_trainable_logprob_without_teacher(self):
        torch.manual_seed(123)
        head=ReadHead()
        raw=np.zeros(ACTOR_DIM,np.float32)
        h=np.ones((1,1,64),np.float32)
        c=np.zeros((1,1,64),np.float32)
        self.assertEqual(tuple(head(gate_features(raw,(h,c))).shape),())
        a=gate_decision("learned_hc",head,raw,(h,c),0,None,stochastic=True)
        b=gate_decision("learned_public_only",head,raw,(h,c),0,None,stochastic=True)
        self.assertIsInstance(a[0],bool)
        self.assertTrue(torch.isfinite(a[2]))
        self.assertTrue(torch.isfinite(a[3]))
        self.assertIsInstance(b[0],bool)
        loss=-a[2]
        loss.backward()
        self.assertTrue(any(p.grad is not None for p in head.parameters()))

    def test_baselines_do_not_consult_future_identity(self):
        raw=np.zeros(ACTOR_DIM,np.float32)
        schedule=[scheduled_read("periodic_4",raw,t,None)[0] for t in range(10)]
        self.assertEqual(schedule,[True,False,False,False,True,False,False,False,True,False])
        self.assertFalse(public_object_visible(raw))
        for mode in ("always","never","periodic_4","public_event_clock"):
            a=gate_decision(mode,None,raw,None,0,None)
            self.assertIsNone(a[2])
            self.assertIsNone(a[3])
        self.assertGreater(READ_COST,0)
        self.assertEqual(len(LEARNED_MODES),2)

if __name__=="__main__":
    unittest.main()
