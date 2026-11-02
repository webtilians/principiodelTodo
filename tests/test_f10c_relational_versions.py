"""F10-C public temporal version integrity and learned pairwise resolver tests."""
import unittest
import numpy as np
import torch
from observatory.f10c_multi_version_worlds import (
    seed_for,world,TRAIN_WORLDS,DEV_WORLDS,GROUP_NAMES
)
from observatory.f10c_relational_actor_training import RelationalActor
from observatory.f10b_temporal_evidence_task import (
    KEYS,WORLD_EVENTS,EVENT_FEATURES,PUBLIC_FEATURES,
    _correct_from_public,stale_memory,forged_credential
)
class F10CTests(unittest.TestCase):
    def test_new_world_ids_never_touch_prior_partitions(self):
        a={seed_for("train",i,j) for i in range(4) for j in range(TRAIN_WORLDS)}
        b={seed_for("dev",i,j) for i in range(4) for j in range(DEV_WORLDS)}
        self.assertEqual(len(a),2048)
        self.assertEqual(len(b),256)
        self.assertFalse(a&b)
        self.assertEqual(min(a),14100000)
        self.assertEqual(min(b),14200000)
        self.assertLess(max(b),14300000)
        for args in (("confirm",0,0),("dev",0,64),("train",0,512),
                     ("train",4,0),("dev",-1,0)):
            with self.assertRaises(ValueError):seed_for(*args)
    def test_public_truth_and_exact_four_shared_history_queries(self):
        for seed in (14100000,14100001,14200000,14200001):
            cases=world(seed)
            self.assertEqual({c["cell"] for c in cases},set(GROUP_NAMES))
            self.assertEqual(len({c["history_sha256"] for c in cases}),1)
            self.assertEqual(len({c["key"] for c in cases}),4)
            for c in cases:
                self.assertEqual(c["number_of_matching_reports"],5)
                self.assertGreaterEqual(c["number_of_contradictions"],1)
                self.assertEqual(c["public"].shape,(61,PUBLIC_FEATURES))
                self.assertEqual(c["memory"].shape,(64,EVENT_FEATURES))
                self.assertTrue(np.array_equal(c["public"][:-1],
                                               cases[0]["public"][:-1]))
                self.assertEqual(float(c["public"][:-1,-1].sum()),0.)
                self.assertEqual(float(c["public"][-1,-1]),1.)
                self.assertEqual(_correct_from_public(c["public"][:-1],c["key"]),
                                 c["truth"])
                self.assertTrue(np.array_equal(c["memory"][-WORLD_EVENTS:],
                                               c["public"][:-1,:-1]))
                stale=stale_memory(c)
                forged=forged_credential(c)
                self.assertTrue(np.all(np.logical_or(stale==c["memory"],stale==0)))
                self.assertEqual(forged.shape,c["memory"].shape)
                if c["last_unsigned"]:
                    self.assertEqual(np.count_nonzero(forged-c["memory"]),1)
    def test_pairwise_actor_reads_public_key_and_handles_blank_memory(self):
        cases=world(14100002)
        x=torch.from_numpy(np.stack([c["public"] for c in cases]))
        m=torch.from_numpy(np.stack([c["memory"] for c in cases]))
        model=RelationalActor()
        with torch.no_grad():
            yes=model(x,m)
            no=model(x,torch.zeros_like(m))
        self.assertEqual(tuple(yes.shape),(4,))
        self.assertEqual(tuple(no.shape),(4,))
        self.assertTrue(torch.isfinite(yes).all())
        self.assertTrue(torch.isfinite(no).all())

if __name__=="__main__":unittest.main()
