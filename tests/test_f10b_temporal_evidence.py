"""Prospective F10-B data validity, no hidden labels, and intervention contracts."""
import unittest
import numpy as np
import torch
from observatory.f10b_temporal_evidence_task import (
    KEYS, CAPACITY, EVENT_FEATURES, PUBLIC_FEATURES, WORLD_EVENTS,
    TRAIN_WORLDS, DEV_WORLDS, CELLS, seed_for, world,
    _correct_from_public, stale_memory, forged_credential,
    PublicEventF4,
)
from observatory.f10b_train_evidence_actors import EvidenceActor

class F10BTests(unittest.TestCase):
    def test_splits_and_no_confirm(self):
        train={seed_for("train",i,j) for i in range(4) for j in range(TRAIN_WORLDS)}
        dev={seed_for("dev",i,j) for i in range(4) for j in range(DEV_WORLDS)}
        self.assertEqual((len(train),len(dev)),(1024,256))
        self.assertFalse(train&dev)
        self.assertLess(max(dev),13300000)
        for a in (("confirm",0,0),("train",0,256),("dev",3,64),
                  ("train",4,0),("dev",-1,0)):
            with self.assertRaises(ValueError):seed_for(*a)

    def test_every_stratum_has_exactly_shared_history_and_late_query(self):
        for seed in (13100000,13100001,13200000,13200001):
            cases=world(seed)
            self.assertEqual(len(cases),4)
            self.assertEqual({c["cell"] for c in cases},
                             {a+"|"+b for a,b in CELLS})
            self.assertEqual(len({c["key"] for c in cases}),4)
            self.assertEqual(len({c["shared_history_sha256"] for c in cases}),1)
            for c in cases:
                public=c["public"]
                memory=c["memory"]
                self.assertEqual(public.shape,(WORLD_EVENTS+1,PUBLIC_FEATURES))
                self.assertEqual(memory.shape,(CAPACITY,EVENT_FEATURES))
                self.assertTrue(np.array_equal(public[:-1],cases[0]["public"][:-1]))
                self.assertEqual(public[:-1,-1].sum(),0)
                self.assertEqual(public[-1,-1],1)
                self.assertEqual(np.argmax(public[-1,:KEYS]),c["key"])
                self.assertTrue(np.array_equal(memory[-WORLD_EVENTS:,:],public[:-1,:-1]))
                self.assertEqual(_correct_from_public(public[:-1],c["key"]),c["truth"])
                self.assertTrue(np.isin(memory[:,KEYS+1],[0,1]).all())
                self.assertEqual(np.count_nonzero(memory[:CAPACITY-WORLD_EVENTS]),0)

    def test_counterfactual_attacks_change_only_external_log(self):
        for c in world(13100000):
            original=c["memory"].copy()
            history=c["public"].copy()
            stale=stale_memory(c)
            forged=forged_credential(c)
            self.assertTrue(np.array_equal(original,c["memory"]))
            self.assertTrue(np.array_equal(history,c["public"]))
            self.assertEqual(stale.shape,original.shape)
            self.assertEqual(forged.shape,original.shape)
            # Stale removes evidence, cannot create fake public evidence.
            self.assertTrue(np.all(np.logical_or(stale==original,stale==0.)))
            if c["latest_unsigned"]:
                self.assertEqual(np.count_nonzero(forged-original),1)
                self.assertEqual(np.max(forged-original),1)
            else:
                self.assertTrue(np.array_equal(forged,original))

    def test_actor_sees_public_query_and_log_only(self):
        cases=world(13100002)
        model=EvidenceActor()
        x=torch.tensor(np.stack([c["public"] for c in cases]))
        m=torch.tensor(np.stack([c["memory"] for c in cases]))
        with torch.no_grad():
            a=model(x,m)
            b=model(x,torch.zeros_like(m))
        self.assertEqual(tuple(a.shape),(4,))
        self.assertEqual(tuple(b.shape),(4,))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.isfinite(b).all())

if __name__=="__main__":
    unittest.main()
