"""F10-E preregistered 2x2 causal task and frozen architecture preflight tests."""
import unittest
import numpy as np
import torch

from observatory.f10e_public_factorial_worlds import (
    dev_seed,world,N_WORLDS,COUNTS,TIMINGS,SLOTS,VARIANTS,
    public_last_any,public_last_signed
)
from observatory.f10c_relational_actor_training import RelationalActor
from observatory.f10b_train_evidence_actors import EvidenceActor
from observatory.f10b_temporal_evidence_task import (
    KEYS,EVENT_FEATURES,PUBLIC_FEATURES,WORLD_EVENTS
)

class F10EPublicFactorial(unittest.TestCase):
    def test_unique_new_seeds_and_no_confirmation(self):
        seeds={dev_seed(i,j) for i in range(4) for j in range(N_WORLDS)}
        self.assertEqual(len(seeds),432)
        self.assertEqual(min(seeds),16200000)
        self.assertLess(max(seeds),16300000)
        for i,j in ((4,0),(-1,0),(0,108)):
            with self.assertRaises(ValueError):dev_seed(i,j)

    def test_truth_table_and_public_counterfactual_locality(self):
        for replica in (0,2):
            for j in range(N_WORLDS):
                cases=world(dev_seed(replica,j),replica,j)
                by={c["variant"]:c for c in cases}
                self.assertEqual(set(by),set(VARIANTS))
                self.assertEqual(len({c["key"] for c in cases}),1)
                self.assertEqual(len({c["base_value"] for c in cases}),1)
                self.assertEqual(len({c["record_count"] for c in cases}),1)
                self.assertEqual(len({c["timing"] for c in cases}),1)
                self.assertEqual(len({c["challenge_source"] for c in cases}),1)
                self.assertEqual(len({c["confirm_source"] for c in cases}),1)
                b=cases[0]["base_value"]
                for c in cases:
                    self.assertEqual(c["truth"],1-b if c["variant"]=="S_after" else b)
                    self.assertEqual(public_last_signed(c),c["truth"])
                    self.assertEqual(c["public"].shape,(61,PUBLIC_FEATURES))
                    self.assertEqual(c["memory"].shape,(64,EVENT_FEATURES))
                    self.assertTrue(np.array_equal(c["memory"][-60:,:],c["public"][:-1,:-1]))
                    self.assertEqual(float(c["public"][:-1,-1].sum()),0.)
                    self.assertEqual(float(c["public"][-1,-1]),1.)
                    self.assertEqual(int(c["public"][-1,c["key"]]),1)
                    self.assertEqual(sum(r[c["key"]]>0 for r in c["public"][:-1]),
                                     c["record_count"])
                    self.assertIn(public_last_any(c),(0,1))
                # The only difference between same-order signed/unsigned:
                # exactly ONE PUBLIC CREDENTIAL BIT of X.
                for a,d in (("U_before","S_before"),("U_after","S_after")):
                    diff=by[d]["public"]-by[a]["public"]
                    self.assertEqual(np.count_nonzero(diff),1)
                    self.assertEqual(float(np.abs(diff).sum()),1.)
                    self.assertTrue(np.array_equal(by[d]["public"][:,-1],
                                                   by[a]["public"][:,-1]))
                # Across order manipulations, NOTHING changes away from
                # the designated two timestamps. Query remains identical.
                slots=by["U_before"]["slots"]
                self.assertEqual(len(slots),2)
                for a,d in (("U_before","U_after"),("S_before","S_after")):
                    x=by[a]["public"];y=by[d]["public"]
                    diff=np.flatnonzero(np.any(x!=y,axis=1))
                    self.assertEqual(set(diff),set(slots))
                    self.assertTrue(np.array_equal(x[-1],y[-1]))

    def test_grid_counts_exact_12_per_cell_each_replica(self):
        for replica in range(4):
            seen={(k,t):0 for k in COUNTS for t in TIMINGS}
            for j in range(N_WORLDS):
                row=world(dev_seed(replica,j),replica,j)[0]
                seen[(row["record_count"],row["timing"])]+=1
            self.assertEqual(set(seen.values()),{12})

    def test_original_model_interfaces_accept_all_four_variants(self):
        cases=world(dev_seed(0,0),0,0)
        history=torch.from_numpy(np.stack([c["public"] for c in cases]))
        memory=torch.from_numpy(np.stack([c["memory"] for c in cases]))
        for model in (RelationalActor(),EvidenceActor()):
            model.eval()
            with torch.no_grad():
                live=model(history,memory)
                blank=model(history,torch.zeros_like(memory))
            self.assertEqual(tuple(live.shape),(4,))
            self.assertEqual(tuple(blank.shape),(4,))
            self.assertTrue(torch.isfinite(live).all())
            self.assertTrue(torch.isfinite(blank).all())

if __name__=="__main__":
    unittest.main()
