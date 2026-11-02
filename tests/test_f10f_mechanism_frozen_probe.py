"""F10-F preflight: new worlds, public winner rule, exact original actor parity."""
import unittest

import numpy as np
import torch

from observatory.f10f_mechanism_worlds import (
    world,dev_seed,N_WORLDS,VARIANTS,COUNTS,TIMINGS,
    public_last_signed,
)
from observatory.f10f_frozen_mechanism_probe import inspect,aggregate_group
from observatory.f10c_relational_actor_training import RelationalActor
from observatory.f10b_train_evidence_actors import EvidenceActor
from observatory.f10b_temporal_evidence_task import KEYS,PUBLIC_FEATURES,EVENT_FEATURES

class ProbeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
    def test_new_worlds_and_seal(self):
        seeds={dev_seed(i,j) for i in range(4) for j in range(N_WORLDS)}
        self.assertEqual((len(seeds),min(seeds)),(432,17200000))
        self.assertTrue(max(seeds)<17300000)
        for a in ((4,0),(0,108),(-1,0)):
            with self.assertRaises(ValueError):dev_seed(*a)
    def test_public_truth_and_factorial_and_grid(self):
        for i in (0,1):
            grid={(k,t):0 for k in COUNTS for t in TIMINGS}
            for j in range(N_WORLDS):
                cases=world(dev_seed(i,j),i,j)
                self.assertEqual({c["variant"] for c in cases},set(VARIANTS))
                self.assertEqual(len({c["key"] for c in cases}),1)
                self.assertEqual(len({c["history_sha256"] for c in cases}),4)
                b=cases[0]["base_value"]
                grid[(cases[0]["record_count"],cases[0]["timing"])]+=1
                for c in cases:
                    self.assertEqual(c["truth"],(1-b if c["variant"]=="S_after" else b))
                    self.assertEqual(public_last_signed(c),c["truth"])
                    self.assertEqual(c["public"].shape,(61,PUBLIC_FEATURES))
                    self.assertEqual(c["memory"].shape,(64,EVENT_FEATURES))
                    self.assertTrue(np.array_equal(c["memory"][-60:],
                                                   c["public"][:-1,:-1]))
            self.assertEqual(set(grid.values()),{12})
    def test_exact_forward_reconstruction_and_oracle_patch_contracts(self):
        torch.manual_seed(12411)
        model=RelationalActor()
        unary=EvidenceActor()
        model.eval();unary.eval()
        for i,j in ((0,0),(0,1),(0,8),(0,17)):
            for c in world(dev_seed(i,j),i,j):
                r=inspect(model,unary,c)
                self.assertLessEqual(r["read_logit_exact_parity_error"],1e-5)
                self.assertLessEqual(r["skip_logit_exact_parity_error"],1e-5)
                self.assertEqual(len(r["indices"]),5)
                self.assertEqual(len(r["topk_reports"]),5)
                self.assertEqual(r["winner_credential"],1)
                self.assertEqual(r["winner_value"],r["truth"])
                self.assertEqual(r["winner_in_topk"],r["winner_slot"] in r["indices"])
                self.assertEqual(r["real_correct"],int(r["scores"]["real"]>=.5)==r["truth"])
                if r["winner_in_topk"]:
                    self.assertAlmostEqual(r["scores"]["patch_winner_include"],
                                           r["scores"]["real"],delta=1e-6)
        stats=aggregate_group([inspect(model,unary,c)
                               for c in world(dev_seed(0,0),0,0)])
        self.assertEqual(stats["cases"],4)
        self.assertEqual(stats["correct"]+stats["errors"],4)

if __name__=="__main__":unittest.main()
