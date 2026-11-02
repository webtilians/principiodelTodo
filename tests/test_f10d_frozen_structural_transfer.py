"""F10-D evidence integrity: no training, no old seeds, variable k, real public truth."""
import unittest
import numpy as np
import torch
from observatory.f10d_structural_transfer_worlds import (
    dev_seed,world,DEV_WORLDS,CELLS,CARDINALITIES,SOURCE_MODES,
    last_signed_public,last_any_observed,majority_observed
)
from observatory.f10b_temporal_evidence_task import KEYS,EVENT_FEATURES,PUBLIC_FEATURES
from observatory.f10c_relational_actor_training import RelationalActor
from observatory.f10b_train_evidence_actors import EvidenceActor
from observatory.f10b_temporal_evidence_task import stale_memory,forged_credential

class FrozenStructuralTransferTests(unittest.TestCase):
    def test_seed_is_entirely_new_development_only(self):
        seeds={dev_seed(i,j) for i in range(4) for j in range(DEV_WORLDS)}
        self.assertEqual(len(seeds),256)
        self.assertEqual(min(seeds),15200000)
        self.assertLess(max(seeds),15300000)
        self.assertTrue(all(s>=15000000 for s in seeds))
        for i,j in ((4,0),(0,64),(-1,0)):
            with self.assertRaises(ValueError):dev_seed(i,j)

    def test_independent_worlds_rotate_4strata_and_4lengths(self):
        for i in (0,2):
            by_cell={cell:{k:0 for k in CARDINALITIES} for cell in CELLS}
            by_source={mode:0 for mode in SOURCE_MODES}
            for j in range(DEV_WORLDS):
                seed=dev_seed(i,j)
                rows=world(seed,i,j)
                self.assertEqual(len(rows),4)
                self.assertEqual({r["cell"] for r in rows},set(CELLS))
                self.assertEqual({r["record_count"] for r in rows},set(CARDINALITIES))
                self.assertEqual(len({r["key"] for r in rows}),4)
                self.assertEqual(len({r["history_sha256"] for r in rows}),1)
                self.assertTrue(all(np.array_equal(rows[0]["public"][:-1],r["public"][:-1]) for r in rows))
                for r in rows:
                    by_cell[r["cell"]][r["record_count"]]+=1
                    self.assertEqual(r["public"].shape,(61,PUBLIC_FEATURES))
                    self.assertEqual(r["memory"].shape,(64,EVENT_FEATURES))
                    self.assertEqual(r["public"][:-1,-1].sum(),0)
                    self.assertEqual(r["public"][-1,-1],1)
                    self.assertEqual(last_signed_public(r),r["truth"])
                    self.assertIn(last_any_observed(r),(0,1))
                    self.assertIn(majority_observed(r),(0,1))
                    self.assertTrue(np.array_equal(r["memory"][-60:],r["public"][:-1,:-1]))
                    signed=[v for v in r["public"][:-1] if v[r["key"]]>0 and v[KEYS+1]==1]
                    self.assertGreaterEqual(len(signed),1)
                    self.assertEqual(int(signed[-1][KEYS]),r["truth"])
                    z=stale_memory(r)
                    fake=forged_credential(r)
                    self.assertTrue(np.all(np.logical_or(z==r["memory"],z==0)))
                    if r["latest_unsigned"]:
                        self.assertEqual(np.count_nonzero(fake-r["memory"]),1)
                by_source[rows[0]["source_mode"]]+=1
            self.assertTrue(all(v==16 for counts in by_cell.values() for v in counts.values()))
            self.assertTrue(all(v==16 for v in by_source.values()))

    def test_frozen_actor_interfaces_accept_unseen_cardinalities(self):
        records=world(dev_seed(0,0),0,0)
        x=torch.from_numpy(np.stack([r["public"] for r in records]))
        m=torch.from_numpy(np.stack([r["memory"] for r in records]))
        for actor in (RelationalActor(),EvidenceActor()):
            actor.eval()
            with torch.no_grad():
                p=actor(x,m)
                q=actor(x,torch.zeros_like(m))
            self.assertEqual(tuple(p.shape),(4,))
            self.assertEqual(tuple(q.shape),(4,))
            self.assertTrue(torch.isfinite(p).all())
            self.assertTrue(torch.isfinite(q).all())

if __name__=="__main__":unittest.main()
