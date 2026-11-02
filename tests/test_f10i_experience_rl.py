"""F10-I preflight: prospective TRAIN only; no DEV/OOD generated in tests."""
import unittest

import numpy as np
import torch

from observatory.f10i_worlds import (
    world,seed_for,KEYS,FEATURES,EVENTS,VARIANTS,
)
from observatory.f10i_actor import ExperienceActor,count_parameters
from observatory.f10i_experiment import tensors,reward,ARMS

class F10IContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_sealed_splits_and_unseen_namespaces(self):
        self.assertEqual(seed_for("train",0,0),20100000)
        self.assertEqual(seed_for("dev",0,0),20200000)
        self.assertEqual(seed_for("ood",0,0),20400000)
        for item in (("confirm",0,0),("train",4,0),
                     ("train",0,512),("ood",0,128),("dev",-1,0)):
            with self.assertRaises(ValueError):seed_for(*item)

    def test_new_training_counterfactuals_have_multiple_signed_changes(self):
        # NEVER open F10-I DEV or OOD inside the preflight.
        for j in range(36):
            rows=world("train",0,j)
            self.assertEqual({r["variant"] for r in rows},set(VARIANTS))
            self.assertEqual(len({r["hash"] for r in rows}),4)
            self.assertEqual(len({r["q"] for r in rows}),1)
            self.assertEqual(len({r["inv"] for r in rows}),1)
            by={r["variant"]:r for r in rows}
            self.assertEqual(by["U_late"]["truth"],by["U_early"]["truth"])
            self.assertEqual(by["U_late"]["truth"],by["S_early"]["truth"])
            self.assertNotEqual(by["U_late"]["truth"],by["S_late"]["truth"])
            self.assertIn(rows[0]["k"],(6,9))
            for r in rows:
                mem=r["memory"]
                h=r["history"]
                q=r["q"]
                self.assertEqual(mem.shape,(EVENTS,FEATURES))
                self.assertEqual(h.shape,(EVENTS+1,FEATURES+1))
                np.testing.assert_array_equal(h[:-1,:FEATURES],mem)
                signed=np.flatnonzero((mem[:,q]>.5)&(mem[:,KEYS+1]>.5))
                self.assertGreaterEqual(len(signed),3)
                self.assertGreaterEqual(r["signed_value_changes"],2)
                top=signed[np.argmax(mem[signed,KEYS+2])]
                self.assertEqual(r["winner"],top)
                self.assertEqual(r["truth"],int(mem[top,KEYS]>0)^r["inv"])
                self.assertTrue(any(v[q]>.5 and v[KEYS+1]==0 and
                                    v[KEYS+2]>mem[top,KEYS+2] for v in mem))
                self.assertIn(r["sensor"],(0,1))
                self.assertIn(r["delay"],(2,3,4))
            for a,b in (("U_late","S_late"),("U_early","S_early")):
                delta=by[a]["memory"]!=by[b]["memory"]
                self.assertEqual(int(np.count_nonzero(delta)),1)
                self.assertEqual(int(np.count_nonzero(delta[:,KEYS+1])),1)
            for a,b in (("U_late","U_early"),("S_late","S_early")):
                delta=by[a]["memory"]!=by[b]["memory"]
                self.assertEqual(int(np.count_nonzero(delta)),2)
                self.assertEqual(int(np.count_nonzero(delta[:,KEYS+2])),2)
            for v in VARIANTS[1:]:
                np.testing.assert_array_equal(
                    by["U_late"]["memory"][:,:KEYS+1],
                    by[v]["memory"][:,:KEYS+1])
                np.testing.assert_array_equal(
                    by["U_late"]["memory"][:,KEYS+3:],
                    by[v]["memory"][:,KEYS+3:])

    def test_policy_is_frozen_shape_and_memory_ablation_causal(self):
        torch.manual_seed(205000)
        cases=world("train",0,0)
        x,m,y,w,sensor,delay=tensors(cases)
        self.assertEqual(set(ARMS),{"rl_standard","rl_pair","rl_pair_predict","supervised_ceiling"})
        self.assertEqual(x.shape,(4,49,30))
        self.assertEqual(m.shape,(4,48,29))
        model=ExperienceActor()
        self.assertGreater(count_parameters(model),0)
        with torch.no_grad():
            regular=model.predict(x,m)
            skip=model.predict(x,m,"force_skip")
            changed=m.clone()
            changed[:,:,KEYS]*=-1
            self.assertTrue(torch.equal(
                model.predict(x,changed,"force_skip")["action"],skip["action"]))
            self.assertTrue(torch.equal(
                model.encode(x,m)[3],model.encode(x,changed)[3]))
            self.assertTrue(torch.equal(
                model.encode(x,m)[4],model.encode(x,changed)[4]))
            self.assertTrue(torch.equal(
                model.predict(x,m,"force_read")["read"],torch.ones(4,dtype=torch.long)))
            sampled=model.sample(x,m)
            self.assertEqual(sampled["log_prob"].shape,(4,))
            self.assertEqual(sampled["sensor_logits"].shape,(4,))
            self.assertEqual(reward(sampled,y,delay).shape,(4,))
            self.assertEqual(regular["action"].shape,(4,))
            self.assertTrue(torch.isfinite(model.expert_objective(x,m,w,y)))

if __name__=="__main__":unittest.main()
