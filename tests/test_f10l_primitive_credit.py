"""F10-L PRE-LAUNCH tests. Only TRAIN cases; OOD and CONFIRM stay sealed."""
import unittest
import numpy as np
import torch

from observatory.f10l_worlds import (
    xor_seed,xor_data,select_seed,select_world,MAX_SLOTS,FEATURES,VARIANTS
)
from observatory.f10l_train import (
    XorActor,SelectActor,xor_tensors,sel_tensors,
    uniform_distinct_pairs,reward,ALL_ARMS
)

class F10LIntegrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_splits_sealed(self):
        self.assertEqual(xor_seed("train",0,0),23100000)
        self.assertEqual(xor_seed("dev",0,0),23200000)
        self.assertEqual(select_seed("train",0,0),23105000)
        self.assertEqual(select_seed("dev",0,0),23205000)
        for request in (
            ("confirm",0,0),("ood",0,0),
            ("train",4,0),("train",0,4096),("dev",0,512)
        ):
            with self.assertRaises(ValueError):xor_seed(*request)
        for request in (
            ("confirm",0,0),("train",4,0),
            ("dev",0,128),("train",0,1024),("ood",0,128)
        ):
            with self.assertRaises(ValueError):select_seed(*request)

    def test_only_train_xor_and_reward_pairs(self):
        cases=xor_data("train",0)
        self.assertEqual(len(cases),4096)
        self.assertEqual(len({r["hash"] for r in cases}),4096)
        for b in (0,1):
            for flip in (0,1):
                sub=[r for r in cases if r["bit"]==b and r["flip"]==flip]
                self.assertEqual(len(sub),1024)
                self.assertTrue(all(r["truth"]==b^flip for r in sub))
        x,y=xor_tensors(cases[:128])
        torch.manual_seed(235000)
        actor=XorActor()
        logits=actor(x)
        a=torch.zeros_like(logits)
        b=torch.ones_like(logits)
        ra=reward(a,y)
        rb=reward(b,y)
        self.assertTrue(torch.equal(ra,-rb))
        loss=-((rb-ra).detach()*torch.sigmoid(logits)).mean()
        loss.backward()
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all()
                            for p in actor.parameters()))
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0
                            for p in actor.parameters()))

    def test_train_selection_variants_and_hidden_values(self):
        self.assertEqual(len(ALL_ARMS),5)
        for j in range(48):
            w=select_world("train",0,j)
            by={r["variant"]:r for r in w}
            self.assertEqual(set(by),set(VARIANTS))
            self.assertEqual(len({r["hash"] for r in w}),4)
            self.assertEqual(by["U_before"]["status"],by["U_after"]["status"])
            self.assertEqual(by["U_after"]["status"],by["S_before"]["status"])
            self.assertNotEqual(by["U_after"]["status"],by["S_after"]["status"])
            self.assertTrue(all(r["n"]==(5 if j%2==0 else 7) for r in w))
            self.assertEqual({r["source_mode"] for r in w},{"iid"})
            for r in w:
                f=r["features"]
                self.assertEqual(f.shape,(MAX_SLOTS,FEATURES))
                self.assertEqual(r["mask"].shape,(MAX_SLOTS,))
                self.assertEqual(r["contents"].shape,(MAX_SLOTS,))
                valid=np.flatnonzero(r["mask"]>.5)
                signed=np.array([i for i in valid if f[i,0]>.5])
                latest=signed[np.argmax(f[signed,1])]
                self.assertEqual(latest,r["winner"])
                self.assertTrue(any(f[i,0]==0 and f[i,1]>f[latest,1]
                                    for i in valid))
                self.assertEqual(int(r["contents"][r["winner"]]>0),r["status"])
            np.testing.assert_array_equal(
                by["U_before"]["contents"],by["U_after"]["contents"])
            np.testing.assert_array_equal(
                by["U_before"]["contents"],by["S_after"]["contents"])
            np.testing.assert_array_equal(
                by["U_before"]["features"][:,3],by["S_after"]["features"][:,3])

    def test_uniform_pair_exploration_repeatable_and_gradients(self):
        cases=[r for j in range(16) for r in select_world("train",0,j)]
        f,m,w=sel_tensors(cases)
        ids=np.arange(len(cases))
        a,b=uniform_distinct_pairs(m,ids,0,0)
        aa,bb=uniform_distinct_pairs(m,ids,0,0)
        self.assertTrue(torch.equal(a,aa))
        self.assertTrue(torch.equal(b,bb))
        self.assertTrue(torch.all(a!=b))
        self.assertTrue(torch.all(m[torch.arange(len(m)),a]>.5))
        self.assertTrue(torch.all(m[torch.arange(len(m)),b]>.5))
        actor=SelectActor()
        logits=actor(f,m)
        # No F4 values are ever given to the selector.
        pi=torch.distributions.Categorical(logits=logits)
        ra,rb=reward(a,w),reward(b,w)
        loss=-.5*((ra-rb).detach()*(pi.log_prob(a)-pi.log_prob(b))).mean()
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum())>0
                            for p in actor.parameters()))

if __name__=="__main__":unittest.main()
