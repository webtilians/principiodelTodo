"""F10-M PRE-FLIGHT tests: only TRAIN; never DEV/OOD/CONFIRM accessed."""
import unittest
import numpy as np
import torch

from observatory.f10m_worlds import (
    OPS,PROGRAMS,VARIANTS,program_features,seed_for,dataset,
    context,evaluate_program,TRAIN_BASE,DEV_BASE,OOD_BASE
)
from observatory.f10m_experiment import (
    GlobalBandit,ContextMLP,Factorized,make_actor,
    ARMS,EPS,behavior_distribution
)

class F10MPreflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_sealed_splits_and_unseen_combination(self):
        self.assertEqual(seed_for("train",0,0,("PUBLIC",0)),25100000)
        self.assertEqual(seed_for("train",0,0,("PUBLIC",1)),25101000)
        self.assertEqual(seed_for("train",0,0,("MEMORY",0)),25102000)
        self.assertEqual(seed_for("dev",0,0,("MEMORY",0)),25202000)
        self.assertEqual(seed_for("ood",0,0),25400000)
        for args in (
            ("confirm",0,0,None),("train",4,0,("PUBLIC",0)),
            ("train",0,256,("PUBLIC",0)),
            ("dev",0,64,("PUBLIC",1)),
            ("train",0,0,("MEMORY",1)),
            ("dev",0,0,("MEMORY",1)),
            ("ood",0,0,("PUBLIC",0)),
            ("ood",0,128,None),
        ):
            with self.assertRaises(ValueError):seed_for(*args)

    def test_catalog_is_exhaustive_unfiltered_and_has_all_operation_orders(self):
        self.assertEqual(len(PROGRAMS),85)
        self.assertEqual(len(set(PROGRAMS)),85)
        self.assertEqual(PROGRAMS[0],())
        self.assertIn(("S","R","X"),PROGRAMS)
        self.assertIn(("X","R","S"),PROGRAMS)
        self.assertIn(("R","S","X"),PROGRAMS)
        self.assertIn(("P","X"),PROGRAMS)
        self.assertIn(("S","R"),PROGRAMS)
        self.assertIn(("P",),PROGRAMS)
        self.assertIn(("X","X","X"),PROGRAMS)
        self.assertEqual(program_features().shape,(85,33))
        for seq in PROGRAMS:
            self.assertTrue(all(op in OPS for op in seq))
            self.assertLessEqual(len(seq),3)

    def test_train_only_data_and_counterfactuals_without_target_leak(self):
        cases=dataset("train",0)
        self.assertEqual(len(cases),3072)
        self.assertEqual(len({r["case_sha"] for r in cases}),3072)
        self.assertEqual(
            {(r["task"],r["flip"]) for r in cases},
            {("PUBLIC",0),("PUBLIC",1),("MEMORY",0)})
        counts={(task,flip):sum(r["task"]==task and r["flip"]==flip
                               for r in cases)
                for task,flip in (("PUBLIC",0),("PUBLIC",1),("MEMORY",0))}
        self.assertEqual(set(counts.values()),{1024})
        for task,flip in counts:
            subset=[r for r in cases if r["task"]==task and r["flip"]==flip]
            for seed in {r["seed"] for r in subset}:
                by={r["variant"]:r for r in subset if r["seed"]==seed}
                self.assertEqual(set(by),set(VARIANTS))
                if task=="MEMORY":
                    self.assertEqual(by["U_before"]["truth"],
                                     by["U_after"]["truth"])
                    self.assertEqual(by["U_after"]["truth"],
                                     by["S_before"]["truth"])
                    self.assertNotEqual(by["S_after"]["truth"],
                                        by["U_after"]["truth"])
                else:
                    self.assertEqual(len({r["truth"] for r in by.values()}),1)
        for row in cases[:128]:
            self.assertEqual(row["features"].shape,(12,4))
            self.assertEqual(row["contents"].shape,(12,))
            self.assertEqual(row["mask"].shape,(12,))
            x=context(row)
            self.assertEqual(x.shape,(3,))
            self.assertTrue(np.array_equal(x,np.array([
                float(row["task"]=="MEMORY"),float(row["flip"]),
                float(row["public_bit"])],np.float32)))
            self.assertNotIn("winner",x.dtype.names or ())

    def test_program_execution_causal_read_xor_and_invalid_operations(self):
        r=next(x for x in dataset("train",0) if x["task"]=="MEMORY")
        selected=r["winner"] # ONLY a unit test; actual model uses F10-L.
        lut=(1,0)
        raw=evaluate_program(r,("S","R"),selected,lut)
        inverted=evaluate_program(r,("S","R","X"),selected,lut)
        self.assertNotEqual(raw["answer"],inverted["answer"])
        self.assertEqual(inverted["answer"],1-raw["answer"])
        self.assertEqual(raw["invalid"],0)
        self.assertEqual(inverted["invalid"],0)
        cannot_read=evaluate_program(r,("R","S"),selected,lut)
        self.assertGreater(cannot_read["invalid"],0)
        self.assertEqual(cannot_read["answer"],0)
        cannot_x=evaluate_program(r,("X",),selected,lut)
        self.assertGreater(cannot_x["invalid"],0)
        public=evaluate_program(r,("P",),selected,lut)
        public_changed=evaluate_program(r,("P",),selected,lut,
                                        mode="public_randomized")
        self.assertNotEqual(public["answer"],public_changed["answer"])
        assert r["task"]=="MEMORY"
        initial_mask=r["contents"].copy()
        shuffled=evaluate_program(r,("S","R","X"),selected,lut,mode="shuffle")
        np.testing.assert_array_equal(r["contents"],initial_mask)
        self.assertTrue(shuffled["answer"] in (0,1))

    def test_controller_logit_gradients_no_gold_program(self):
        obs=torch.tensor([[0.,0.,0.],[0.,1.,1.],[1.,0.,1.]])
        self.assertEqual(EPS,.20)
        self.assertEqual(len(ARMS),4)
        torch.manual_seed(255000)
        for arm in ARMS:
            model=make_actor(arm)
            logits=model(obs)
            self.assertEqual(logits.shape,(3,85))
            pure,mixed=behavior_distribution(logits)
            self.assertTrue(torch.allclose(mixed.sum(-1),torch.ones(3),atol=1e-6))
            sampler=torch.distributions.Categorical(probs=mixed)
            a=sampler.sample()
            # Only arbitrary observed dummy *returns* are used for gradient.
            returns=torch.tensor([1.,-1.,.7])
            loss=-(returns*sampler.log_prob(a)).mean()
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()))

if __name__=="__main__":unittest.main()
