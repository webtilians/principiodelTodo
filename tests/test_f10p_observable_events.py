"""F10-P preflight uses TRAIN ONLY; never opens DEV/OOD/28300000+."""
import unittest
import numpy as np

from observatory import f10p_worlds as world
from observatory.f10n_memory import execute
from observatory.f10p_events import (
    EventEffects,CHANNELS,public_events,project
)

class F10PPreflight(unittest.TestCase):
    def test_frozen_splits_holdouts(self):
        self.assertEqual(world.seed_for("train",0,0,("PUBLIC",0)),28100000)
        self.assertEqual(world.seed_for("train",0,0,("PUBLIC",1)),28101000)
        self.assertEqual(world.seed_for("train",0,0,("MEMORY",0)),28102000)
        self.assertEqual(world.seed_for("dev",0,0,("MEMORY",0)),28202000)
        self.assertEqual(world.seed_for("ood",0,0),28400000)
        with self.assertRaises(ValueError):
            world.seed_for("confirm",0,0,("MEMORY",1))
        with self.assertRaises(ValueError):
            world.seed_for("train",0,0,("MEMORY",1))
        with self.assertRaises(ValueError):
            world.dataset("confirm",0)
        self.assertEqual(len(world.PROGRAMS),85)
        self.assertEqual(len(world.TRAIN_PROGRAMS),84)
        self.assertNotIn(("S","R","X"),world.TRAIN_PROGRAMS)

    def test_public_contract_does_not_expose_private_internal_state(self):
        cases=world.dataset("train",0)
        self.assertEqual(len(cases),3072)
        self.assertEqual(len(set(r["case_sha"] for r in cases)),3072)
        for row in cases[:60]:
            for seq in (("S","R"),("P","X"),("X",),("R",),("S","P","X")):
                result=execute(row,seq,row["winner"],(1,0))
                e=public_events(result)
                self.assertEqual(len(e),len(seq))
                for z in e:
                    self.assertEqual(set(z),{"op","output"})
                    self.assertIn(z["op"],("P","S","R","X"))
                    self.assertTrue(
                        set(z["output"]) in ({"value"},{"ack"},{"error"}))
                    self.assertFalse(any(k in str(z) for k in (
                        "read_observed","before","after","winner","truth",
                        "contents","mask","invalid_delta")))
                mask=np.random.default_rng(123)
                noise=np.random.default_rng(456)
                for arm in CHANNELS:
                    p=project(e,arm,mask,noise)
                    self.assertEqual(len(p),len(e))
                    if arm=="E100":self.assertEqual(p,e)
                    if arm=="E0":
                        self.assertTrue(all(k["output"] is None for k in p))
                    if arm=="E_noX":
                        self.assertTrue(all(z["output"] is None for z in p
                                            if z["op"]=="X"))

    def test_real_train_only_examples_synthesize_never_executed_srx(self):
        cases=world.dataset("train",0)
        mem=EventEffects()
        blank=EventEffects()
        no_x=EventEffects()
        for row in cases[::5]:
            for seq in (("P",),("P","X"),("S","R"),("S","P","X")):
                assert seq!=("S","R","X")
                real=execute(row,seq,row["winner"],(1,0))
                events=public_events(real)
                mem.observe(events,row["public_bit"])
                no_x.observe(
                    project(events,"E_noX",np.random.default_rng(1),
                            np.random.default_rng(2)),row["public_bit"])
                blank.observe(
                    project(events,"E0",np.random.default_rng(1),
                            np.random.default_rng(2)),row["public_bit"])
        self.assertGreaterEqual(mem.s_ack,10)
        self.assertGreaterEqual(mem.r_valid_after_s,10)
        self.assertEqual(mem.learned(mem.x[0]),1)
        self.assertEqual(mem.learned(mem.x[1]),0)
        for bit in (0,1):
            seq,score=mem.plan("MEMORY",1,bit)
            self.assertEqual(seq,("S","R","X"))
            self.assertGreater(score,.90)
            self.assertEqual(no_x.plan("MEMORY",1,bit)[0],())
            self.assertEqual(blank.plan("MEMORY",1,bit)[0],())

if __name__=="__main__":
    unittest.main()
