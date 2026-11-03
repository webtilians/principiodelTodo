"""F10-N PRE-FLIGHT — TRAIN ONLY; no DEV/OOD/CONFIRM generated."""
import unittest
import numpy as np

from observatory.f10n_worlds import (
    PROGRAMS,TRAIN_PROGRAMS,HELDOUT_PROGRAM,VARIANTS,
    dataset,seed_for,context,evaluate_program
)
from observatory.f10n_memory import (
    execute,EffectMemory,RewardMemory
)

class TestF10NProspective(unittest.TestCase):
    def test_registered_world_namespaces_and_holdout(self):
        self.assertEqual(seed_for("train",0,0,("PUBLIC",0)),26100000)
        self.assertEqual(seed_for("train",0,0,("PUBLIC",1)),26101000)
        self.assertEqual(seed_for("train",0,0,("MEMORY",0)),26102000)
        self.assertEqual(seed_for("dev",0,0,("MEMORY",0)),26202000)
        self.assertEqual(seed_for("ood",0,0),26400000)
        for args in (
            ("confirm",0,0,None),
            ("train",0,0,("MEMORY",1)),
            ("dev",0,0,("MEMORY",1)),
            ("ood",0,0,("PUBLIC",1)),
            ("ood",0,128,None),
            ("train",4,0,("PUBLIC",0)),
        ):
            with self.assertRaises(ValueError):seed_for(*args)
        self.assertEqual(len(PROGRAMS),85)
        self.assertEqual(len(TRAIN_PROGRAMS),84)
        self.assertIn(HELDOUT_PROGRAM,PROGRAMS)
        self.assertNotIn(HELDOUT_PROGRAM,TRAIN_PROGRAMS)
        self.assertEqual(HELDOUT_PROGRAM,("S","R","X"))

    def test_train_worlds_and_context_never_leak_private_f4(self):
        cases=dataset("train",0)
        self.assertEqual(len(cases),3072)
        self.assertEqual(len({r["case_sha"] for r in cases}),3072)
        self.assertEqual({(r["task"],r["flip"]) for r in cases},
                         {("PUBLIC",0),("PUBLIC",1),("MEMORY",0)})
        for source,flip in (("PUBLIC",0),("PUBLIC",1),("MEMORY",0)):
            subset=[r for r in cases if (r["task"],r["flip"])==(source,flip)]
            self.assertEqual(len(subset),1024)
            seeds={r["seed"] for r in subset}
            self.assertEqual(len(seeds),256)
            for seed in seeds:
                by={r["variant"]:r for r in subset if r["seed"]==seed}
                self.assertEqual(set(by),set(VARIANTS))
                if source=="MEMORY":
                    self.assertEqual(by["U_before"]["truth"],by["U_after"]["truth"])
                    self.assertEqual(by["U_after"]["truth"],by["S_before"]["truth"])
                    self.assertNotEqual(by["U_after"]["truth"],by["S_after"]["truth"])
                else:
                    self.assertEqual(len({x["truth"] for x in by.values()}),1)
        for r in cases[:120]:
            self.assertTrue(np.array_equal(context(r),np.array([
                float(r["task"]=="MEMORY"),float(r["flip"]),
                float(r["public_bit"])],np.float32)))
            self.assertEqual(r["contents"].shape,(12,))

    def test_environment_trace_matches_original_execution(self):
        cases=dataset("train",0)
        try_programs=(
            (),("P",),("S","R"),("P","X"),("S","P","X"),
            ("S","X","R"),("R","S","X"),("P","X","S"),
        )
        for row in cases[::97]:
            for seq in try_programs:
                for mode in ("normal","shuffle","x_disabled","public_randomized"):
                    original=evaluate_program(row,seq,row["winner"],(1,0),mode)
                    observed=execute(row,seq,row["winner"],(1,0),mode)
                    self.assertEqual(original["answer"],observed["answer"])
                    self.assertEqual(original["success"],observed["success"])
                    self.assertEqual(original["invalid"],observed["invalid"])
                    self.assertAlmostEqual(original["return"],observed["reward"])
                    self.assertEqual(original["operations"],observed["operations"])
                    for step in observed["trace"]:
                        if step["op"]!="R" or not step["before"][0]:
                            self.assertEqual(step["read_observed"],-1)
                        else:
                            self.assertIn(step["read_observed"],(0,1))

    def test_observed_effects_propose_never_executed_srx(self):
        cases=dataset("train",0)
        em=EffectMemory()
        rm=RewardMemory()
        # TRAIN-only observed examples. The causal effect of X with a slot
        # selected is learned from SPX (public input), and that of R from SR.
        # The combination SRX itself is NEVER executed.
        episodes=0
        for row in cases[::11]:
            for seq in (("P",),("P","X"),("S","R"),("S","P","X")):
                self.assertNotEqual(seq,HELDOUT_PROGRAM)
                result=execute(row,seq,row["winner"],(1,0))
                em.observe(result,row["public_bit"])
                rm.observe(row,PROGRAMS.index(seq),result)
                episodes+=1
        self.assertEqual(em.episodes,episodes)
        self.assertEqual(rm.episodes,episodes)
        for bit in (0,1):
            proposed,score=em.plan("MEMORY",1,bit)
            self.assertEqual(proposed,HELDOUT_PROGRAM)
            self.assertGreater(score,.85)
            self.assertNotIn(HELDOUT_PROGRAM,TRAIN_PROGRAMS)
        self.assertTrue(em.counts)
        self.assertTrue(rm.table)

    def test_unknown_transition_cannot_be_claimed_as_known(self):
        em=EffectMemory()
        self.assertEqual(em.outcome("X",0,-1,0,-1),(None,0))
        p,score=em.plan("MEMORY",1,0)
        self.assertEqual(p,())
        self.assertLessEqual(score,0.5)
        self.assertEqual(em.novelty(("S","R","X"),"MEMORY",0)>0,True)

if __name__=="__main__":
    unittest.main()
