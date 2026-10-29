"""F9-L: forced-prefix F4 interruption and temporal probe integrity tests."""
import unittest

import numpy as np

from observatory import f9i_selective_navigation as nav
from observatory.f9d_autonomous_navigation import checked_seed
from observatory.f9k_temporal_memory_probes import stage_indices
from observatory.f9l_f4_interruption import (
    ACTIVE_MEMORY_END, CONDITIONS, cutoff_for, paired_case, build_dataset,
    assert_neutral_exact, action_scores,
)


class F9LInterruptionTests(unittest.TestCase):
    def test_cutoff_schedule_is_bounded_and_fixed(self):
        times={"post_cue":3,"halfway":7,"fork":11}
        self.assertEqual(cutoff_for("normal",times),None)
        self.assertEqual(cutoff_for("cut_post_cue",times),3)
        self.assertEqual(cutoff_for("cut_halfway",times),7)
        self.assertEqual(cutoff_for("cut_last_step",times),10)
        self.assertEqual(cutoff_for("neutral_tail",times),3)
        self.assertEqual(len(CONDITIONS),5)
        self.assertEqual(ACTIVE_MEMORY_END,212)
        with self.assertRaises(ValueError):
            cutoff_for("unknown",times)
        with self.assertRaises(AssertionError):
            cutoff_for("cut_halfway",{"post_cue":1,"halfway":4,"fork":4})

    def test_development_holdout_and_train_s13_prohibited(self):
        with self.assertRaises(ValueError):
            checked_seed("heldout",0,0)
        with self.assertRaises(ValueError):
            build_dataset(None,"context",0,7,"heldout",1,"normal")
        with self.assertRaises(ValueError):
            build_dataset(None,"context",0,13,"train",1,"normal")
        with self.assertRaises(ValueError):
            build_dataset(None,"context",0,7,"train",81,"normal")
        with self.assertRaises(ValueError):
            build_dataset(None,"context",0,13,"development",11,"normal")

    def test_natural_neutral_and_blackouts_keep_fork_view_matched(self):
        old_capacity,old_seed=nav.EVENT_CAPACITY,nav.SELECTOR_SEED
        nav.EVENT_CAPACITY,nav.SELECTOR_SEED=8,81000
        model=nav.make_model("context",0,2)
        zero=nav.make_model("zero",0,2)
        try:
            for size in (7,13):
                seed=checked_seed("development",0,0)
                normal=paired_case(model,"context",0,size,seed,"normal")
                neutral=paired_case(model,"context",0,size,seed,"neutral_tail")
                assert_neutral_exact([normal],[neutral])
                self.assertTrue(normal["raw_equal"])
                self.assertTrue(normal["f4_equal"])
                self.assertEqual(normal["masked_steps"],0)
                self.assertEqual(neutral["effective_steps"],0)
                for mode in ("cut_post_cue","cut_halfway","cut_last_step"):
                    with self.subTest(size=size,condition=mode):
                        ablated=paired_case(model,"context",0,size,seed,mode)
                        self.assertTrue(ablated["raw_equal"])
                        self.assertTrue(ablated["f4_equal"])
                        self.assertGreater(ablated["masked_steps"],0)
                        self.assertGreater(ablated["effective_steps"],0)
                        self.assertTrue(np.array_equal(
                            ablated["a"]["M"], normal["a"]["M"]))
                        self.assertTrue(np.array_equal(
                            ablated["b"]["M"], normal["b"]["M"]))
                        self.assertTrue(np.array_equal(
                            ablated["a"]["O"], normal["a"]["O"]))
                        self.assertTrue(np.array_equal(
                            ablated["b"]["O"], normal["b"]["O"]))
                empty=paired_case(zero,"zero",0,size,seed,"normal")
                self.assertTrue(np.all(empty["a"]["M"]==0))
                with self.assertRaises(ValueError):
                    paired_case(zero,"zero",0,size,seed,"cut_post_cue")
                metric=action_scores([normal])
                self.assertEqual(metric["pairs"],1)
                self.assertIn(metric["both_teacher_actions_correct"],(0,1))
        finally:
            model.get_env().close()
            zero.get_env().close()
            nav.EVENT_CAPACITY,nav.SELECTOR_SEED=old_capacity,old_seed


if __name__=="__main__":
    unittest.main()
