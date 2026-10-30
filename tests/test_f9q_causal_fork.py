"""F9-Q causal transfer and decoder contracts: TRAIN 750000 only, never confirm."""
import unittest

import numpy as np

from observatory.f9q_causal_fork import (
    REGIMES,INTERVENTIONS,matched_pair,history_pair,clone_state,intervene,
    continue_from_fork,analyze_pair,summarize,
)
from observatory.f9q_train_cue_probes import train_seed,fit_and_serialize
from observatory.f9q_confirm_causal_fork import confirmation_seed
from observatory.f9o_long_blackout_training import make_model


class F9QContracts(unittest.TestCase):
    def test_disjoint_train_and_prospective_confirmation(self):
        self.assertEqual(train_seed(0,0),750000)
        self.assertEqual(train_seed(3,79),780079)
        self.assertEqual(confirmation_seed(0,0),800000)
        self.assertEqual(confirmation_seed(3,39),830039)
        self.assertLess(confirmation_seed(3,39),900000)
        for invalid in ((4,0),(0,80),(-1,0)):
            with self.assertRaises(ValueError):
                train_seed(*invalid)
        for invalid in ((4,0),(0,40),(-1,0)):
            with self.assertRaises(ValueError):
                confirmation_seed(*invalid)

    def test_swap_H_C_and_zero_independently(self):
        h=np.ones((1,1,64),np.float32)
        c=np.full((1,1,64),2.,np.float32)
        other_h=np.full((1,1,64),3.,np.float32)
        other_c=np.full((1,1,64),4.,np.float32)
        expected={
            "intact":(1,2),"swap_HC":(3,4),
            "swap_H":(3,2),"swap_C":(1,4),
            "zero_HC":(0,0),"zero_H":(0,2),"zero_C":(1,0),
        }
        self.assertEqual(set(expected),set(INTERVENTIONS))
        for name,(a,b) in expected.items():
            state=intervene((h,c),(other_h,other_c),name)
            self.assertTrue(np.all(state[0]==a))
            self.assertTrue(np.all(state[1]==b))
        np.testing.assert_array_equal(h,1)
        np.testing.assert_array_equal(c,2)
        np.testing.assert_array_equal(other_h,3)
        np.testing.assert_array_equal(other_c,4)
        with self.assertRaises(ValueError):
            intervene((h,c),(other_h,other_c),"illegal")
        with self.assertRaises(AssertionError):
            clone_state((np.zeros((2,1,64)),c))

    def test_teacher_common_prefix_and_f4_zero_at_fork(self):
        model=make_model("continuous6",0)
        try:
            world=matched_pair(model,0,train_seed(0,0))
            self.assertEqual(set(REGIMES),{"history_f4_normal","history_f4_postcue_off"})
            self.assertNotEqual(world["labels"][0]["cue"],world["labels"][1]["cue"])
            self.assertNotEqual(world["labels"][0]["correct_side"],
                                world["labels"][1]["correct_side"])
            self.assertGreater(world["times"]["fork"],world["times"]["post_cue"])
            np.testing.assert_array_equal(
                world["frames"][0][world["times"]["fork"]],
                world["frames"][1][world["times"]["fork"]])
            for regime in REGIMES:
                states=history_pair(model,0,world,regime)
                np.testing.assert_array_equal(states[0]["raw"],states[1]["raw"])
                for h in states:
                    self.assertEqual(h["raw"].shape,(148,))
                    self.assertEqual(h["state"][0].shape,(1,1,64))
                    self.assertEqual(h["state"][1].shape,(1,1,64))
                    self.assertTrue(np.isfinite(h["state"][0]).all())
                if regime=="history_f4_normal":
                    self.assertEqual(states[0]["effective_prefix_masks"],0)
                else:
                    self.assertGreater(states[0]["effective_prefix_masks"],0)
                    self.assertEqual(states[0]["scheduled_prefix_masks"],
                                     states[1]["scheduled_prefix_masks"])
                for kind in INTERVENTIONS:
                    for recipient in (0,1):
                        r=continue_from_fork(model,0,world,recipient,states,kind)
                        self.assertTrue(r["F4_zero_throughout_decision"])
                        self.assertIn(r["first_action"],range(7))
                        self.assertIn(r["first_committed_side"],("upper","lower","none"))
                        self.assertEqual(r["recipient_correct_side"],
                                         world["labels"][recipient]["correct_side"])
        finally:
            model.get_env().close()

    def test_probe_labeling_by_true_object_not_original_flag(self):
        model=make_model("continuous6",0)
        try:
            w=matched_pair(model,0,train_seed(0,0))
            states=history_pair(model,0,w,"history_f4_normal")
            self.assertEqual({x["cue"] for x in w["labels"]},{"key","ball"})
            self.assertEqual(len(states),2)
            result=analyze_pair(model,0,train_seed(0,0),
                                "history_f4_postcue_off",include_all=True)
            stats=summarize([result])
            self.assertEqual(stats["pairs"],1)
            self.assertIn("causal_donor_specific_full_pair",stats["transplant"])
            self.assertIn("intact",stats["interventions"])
            self.assertIn("swap_HC",stats["interventions"])
        finally:
            model.get_env().close()


if __name__=="__main__":
    unittest.main()
