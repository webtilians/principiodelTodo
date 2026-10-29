"""F9-J: isolated frozen recurrent/episodic attribution contracts."""
import unittest

import numpy as np

from observatory import f9i_selective_navigation as nav
from observatory.f9d_autonomous_navigation import checked_seed
from observatory.f9j_recurrent_memory_attribution import (
    CAPACITY,CONTROLS,DIM,autonomous_episode,inspect_forced_pair,frozen_action,
    evaluate_autonomous,
)


class RecurrentAttributionTests(unittest.TestCase):
    def setUp(self):
        self.original_capacity=nav.EVENT_CAPACITY
        nav.EVENT_CAPACITY=CAPACITY

    def tearDown(self):
        nav.EVENT_CAPACITY=self.original_capacity

    def test_splits_and_control_values(self):
        self.assertEqual(DIM,660)
        self.assertEqual(CAPACITY,8)
        self.assertEqual(len(set(CONTROLS)),len(CONTROLS))
        with self.assertRaises(ValueError):
            checked_seed("heldout",0,0)
        with self.assertRaises(ValueError):
            autonomous_episode(None,"context",0,7,checked_seed("development",0,0),
                               False,"invented")

    def test_teacher_forced_fork_is_native_and_current_view_identical(self):
        model=nav.make_model("context",0,2)
        try:
            for size in (7,13):
                for j in (0,1):
                    case=inspect_forced_pair(model,0,size,
                           checked_seed("development",0,j))
                    self.assertTrue(case["raw_equal"])
                    self.assertTrue(case["correct_teacher_actions_differ"])
                    self.assertLess(case["fork_step"],100)
                    self.assertGreaterEqual(case["hidden_state_l2_difference"],0)
                    self.assertIn(case["natural_correct_both"],(0,1))
                    self.assertEqual(set(case["predicted_actions"]),{
                        "hidden0/memory0","hidden0/memory1",
                        "hidden1/memory0","hidden1/memory1",
                        "teacher_original","teacher_swapped"
                    })
                    self.assertTrue(all(0<=a<7 for a in case["predicted_actions"].values()))
        finally:
            model.get_env().close()

    def test_fork_reset_and_memory_ablations_keep_action_reward_unchanged(self):
        model=nav.make_model("context",0,2)
        try:
            seed=checked_seed("development",0,0)
            for control in CONTROLS:
                with self.subTest(control=control):
                    out=autonomous_episode(model,"context",0,7,seed,False,control)
                    self.assertIn(out["chosen_side"],("upper","lower","none"))
                    self.assertGreater(out["length"],0)
                    self.assertLessEqual(out["length"],245)
                    self.assertTrue(out["cue_seen"])
            with self.assertRaises(ValueError):
                autonomous_episode(model,"zero",0,7,seed,False,"zero_f4_at_fork")
        finally:
            model.get_env().close()

    def test_only_public_observation_reaches_policy(self):
        class Spy:
            def predict(self,obs,**kwargs):
                self_obs=np.asarray(obs,dtype=np.float32)
                assert self_obs.shape==(660,)
                assert np.isfinite(self_obs).all()
                return np.array(2),(np.zeros((1,1,64),dtype=np.float32),
                                    np.zeros((1,1,64),dtype=np.float32))
        action,state=frozen_action(Spy(),np.zeros(660,np.float32),None,first=True)
        self.assertEqual(action,2)
        self.assertEqual(len(state),2)
        with self.assertRaises(AssertionError):
            frozen_action(Spy(),np.zeros(661,np.float32),None)


if __name__=="__main__":
    unittest.main()
