"""F9-C visibility and memory causal-invariance tests with authentic MiniGrid."""
import unittest

import numpy as np
from observatory.f9_external_minigrid import seed_for
from observatory.f9c_visible_cue import make_env, compare_pair, rollout, CueForkEnv, observation_only_expert


class ConstantExit:
    def predict(self, obs, **kwargs):
        return 1, None


class VisibleCueTests(unittest.TestCase):
    def test_cue_visible_but_decision_view_identical_for_paired_worlds(self):
        for size in (7, 13):
            for j in range(50):
                with self.subTest(size=size, episode=j):
                    check = compare_pair(seed_for("development",0,j), size)
                    self.assertTrue(check["initial_cue_changed"])
                    self.assertTrue(check["fork_observation_identical"])
                    self.assertTrue(check["upper_candidate_visible"])

    def test_no_current_cue_shortcut_and_never_uses_oracle_labels(self):
        seed = seed_for("development",0,0)
        pairs=[]
        for swap in (False,True):
            e=make_env(7,reverse_cue=swap)
            try:
                a,_=e.reset(seed=seed)
                b,reward,done,truncated,_=e.step(0)
                self.assertEqual(a.shape,(148,))
                self.assertEqual(b.shape,(148,))
                self.assertEqual(reward,0)
                self.assertFalse(done)
                self.assertFalse(truncated)
                self.assertTrue(np.isfinite(a).all())
                self.assertTrue(np.isfinite(b).all())
                pairs.append((a.copy(),b.copy()))
                _,value,terminated,truncated,labels=e.step(1)
                self.assertTrue(terminated)
                self.assertFalse(truncated)
                self.assertIn(value,(0.,1.))
                self.assertEqual(labels["correct"],bool(value))
            finally:
                e.close()
        self.assertFalse(np.array_equal(pairs[0][0],pairs[1][0]))
        np.testing.assert_array_equal(pairs[0][1],pairs[1][1])

    def test_observation_only_one_bit_expert_solves_all_paired_worlds(self):
        # This positive control rules out a missing clue at either stage.
        for size in (7,13):
            for j in range(40):
                seed=seed_for("development",0,j)
                original=observation_only_expert(seed,size,reverse=False)
                inverted=observation_only_expert(seed,size,reverse=True)
                self.assertEqual(original["success"],1)
                self.assertEqual(inverted["success"],1)
                self.assertNotEqual(original["remembered_cue"],inverted["remembered_cue"])
                self.assertEqual(original["visible_upper"],inverted["visible_upper"])
                self.assertNotEqual(original["chosen_side"],inverted["chosen_side"])

    def test_constant_choice_cannot_solve_both_paired_conditions(self):
        model=ConstantExit()
        for size in (7,13):
            for j in range(8):
                seed=seed_for("development",0,j)
                a=rollout(model,False,size,seed,False)
                b=rollout(model,False,size,seed,True)
                self.assertEqual(a["chosen_side"],b["chosen_side"])
                self.assertNotEqual(a["correct_side"],b["correct_side"])
                self.assertEqual(a["success"] + b["success"],1)


if __name__=="__main__":
    unittest.main()
