"""Unit checks for F9-B forensic audit, independent of training results."""
import unittest

import numpy as np

from observatory.f9_external_minigrid import make_env as existing_env
from observatory.f9b_behavior_audit import build_env, get_world_labels, swap_cue, rollout, reduce_rows


class ForwardOnlyPolicy:
    def predict(self, obs, deterministic=True, **kwargs):
        # Does not consult or need hidden environmental labels.
        return 2, None


class AuditTests(unittest.TestCase):
    def test_same_7x7_partial_vector_as_f9_training_for_both_world_sizes(self):
        for size in (7,13):
            for seed in (100000, 100003):
                env=build_env(size)
                try:
                    obs, _=env.reset(seed=seed)
                    self.assertEqual(obs.shape, (148,))
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(env.action_space.n,7)
                finally:
                    env.close()
        for seed in (100000,100001,100002):
            a,b=build_env(7),existing_env("memory")
            try:
                observation,_=a.reset(seed=seed)
                original,_=b.reset(seed=seed)
                np.testing.assert_array_equal(observation,original)
            finally:
                a.close()
                b.close()

    def test_cue_swap_preserves_exit_layout_and_start_but_reverses_target(self):
        for size in (7,13):
            for seed in (100000,100001,100005):
                env=build_env(size)
                try:
                    observation,_=env.reset(seed=seed)
                    old=get_world_labels(env)
                    changed,new=swap_cue(env,old)
                    self.assertEqual(changed.shape,observation.shape)
                    self.assertNotEqual(old["cue"],new["cue"])
                    self.assertNotEqual(old["correct_side"],new["correct_side"])
                    self.assertEqual(old["success_position"],new["failure_position"])
                    self.assertEqual(old["failure_position"],new["success_position"])
                    self.assertEqual(old["upper_object"],new["upper_object"])
                    self.assertEqual(old["lower_object"],new["lower_object"])
                    self.assertEqual(old["start_position"],new["start_position"])
                    self.assertEqual(old["start_direction"],new["start_direction"])
                    self.assertEqual(env.unwrapped.grid.get(1,size//2-1).type,new["cue"])
                finally:
                    env.close()

    def test_action_trace_and_reward_swapped_are_reported_without_inference(self):
        rows={}
        for policy in ("100k:ppo","300k:ppo"):
            original=[]
            changed=[]
            for seed in (100000,100001):
                a=rollout(ForwardOnlyPolicy(),False,7,seed,False)
                b=rollout(ForwardOnlyPolicy(),False,7,seed,True)
                self.assertEqual(a["starting_position"],b["starting_position"])
                self.assertEqual(a["upper_object"],b["upper_object"])
                self.assertEqual(a["lower_object"],b["lower_object"])
                self.assertEqual(a["trajectory"][0]["action"],2)
                self.assertEqual(b["trajectory"][0]["action"],2)
                self.assertEqual(a["correct_side"]!=b["correct_side"],True)
                original.append(a)
                changed.append(b)
            rows[policy]={"original":original,"cue_swapped":changed}
        summary=reduce_rows(rows)
        self.assertEqual(summary["100k:ppo"]["episodes"],2)
        self.assertEqual(summary["100k:ppo"]["first_action_changed_by_swap"],0)
        self.assertEqual(summary["100k:ppo"]["same_full_action_sequence_after_swap"],2)


if __name__=="__main__":
    unittest.main()
