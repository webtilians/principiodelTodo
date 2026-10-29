"""Integrity tests run on GitHub-hosted CPU, not the user's machines."""
import unittest
import numpy as np
from observatory.f9d_autonomous_navigation import (
    TrainingWorlds, checked_seed, demonstration, actor_action, rollout,
    make_model, clone, cue_visible)
from observatory.f9b_behavior_audit import build_env


class NavigationIntegrity(unittest.TestCase):
    def test_native_transition_equivalence_all_seven_actions(self):
        a, b = TrainingWorlds(0), build_env(7)
        try:
            x, info = a.reset()
            y, _ = b.reset(seed=checked_seed("train", 0, 0))
            np.testing.assert_array_equal(x, y)
            self.assertEqual(a.action_space.n, 7)
            self.assertEqual(info, {})
            for action in [0, 1, 2, 3, 4, 5, 6, 0, 0, 2]:
                xa, ra, da, ta, ia = a.step(action)
                xb, rb, db, tb, _ = b.step(action)
                np.testing.assert_array_equal(xa, xb)
                self.assertEqual((ra, da, ta, ia), (rb, db, tb, {}))
        finally:
            a.close()
            b.close()

    def test_teacher_native_routes_and_causal_prefix(self):
        for size in (7, 13):
            for j in range(20):
                seed = checked_seed("train", 0, j)
                x, y, seen = demonstration(seed, size)
                xx, yy, ss = demonstration(seed, size, True)
                self.assertTrue(any(seen) and any(ss))
                self.assertTrue(set(y) <= {0, 1, 2})
                # Teacher first acquires cue and arrives at same fork pose.
                # There must be equal current frames with different action labels.
                self.assertTrue(any(np.array_equal(a, b) and c != d
                                    for a, b, c, d in zip(x, xx, y, yy)))

    def test_partition_is_bounded_and_sealed(self):
        train = {checked_seed("train", i, j) for i in range(4) for j in range(5000)}
        dev = {checked_seed("development", i, j) for i in range(4) for j in range(100)}
        self.assertFalse(train & dev)
        self.assertLess(max(train), min(dev))
        with self.assertRaises(ValueError):
            checked_seed("heldout", 0, 0)
        with self.assertRaises(ValueError):
            checked_seed("train", 10, 0)

    def test_visibility_observer_restores_state(self):
        env = build_env(7)
        try:
            x, _ = env.reset(seed=1000)
            pos, direction = tuple(env.unwrapped.agent_pos), env.unwrapped.agent_dir
            cue_visible(env)
            np.testing.assert_array_equal(x, env.observation(env.unwrapped.gen_obs()))
            self.assertEqual((pos, direction),
                             (tuple(env.unwrapped.agent_pos), env.unwrapped.agent_dir))
        finally:
            env.close()

    def test_actor_boundary(self):
        class Spy:
            def predict(self, obs, **kwargs):
                self.obs, self.kwargs = obs, kwargs
                return np.array(2), None
        spy = Spy()
        actor_action(spy, np.zeros(148, np.float32), None, True, False)
        self.assertEqual(spy.obs.shape, (148,))
        self.assertEqual(spy.kwargs, {"deterministic": True})

    def test_bc_sequence_api_and_native_rl_smoke(self):
        data = [demonstration(1000 + j, reverse=r)
                for j in range(2) for r in (False, True)]
        for recurrent in (False, True):
            model = make_model(recurrent, 0, 2)
            history = clone(model, recurrent, data, 1, 0)
            self.assertTrue(np.isfinite(history).all())
            model.learn(total_timesteps=128)
            result = rollout(model, recurrent, 7, 100000)
            self.assertTrue(all(0 <= t["action"] < 7 for t in result["trace"]))
            model.get_env().close()


if __name__ == "__main__":
    unittest.main()
