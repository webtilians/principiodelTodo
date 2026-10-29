import unittest
import numpy as np
from minigrid.core.constants import OBJECT_TO_IDX
from observatory.selective_memory import Store, FIFO
from observatory.f9b_behavior_audit import build_env
from observatory.f9g_context_memory import ContextF4, ContextObservation, DIM
from observatory.f9g_context_gate import paired


def observation(kind="key", x=1, y=2):
    raw = np.zeros(148, np.float32)
    raw[:147].reshape(7, 7, 3)[x, y, 0] = OBJECT_TO_IDX[kind]/max(OBJECT_TO_IDX.values())
    return raw


class ContextIntegrity(unittest.TestCase):
    def test_context_distinguishes_position_order_and_age(self):
        a, b = ContextF4(), ContextF4()
        self.assertFalse(np.array_equal(a.encode(observation(x=1))[148:],
                                        b.encode(observation(x=2))[148:]))
        a.reset(); b.reset()
        for kind in ("key", "ball"):
            a.encode(observation(kind))
        for kind in ("ball", "key"):
            b.encode(observation(kind))
        self.assertFalse(np.array_equal(a.readout(), b.readout()))
        before = a.readout().reshape(-1, 8).copy()
        a.encode(np.zeros(148, np.float32))
        after = a.readout().reshape(-1, 8)
        self.assertGreater(after[0, 5], before[0, 5])
        np.testing.assert_array_equal(after[:, :5], before[:, :5])

    def test_real_fifo_bounded_and_eventually_forgets(self):
        a, b = ContextF4(capacity=3), ContextF4(capacity=3)
        self.assertIsInstance(a.store, Store)
        self.assertIsInstance(a.selector, FIFO)
        a.encode(observation("key")); b.encode(observation("ball"))
        for _ in range(3):
            a.encode(observation()); b.encode(observation())
        self.assertEqual(a.store.order, [1, 2, 3])
        self.assertEqual(len(a.store.saved), 3)
        np.testing.assert_array_equal(a.readout(), b.readout())

    def test_reset_all_state_and_replay_no_hidden_input(self):
        for mode in ("f4", "zero", "erase", "swap"):
            a, fresh = ContextF4(mode), ContextF4(mode)
            for _ in range(70):
                a.encode(observation())
            a.reset()
            self.assertEqual((a.tick, a.serial, a.store.order), (-1, 0, []))
            self.assertFalse(a.readout().any())
            for raw in (observation("ball"), observation("key"), np.zeros(148)):
                np.testing.assert_array_equal(a.encode(raw), fresh.encode(raw))
        with self.assertRaises(ValueError):
            ContextF4().encode(np.zeros(149))
        with self.assertRaises(ValueError):
            ContextF4("oracle")

    def test_native_all_seven_actions_rewards_and_wrapper_reset(self):
        for size in (7, 13):
            for mode in ("f4", "zero", "erase", "swap"):
                a, b = ContextObservation(build_env(size), mode), build_env(size)
                try:
                    self.assertEqual(a.action_space.n, 7)
                    for action in range(7):
                        x, _ = a.reset(seed=100000)
                        y, _ = b.reset(seed=100000)
                        self.assertEqual(x.shape, (DIM,))
                        np.testing.assert_array_equal(x[:148], y)
                        xa, ra, da, ta, ia = a.step(action)
                        xb, rb, db, tb, ib = b.step(action)
                        np.testing.assert_array_equal(xa[:148], xb)
                        self.assertEqual((ra, da, ta, ia), (rb, db, tb, ib))
                        self.assertTrue(a.observation_space.contains(xa))
                    a.reset(seed=100001)
                    fresh = ContextF4(mode)
                    raw, _ = b.reset(seed=100001)
                    np.testing.assert_array_equal(a.writer.readout(), fresh.encode(raw)[148:])
                finally:
                    a.close(); b.close()

    def test_paired_native_s7_s13(self):
        for size in (7, 13):
            result = paired(size, 100000)
            self.assertTrue(result["gate_pass"])
            self.assertTrue(result["both_objects_seen"])


if __name__ == "__main__":
    unittest.main()
