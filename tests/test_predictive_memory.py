"""F5: no future-information path, matched budget, holdout, null and drift."""
import unittest

import numpy as np

from observatory.predictive_memory import (
    AdmissionController, world_weights, make_data, fit_predictor,
    evaluate_one, run, N_CONTEXTS, N_EVENTS, MEMORY_SLOTS,
)


class F5Tests(unittest.TestCase):
    def test_independent_contextual_drivers(self):
        w = world_weights(123)
        self.assertEqual(w.shape, (N_CONTEXTS, N_EVENTS))
        selected = [set(np.where(np.abs(row) > .8)[0]) for row in w]
        self.assertTrue(all(len(s) == 3 for s in selected))
        for i in range(N_CONTEXTS):
            for j in range(i + 1, N_CONTEXTS):
                self.assertFalse(selected[i] & selected[j])

    def test_training_only_uses_available_labels(self):
        w = world_weights(44)
        train = make_data(np.random.default_rng(44), w, 480)
        learned, pooled = fit_predictor(train)
        self.assertEqual(learned.shape, w.shape)
        self.assertLess(np.linalg.norm(learned - w) / np.linalg.norm(w), .12)
        self.assertEqual(pooled.shape, (N_EVENTS,))

    def test_controller_never_needs_future_or_target(self):
        w = world_weights(8)
        policy = AdmissionController(
            "future_lambda1", 0, w, np.mean(w, axis=0),
            np.random.default_rng(21))
        # The API only accepts the current event. No next state/answer.
        for j in range(N_EVENTS):
            policy.observe(j, 1 if j % 2 else -1)
            self.assertLessEqual(len(policy.remembered()), MEMORY_SLOTS)
        self.assertEqual(len(policy.remembered()), MEMORY_SLOTS)
        self.assertEqual(set(policy.remembered()),
                         set(np.argsort(abs(w[0]))[-3:]))

    def test_identical_scenario_is_reproducible(self):
        a = evaluate_one(201, 160, 40)
        b = evaluate_one(201, 160, 40)
        self.assertEqual(a, b)

    def test_stable_signal_and_negatives(self):
        report = run(seeds=6, train_n=320, test_n=90)
        stable = report["conditions"]["stable"]["policies"]
        self.assertLess(stable["future_lambda1"]["mse"],
                        stable["fifo"]["mse"])
        self.assertLess(stable["future_lambda1"]["mse"],
                        stable["global_blind"]["mse"])
        shifted = report["conditions"]["shift"]["policies"]
        self.assertGreater(shifted["future_lambda1"]["mse"],
                           stable["future_lambda1"]["mse"])
        null = report["conditions"]["null"]["policies"]
        self.assertGreater(null["future_lambda1"]["mse"], 1.0)
        self.assertTrue(0 <= null["future_lambda1"]["sign_accuracy"] <= 1)


if __name__ == "__main__":
    unittest.main()
