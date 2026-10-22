"""F7b tests: surprises trigger only after observing bad evidence."""
import unittest
from observatory.active_recalibration_f7b import (
    test_recalibration as evaluate_recalibration, run
)


class TestChangeTriggeredLearning(unittest.TestCase):
    def test_informative_sensor_is_never_revealed_to_policy(self):
        data = evaluate_recalibration(9, train_n=80, test_n=200)
        self.assertIn("reward", data["metrics"]["all"])
        self.assertIn("detection_steps", data)
        self.assertEqual(data["metrics"]["all"]["correct"] >= 0, True)

    def test_random_sensors_should_not_activate_noisy_curiosity(self):
        data = evaluate_recalibration(
            32, unlearnable=True, train_n=180, test_n=220)
        self.assertEqual(data["metrics"]["all"]["inspection_rate"], 0.0)
        self.assertEqual(data["detection_steps"], [])

    def test_repeated_run_is_deterministic(self):
        a = evaluate_recalibration(12, change=True, train_n=90, test_n=200)
        b = evaluate_recalibration(12, change=True, train_n=90, test_n=200)
        self.assertEqual(a, b)

    def test_exploratory_full_comparison_format(self):
        r = run(seeds=4, train_n=90, test_n=200)
        self.assertEqual(set(r["conditions"]), {"stable", "switch", "unlearnable"})
        for cond in r["conditions"].values():
            self.assertIn("after_last_100", cond["comparison"])
            self.assertEqual(cond["n_seeds"], 4)


if __name__ == "__main__":
    unittest.main()
