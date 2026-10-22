"""F7 causal boundary and heldout protocol tests; no PyTorch dependencies."""
import unittest

from observatory.active_information_f7 import (
    BayesianCalibration, Episode, PROBE_COST, WorldRules, act,
    bootstrap_differences, calibrated_model, choose_probe, episodes,
    episodes_with_independent_noise, evaluate, expected_inspection_value,
    expected_terminal_value, world_rules, public_view,
)


class F7ValueOfInformationTests(unittest.TestCase):
    def test_unseen_signal_has_no_expected_value(self):
        for p in (.1, .4, .5, .8):
            self.assertAlmostEqual(
                expected_inspection_value(p, 1, .5),
                expected_terminal_value(p, 1) - PROBE_COST)

    def test_perfect_signal_benefits_uncertain_high_stakes(self):
        self.assertGreater(expected_inspection_value(.6, 1, .95),
                           expected_terminal_value(.6, 1))
        self.assertLess(expected_inspection_value(.6, .2, .89),
                        expected_terminal_value(.6, .2))

    def test_model_updates_only_after_terminal_feedback(self):
        ep = Episode(1, 1, 0, 1, (1, 0))
        model = BayesianCalibration()
        before = (model.sensor_accuracy(0), model.sensor_accuracy(1))
        choose_probe("voi_adaptive", model, public_view(ep), __import__("random").Random(1))
        self.assertEqual(before, (model.sensor_accuracy(0), model.sensor_accuracy(1)))
        model.observe(ep, 0)
        self.assertGreater(model.sensor_accuracy(0), before[0])
        self.assertEqual(model.sensor_accuracy(1), before[1])

    def test_same_world_mechanism_heldout(self):
        seed = 151
        train = episodes(seed, 80)
        heldout = episodes_with_independent_noise(
            seed, 80, unlearnable=False, change_at=None)
        self.assertNotEqual(train, heldout)
        self.assertEqual(world_rules(seed).reliabilities,
                         world_rules(seed).reliabilities)

    def test_hidden_shift_swaps_only_informative_sensor(self):
        rule = world_rules(2)
        swapped = world_rules(2, swapped=True)
        self.assertEqual(rule.reliabilities[::-1], swapped.reliabilities)
        self.assertEqual(rule.cue_accuracies, swapped.cue_accuracies)

    def test_learned_voi_is_selective_in_stable_world(self):
        r = evaluate(seed=6, train_n=120, test_n=400)
        p = r["outcomes"]
        self.assertGreater(p["voi_adaptive"]["all"]["reward"],
                           p["never"]["all"]["reward"])
        self.assertLess(p["voi_adaptive"]["all"]["inspection_rate"], .99)
        self.assertGreater(p["always"]["all"]["inspection_rate"], .99)

    def test_no_learnable_sensor_should_not_help(self):
        r = evaluate(seed=7, unlearnable=True, train_n=200, test_n=400)
        p = r["outcomes"]
        self.assertLess(p["voi_adaptive"]["all"]["inspection_rate"], .10)
        self.assertGreater(p["never"]["all"]["reward"],
                           p["always"]["all"]["reward"])

    def test_deterministic_repeated_evaluation(self):
        self.assertEqual(
            evaluate(10, train_n=60, test_n=100),
            evaluate(10, train_n=60, test_n=100))


if __name__ == "__main__":
    unittest.main()
