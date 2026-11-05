"""Technical contracts for F11-B meta-controller; never opens DEV/CONFIRM."""
import json
import math
import random
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f11a_train_v2 import World
from observatory.f11b_meta_gate import (
    ARMS, BASE_SEED, BASE_WORLDS, FEATURES, HELDOUT_WORLDS, META_WORLDS,
    Feedback, LogisticGate, features_for, fixed_heuristic, flip_at,
    run, world_seed,
)


class F11BMetaControlTests(unittest.TestCase):
    def test_only_f11b_train_seeds_and_fixed_fit_gate_heldout_partition(self):
        for r in range(4):
            base = {world_seed(r, i) for i in range(BASE_WORLDS)}
            gate = {world_seed(r, i) for i in range(BASE_WORLDS, BASE_WORLDS + META_WORLDS)}
            hold = {world_seed(r, i) for i in range(BASE_WORLDS + META_WORLDS, 512)}
            self.assertEqual((len(base), len(gate), len(hold)), (256, 128, 128))
            self.assertFalse(base & gate or base & hold or gate & hold)
            self.assertTrue(all(BASE_SEED <= i < 62_200_000 for i in base | gate | hold))
        for x in ((-1, 0), (4, 0), (0, -1), (0, 512)):
            with self.assertRaises(ValueError):
                world_seed(*x)

    def test_flip_varies_and_is_not_part_of_observation(self):
        worlds = [World.generate(world_seed(0, i)) for i in range(25)]
        flips = {flip_at(w) for w in worlds}
        self.assertGreater(len(flips), 4)
        self.assertTrue(all(14 <= flip_at(w) <= 24 for w in worlds))
        for w in worlds:
            self.assertEqual(w.reset(phase=0).observe(), w.reset(phase=1).observe())

    def test_selector_inputs_have_no_seed_phase_episode_counter(self):
        hist = Feedback()
        x = features_for(hist, "A2", True)
        self.assertEqual(len(x), FEATURES)
        self.assertTrue(all(-1 <= a <= 1 for a in x))
        hist.add("A2", False)
        hist.add("A3", True)
        y = features_for(hist, "A2", True)
        self.assertNotEqual(x, y)
        z = features_for(hist, "A2", True, use_history=False)
        self.assertEqual(z, x)
        self.assertEqual(len(x), 11)

    def test_logistic_learns_from_chosen_arm_only(self):
        gate = LogisticGate()
        hist = Feedback()
        x = features_for(hist, "A3", True)
        old = deepcopy(gate.weights)
        for _ in range(12):
            gate.update("A3", x, True)
        self.assertEqual(gate.weights["A2"], old["A2"])
        self.assertGreater(gate.predict("A3", x), .5)
        self.assertEqual(gate.updates, 12)

    def test_switch_after_fail_is_strong_untrained_control(self):
        history = Feedback()
        self.assertEqual(fixed_heuristic(history), "A2")
        history.add("A2", False)
        self.assertEqual(fixed_heuristic(history), "A3")
        history.add("A3", True)
        self.assertEqual(fixed_heuristic(history), "A3")
        history.add("A3", False)
        self.assertEqual(fixed_heuristic(history), "A2")
        self.assertEqual(history.switches, 1)

    def test_small_run_reproducible_except_measured_runtime(self):
        a = run(0, base_worlds=4, meta_worlds=3, heldout_worlds=2,
                base_episodes=8, sequence_episodes=8)
        b = run(0, base_worlds=4, meta_worlds=3, heldout_worlds=2,
                base_episodes=8, sequence_episodes=8)
        a.pop("evaluation_elapsed_seconds")
        b.pop("evaluation_elapsed_seconds")
        self.assertEqual(a, b)
        self.assertEqual(a["base"]["base_fit_episodes"], 32)
        self.assertEqual(a["meta"]["gate_fit_episodes"], 24)
        self.assertEqual(a["meta"]["gate_updates"], 24)
        self.assertEqual(a["meta"]["counterfactual_arm_executions_for_learning"], 0)
        self.assertEqual(a["base_fit_seed_end"], world_seed(0, 3))
        self.assertEqual(a["gate_fit_seed_start"], world_seed(0, 256))
        self.assertEqual(a["heldout_seed_start"], world_seed(0, 384))
        self.assertTrue(a["heldout_was_excluded_from_base_and_meta_fit"])
        self.assertTrue(a["global_gate_weights_frozen_before_heldout"])
        self.assertFalse(a["dev_opened"])
        self.assertFalse(a["confirm_opened"])
        for label in ARMS:
            self.assertEqual(a["train_heldout_results"][label]["episodes"], 16)

    def test_logs_validate_observed_only_and_immutable_holdout_gate(self):
        with TemporaryDirectory() as d:
            report = run(1, base_worlds=3, meta_worlds=2, heldout_worlds=2,
                         base_episodes=6, sequence_episodes=6, out=Path(d))
            lines = Path(d, "meta_events.jsonl.gz")
            import gzip
            with gzip.open(lines, "rt", encoding="utf-8") as f:
                events = [json.loads(line) for line in f]
            self.assertEqual(len(events), 12)
            self.assertTrue(all("arm_chosen_by_randomized_behavior" in x and
                                "observed_success" in x for x in events))
            self.assertFalse(any("counterfactual" in x for x in events))
            results = [json.loads(x) for x in Path(d, "train_heldout_per_episode.jsonl")
                       .read_text().splitlines()]
            self.assertEqual(len(results), 2 * 6 * len(ARMS))
            self.assertTrue(all(x["world_seed_for_audit"] in
                                {world_seed(1, 384), world_seed(1, 385)} for x in results))
            frozen = json.loads(Path(d, "frozen_gate.json").read_text())
            self.assertEqual(frozen["updates"], 12)
            self.assertTrue(report["global_gate_weights_frozen_before_heldout"])

    def test_reject_out_of_protocol(self):
        for kwargs in (
            {"replica": 4},
            {"replica": 0, "base_worlds": 257},
            {"replica": 0, "meta_worlds": 129},
            {"replica": 0, "heldout_worlds": 129},
            {"replica": 0, "base_episodes": 31},
            {"replica": 0, "sequence_episodes": 41},
        ):
            with self.assertRaises(ValueError):
                run(**kwargs)


if __name__ == "__main__":
    unittest.main()
