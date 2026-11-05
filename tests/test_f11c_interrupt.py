"""F11-C contracts: trainable within-episode interruption without hidden oracle."""
import gzip
import hashlib
import inspect
import json
import random
import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f11a_train_v2 import World, episode
from observatory.f11a_verify_v3 import BoundedOptions, TemporalPlanner
from observatory.f11c_interrupt import (
    ARMS, FEATURES, GATE_WORLDS, BASE_WORLDS, HELDOUT_WORLDS,
    LearnedInterrupt, change_step, gate_features, run, simulate, study_seed,
)


class IntraEpisodeInterruptionContracts(unittest.TestCase):
    def test_new_training_seeds_disjoint_from_f11a_and_f11b(self):
        for r in range(4):
            a = {study_seed(r, j) for j in range(BASE_WORLDS)}
            b = {study_seed(r, j) for j in range(BASE_WORLDS, BASE_WORLDS + GATE_WORLDS)}
            c = {study_seed(r, j) for j in range(BASE_WORLDS + GATE_WORLDS, 512)}
            self.assertEqual((len(a), len(b), len(c)), (256, 128, 128))
            self.assertFalse(a & b or a & c or b & c)
            self.assertTrue(all(63_100_000 <= s < 63_200_000 for s in a | b | c))
        for args in ((4, 0), (-1, 0), (0, -1), (0, 512)):
            with self.assertRaises(ValueError):
                study_seed(*args)

    def test_flip_is_inside_episode_and_not_in_gate_inputs(self):
        worlds = [World.generate(study_seed(0, i)) for i in range(12)]
        self.assertGreater(len({change_step(w) for w in worlds}), 1)
        self.assertTrue(all(3 <= change_step(w) <= 7 for w in worlds))
        for world in worlds:
            self.assertEqual(world.reset(0).observe(), world.reset(1).observe())
        values = gate_features((0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 1),
                               -0.10, 5, 2, 7)
        self.assertEqual(len(values), FEATURES)
        self.assertFalse(any(abs(x) > 1 for x in values))
        assert "world.roles" not in inspect.getsource(LearnedInterrupt)
        assert "simulator.phase" not in inspect.getsource(LearnedInterrupt)

    def test_ablation_masks_observable_surprise_not_progress(self):
        x = gate_features((0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 1), -.12, 6, 2, 4)
        y = gate_features((0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 1),
                          -.12, 6, 2, 4, ablate=True)
        self.assertEqual((y[0], y[5], y[7]), (x[0], x[5], x[7]))
        self.assertEqual(y[1:5], (0, 0, 0, 0))
        self.assertEqual(y[6], 0)
        self.assertGreater(x[1], 0)

    def test_reinforce_updates_only_taken_actions_and_observed_reward(self):
        gate = LearnedInterrupt()
        f = gate_features((0, 0, 0, 0, 0, 0), (0, 0, 1, 0, 0, 0), -.1, 4, 1, 5)
        previous = gate.snapshot()
        gate.learn([(f, True, gate.probability(f))], observed_reward=1.0)
        self.assertNotEqual(gate.w, previous["weights"])
        self.assertEqual(gate.updates, 1)
        self.assertEqual(gate.trajectories, 1)

    def test_fixed_guard_is_original_strict_behavior_not_learned(self):
        world = World(123, 0, 0, 0, 2, 0.0)
        commands = iter(("A", "B", "E", "E", "D"))
        successful, _ = episode(world, lambda obs, rng: next(commands), 23, 0)
        options, model = BoundedOptions(), TemporalPlanner()
        options.observe(successful)
        model.observe(successful, 0)
        orig = options.snapshot()
        guard, _ = simulate(world, model, options, None, "G", "stable", 17)
        continue_control, _ = simulate(world, model, options, None, "C", "stable", 17)
        self.assertTrue(guard["had_option"])
        self.assertTrue(continue_control["had_option"])
        self.assertEqual(options.snapshot(), orig)
        self.assertGreaterEqual(guard["actions"], 1)
        self.assertEqual(continue_control["selector_decisions"], 0)

    def test_four_splits_and_frozen_heldout_small(self):
        a = run(0, base_worlds=3, gate_worlds=2, heldout_worlds=2,
                base_episodes=5, gate_episodes=4)
        b = run(0, base_worlds=3, gate_worlds=2, heldout_worlds=2,
                base_episodes=5, gate_episodes=4)
        a.pop("seconds_total_workflow")
        b.pop("seconds_total_workflow")
        self.assertEqual(a, b)
        self.assertEqual(a["base_fit"]["base_fit_episodes"], 15)
        self.assertEqual(a["gate_fit"]["episodes"], 8)
        self.assertEqual(a["base_fit_start"], study_seed(0, 0))
        self.assertEqual(a["gate_fit_start"], study_seed(0, 256))
        self.assertEqual(a["heldout_start"], study_seed(0, 384))
        self.assertEqual(a["frozen_selector"]["feature_count"] if "feature_count" in a["frozen_selector"] else len(a["frozen_selector"]["weights"]), FEATURES)
        self.assertEqual(set(a["heldout"]), {"stable", "flip"})
        for phase in a["heldout"].values():
            self.assertEqual(set(phase), set(ARMS))
            self.assertTrue(all(row["episodes"] == 2 for row in phase.values()))
        self.assertTrue(a["global_weights_frozen_during_heldout"])
        self.assertFalse(a["dev_opened"])
        self.assertFalse(a["confirm_opened"])

    def test_raw_artifacts_observations_and_separation(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            result = run(1, base_worlds=3, gate_worlds=2, heldout_worlds=2,
                         base_episodes=5, gate_episodes=4, out=out)
            with gzip.open(out / "gate_events.jsonl.gz", "rt", encoding="utf-8") as f:
                events = [json.loads(s) for s in f]
            self.assertEqual(len(events), 8)
            self.assertTrue(all(set(s) == {"before", "action", "after", "reward", "done"}
                                for e in events for s in e["steps"]))
            outcomes = [json.loads(s) for s in (out / "heldout_per_world.jsonl").read_text().splitlines()]
            self.assertEqual(len(outcomes), 2 * 2 * len(ARMS))
            self.assertTrue(all(o["world_seed_for_audit"] in
                                {study_seed(1, 384), study_seed(1, 385)} for o in outcomes))
            self.assertTrue(all(o["scenario_for_audit"] == "flip" and 3 <= o["change_step_for_audit"] <= 7
                                or o["scenario_for_audit"] == "stable" and o["change_step_for_audit"] is None
                                for o in outcomes))
            self.assertEqual(result["frozen_selector"],
                             json.loads((out / "frozen_model.json").read_text())["selector"])

    def test_reject_excess_budget(self):
        for args in ({"replica": 4}, {"replica": 0, "base_worlds": 257},
                     {"replica": 0, "gate_worlds": 129},
                     {"replica": 0, "heldout_worlds": 129},
                     {"replica": 0, "base_episodes": 31},
                     {"replica": 0, "gate_episodes": 31}):
            with self.assertRaises(ValueError):
                run(**args)


if __name__ == "__main__":
    unittest.main()
