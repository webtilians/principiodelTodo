"""Technical contracts for F11-A TRAIN; never consume DEV or CONFIRM."""
import gzip
import hashlib
import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f11a_train import (
    ACTIONS, HORIZON, TRAIN_BASE, Effects, OptionController, Options,
    World, episode, train, train_seed, won,
)


class F11ATrainContracts(unittest.TestCase):
    def test_train_seed_ranges_are_disjoint_and_bounded(self):
        for replica in range(4):
            self.assertEqual(train_seed(replica, 0), TRAIN_BASE + 10000 * replica)
            self.assertEqual(train_seed(replica, 511), TRAIN_BASE + 10000 * replica + 511)
            self.assertLess(train_seed(replica, 511), 61_200_000)
        with self.assertRaises(ValueError):
            train_seed(4, 0)
        with self.assertRaises(ValueError):
            train_seed(0, 512)

    def test_agent_public_observations_do_not_contain_world_seed_or_roles(self):
        world = World.generate(train_seed(0, 0))
        obs = world.reset()
        self.assertEqual(len(obs), 4)
        self.assertEqual(obs[:2], (world.mode, world.goal))
        self.assertNotIn(world.seed, obs)
        self.assertTrue(all(isinstance(x, int) for x in obs))
        with self.assertRaises(ValueError):
            world.step(obs, "UNAUTHORIZED", random.Random(1))

    def test_success_induces_guarded_option_without_expert_actions(self):
        world = World(seed=123, mode=0, goal=0, initial_route=0, slip=0.0)
        script = iter(("A", "D"))
        trace = episode(world, lambda obs, rng: next(script), 42)
        self.assertTrue(won(trace))
        options, effects = Options(), Effects()
        options.observe(trace)
        effects.observe(trace)
        controller = OptionController(options, effects)
        rng = random.Random(2)
        self.assertEqual(controller.choose(world.reset(), rng), "A")
        next_obs, _, _ = world.step(world.reset(), "A", rng)
        self.assertEqual(controller.choose(next_obs, rng), "D")
        self.assertTrue(controller.reused)
        # The learned procedure is conditional on observed intermediate state.
        mismatched = (0, 0, 0, 1)
        controller.choose(mismatched, rng)
        self.assertTrue(controller.aborted)
        self.assertEqual(len(options.support), 1)

    def test_unsuccessful_experience_creates_no_option(self):
        world = World(seed=4, mode=0, goal=0, initial_route=0, slip=0.0)
        trace = episode(world, lambda obs, rng: "C", 10)
        self.assertFalse(won(trace))
        options = Options()
        options.observe(trace)
        self.assertIsNone(options.start(world.reset()))

    def test_small_train_is_reproducible_and_train_only(self):
        one = train(1, worlds=8, episodes=5, probes=1)
        two = train(1, worlds=8, episodes=5, probes=1)
        self.assertEqual(one, two)
        self.assertEqual(one["training_episodes"], 40)
        self.assertEqual(one["training_actions"], one["effects_observed_steps"])
        self.assertTrue(one["probe_worlds_are_training_worlds"])
        self.assertFalse(one["dev_opened"])
        self.assertFalse(one["confirm_opened"])
        self.assertEqual(set(one["probe_results"]), {"A0", "A1", "A2", "A3", "A4"})
        self.assertEqual({x["episodes"] for x in one["probe_results"].values()}, {8})

    def test_logged_shared_events_match_report_and_checkpoints(self):
        with TemporaryDirectory() as tmp:
            out = Path(tmp)
            summary = train(0, worlds=6, episodes=4, probes=0, out=out)
            with gzip.open(out / "train_events.jsonl.gz", "rt", encoding="utf-8") as f:
                text = f.read()
            self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                             summary["shared_event_sha256"])
            rows = [json.loads(line) for line in text.splitlines()]
            self.assertEqual(len(rows), 24)
            self.assertEqual({row["split"] for row in rows}, {"TRAIN"})
            self.assertTrue(all(TRAIN_BASE <= row["world_seed"] < 61_200_000
                                for row in rows))
            checkpoint = json.loads((out / "checkpoint.json").read_text())
            self.assertEqual(checkpoint["shared_event_sha256"], summary["shared_event_sha256"])
            self.assertEqual(checkpoint["status"], "TRAIN_ONLY")

    def test_limits_stop_unregistered_training(self):
        for kwargs in ({"replica": 4}, {"replica": 0, "worlds": 513},
                       {"replica": 0, "episodes": 31}, {"replica": 0, "probes": 5}):
            with self.assertRaises(ValueError):
                train(**kwargs)


if __name__ == "__main__":
    unittest.main()
