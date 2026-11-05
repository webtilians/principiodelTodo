"""F11-A v3: blind TRAIN-heldout verification contracts (never DEV/CONFIRM)."""
import gzip
import hashlib
import inspect
import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f11a_train_v2 import World, episode
from observatory.f11a_verify_v3 import (
    ACTIONS, FIT_WORLDS, HOLDOUT_WORLDS, RECALL_K, BalancedEpisodic,
    TemporalPlanner, BoundedOptions, synthetic_bytes, train, train_seed,
)


class F11AV3VerificationContracts(unittest.TestCase):
    def test_disjoint_fit_and_heldout_seeds_are_in_train_only(self):
        self.assertEqual(FIT_WORLDS, 384)
        self.assertEqual(HOLDOUT_WORLDS, 128)
        self.assertEqual(FIT_WORLDS + HOLDOUT_WORLDS, 512)
        for replica in range(4):
            fit = {train_seed(replica, i) for i in range(FIT_WORLDS)}
            held = {train_seed(replica, i) for i in range(FIT_WORLDS, 512)}
            self.assertFalse(fit & held)
            self.assertEqual(len(fit), 384)
            self.assertEqual(len(held), 128)
            self.assertTrue(all(x < 61_200_000 for x in fit | held))

    def test_a1_raw_episodic_retrieval_has_no_q_policy(self):
        bank = BalancedEpisodic()
        world = World(123, 0, 0, 0, 2, 0.0)
        script = iter(("A", "B", "E", "E", "D"))
        record, _ = episode(world, lambda obs, rng: next(script), 9, 0)
        bank.observe(record)
        self.assertFalse(hasattr(bank, "q"))
        self.assertFalse(hasattr(bank, "by_state"))
        self.assertIs(bank.episodes[0][0], record)
        self.assertEqual(bank.choose(world.reset(0).observe(), random.Random(1)), "A")
        self.assertEqual(bank.snapshot()["retrieval_K"], RECALL_K)
        self.assertEqual(bank.candidates_examined, 1)

    def test_a2_deduces_terminal_and_delayed_features_from_observed_events(self):
        model = TemporalPlanner()
        world = World(123, 0, 0, 0, 2, 0.0)
        # A/B are NOT supplied as charge/gather to the learner.
        for _ in range(3):
            script = iter(("A", "B", "E", "E", "D"))
            record, _ = episode(world, lambda obs, rng: next(script), 9, 0)
            model.observe(record)
        self.assertEqual(model.terminals[0]["D"], 3)
        self.assertGreater(model.effects[(0, "A")][2], 0)
        self.assertGreater(model.effects[(0, "B")][4], 0)
        self.assertIn(model.choose(world.reset(0).observe(), random.Random(1)), ACTIONS)
        self.assertGreater(model.attributions, 0)
        self.assertNotIn("world.roles", inspect.getsource(TemporalPlanner))
        self.assertNotIn("Simulator.pending", inspect.getsource(TemporalPlanner))

    def test_learner_role_labels_not_read_or_transmitted(self):
        source = inspect.getsource(BalancedEpisodic) + inspect.getsource(BoundedOptions)
        self.assertNotIn("world.roles", source)
        model = TemporalPlanner()
        self.assertNotIn("roles", model.snapshot())
        self.assertNotIn("phase", model.snapshot())

    def test_bounded_options_never_store_unbounded_policy_traces_per_state(self):
        options = BoundedOptions()
        start = (0, 0, 0, 0, 0, 0)
        for i in range(6):
            action = ACTIONS[i]
            steps = [{"before": start, "action": action, "after": start,
                      "reward": 1.0, "done": True}]
            options.observe(steps)
        self.assertEqual(len(options.support), 4)
        self.assertEqual(len(options.by_start[start]), 4)

    def test_reproducible_neutral_exploration_and_frozen_heldout(self):
        x = train(1, fit_worlds=4, holdout_worlds=3, episodes=8)
        y = train(1, fit_worlds=4, holdout_worlds=3, episodes=8)
        self.assertEqual(x, y)
        self.assertEqual(x["fit_episodes"], 32)
        self.assertEqual(x["fit_episodic_entries"], 32)
        self.assertEqual(x["neutral_collector_fraction"], 1.0)
        self.assertEqual(x["actor_collector_fraction"], 0.0)
        self.assertTrue(x["worlds_disjoint"])
        self.assertFalse(x["dev_opened"])
        self.assertFalse(x["confirm_opened"])
        self.assertEqual(set(x["train_heldout_results"]), {"0", "1"})
        self.assertEqual({z["episodes"] for byarm in x["train_heldout_results"].values()
                          for z in byarm.values()}, {3})
        self.assertEqual(set(x["train_heldout_results"]["0"]),
                         {"A0", "A1", "A2", "A3", "A3_ablated", "A4"})

    def test_fit_stream_hash_and_independent_audit_rows(self):
        with TemporaryDirectory() as folder:
            out = Path(folder)
            report = train(0, fit_worlds=3, holdout_worlds=2, episodes=6, out=out)
            with gzip.open(out / "fit_events.jsonl.gz", "rt", encoding="utf-8") as f:
                raw = f.read()
            self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(),
                             report["fit_event_sha256"])
            fitted = [json.loads(row) for row in raw.splitlines()]
            self.assertEqual(len(fitted), 18)
            self.assertTrue(all(row["source"] == "independent_random" for row in fitted))
            self.assertTrue(all(row["split"] == "TRAIN_FIT" for row in fitted))
            self.assertTrue(all(row["world_seed"] in [train_seed(0, n) for n in range(3)]
                                for row in fitted))
            audited = [json.loads(x) for x in (out / "heldout_results.jsonl").read_text().splitlines()]
            self.assertEqual(len(audited), 2 * 2 * 6)
            self.assertTrue(all(a["world_seed"] in [train_seed(0, n) for n in (384, 385)]
                                for a in audited))
            self.assertFalse({a["world_seed"] for a in audited} & {x["world_seed"] for x in fitted})
            checkpoint = json.loads((out / "fit_checkpoint.json").read_text())
            self.assertTrue(checkpoint["frozen_before_heldout"])
            self.assertEqual(checkpoint["fit_event_sha256"], report["fit_event_sha256"])

    def test_work_is_forbidden_outside_preregistered_fit_and_holdout(self):
        for kwargs in (
            {"replica": 4},
            {"replica": 0, "fit_worlds": 385},
            {"replica": 0, "holdout_worlds": 129},
            {"replica": 0, "episodes": 31},
        ):
            with self.assertRaises(ValueError):
                train(**kwargs)

    def test_memory_ratio_is_measured_not_assumed(self):
        r = train(0, fit_worlds=2, holdout_worlds=2, episodes=4)
        self.assertGreater(r["approx_a2_bytes"], 0)
        self.assertGreaterEqual(r["approx_a3_bytes_including_a2"], r["approx_a2_bytes"])
        self.assertGreaterEqual(r["approx_a3_to_a2_memory_ratio"], 1.0)


if __name__ == "__main__":
    unittest.main()
