"""F9-A independent external benchmark invariants."""
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f9_external_minigrid import (
    ALGORITHMS, TASKS, seed_for, make_env, grouped_bootstrap, ensure_dependencies,
    run_development,
)


class MiniGridHarnessTests(unittest.TestCase):
    def test_partitions_do_not_overlap(self):
        train = {seed_for("train", i) for i in range(5)}
        dev = {seed_for("development", i, e) for i in range(5) for e in range(30)}
        holdout = {seed_for("heldout", i, e) for i in range(5) for e in range(30)}
        self.assertFalse(train & dev)
        self.assertFalse(train & holdout)
        self.assertFalse(dev & holdout)

    def test_no_episode_substitutes_for_training_seed(self):
        one = [{"ppo": {"success_rate": .2},
                "recurrent_ppo": {"success_rate": .8}}]
        self.assertIsNone(grouped_bootstrap(one, "recurrent_ppo", "ppo")["bootstrap_95"])

    def test_every_environment_uses_partial_compact_observations(self):
        ensure_dependencies()
        for task in TASKS:
            with self.subTest(task=task):
                env = make_env(task)
                try:
                    a, _ = env.reset(seed=120101)
                    b, _ = env.reset(seed=120101)
                    self.assertEqual(a.shape, (148,))
                    self.assertTrue((a == b).all())
                    self.assertTrue(env.observation_space.contains(a))
                    self.assertTrue((0 <= a).all() and (a <= 1).all())
                    self.assertEqual(env.action_space.n, 7)
                finally:
                    env.close()

    def test_real_short_memory_training_and_checkpoints(self):
        with TemporaryDirectory() as dirname:
            report = run_development(
                "memory", seeds=1, timesteps=128,
                eval_episodes=2, out=Path(dirname))
            self.assertEqual(report["phase"], "development")
            self.assertEqual(set(report["summary"]["policies"]), set(ALGORITHMS))
            self.assertTrue((Path(dirname) / "report.json").exists())
            self.assertTrue((Path(dirname) / "frozen_manifest.json").exists())
            for algorithm in ("ppo", "recurrent_ppo"):
                train = report["per_seed"][0][algorithm]["training"]
                self.assertGreater(train["parameters"], 0)
                self.assertTrue((Path(dirname) / "checkpoints" /
                                 train["model_path"]).exists())


if __name__ == "__main__":
    unittest.main()
