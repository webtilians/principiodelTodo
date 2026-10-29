"""F9-B: real F4 Store/FIFO, controlled ablations and holdout restrictions."""
import json
import tempfile
import unittest
from pathlib import Path

from observatory.f9b_causal_memory import (
    MIN_HELDOUT_EPISODES, VARIANTS, ablations, baselines, competence, heldout,
    make_env,
)


class CausalMemoryTests(unittest.TestCase):
    def test_f4_uses_only_partial_visible_observation_and_zero_control(self):
        for mode in ("none", "f4", "zero", "corrupt"):
            with self.subTest(mode=mode):
                env = make_env(mode)
                try:
                    obs, _ = env.reset(seed=10001)
                    self.assertTrue(env.observation_space.contains(obs))
                    self.assertEqual(obs.shape, (148 if mode == "none" else 160,))
                    if mode == "zero":
                        self.assertTrue((obs[148:] == 0).all())
                    for _ in range(4):
                        obs, reward, terminated, truncated, _ = env.step(2)
                        self.assertTrue(env.observation_space.contains(obs))
                        if terminated or truncated:
                            break
                finally:
                    env.close()

    def test_f4_past_snapshot_is_different_from_zero_and_corrupted(self):
        all_history = {}
        for mode in ("f4", "zero", "corrupt"):
            env = make_env(mode)
            try:
                history = []
                obs, _ = env.reset(seed=10013)
                history.append(obs)
                for _ in range(8):
                    obs, _, done, truncated, _ = env.step(2)
                    history.append(obs)
                    if done or truncated:
                        break
                all_history[mode] = history
            finally:
                env.close()
        self.assertEqual(len(all_history["f4"]), len(all_history["zero"]))
        # Every current observation is exactly identical for all interventions.
        for snapshots in zip(*all_history.values()):
            self.assertTrue((snapshots[0][:148] == snapshots[1][:148]).all())
            self.assertTrue((snapshots[0][:148] == snapshots[2][:148]).all())
        # Once the object is seen, the persistent snapshot is different.
        self.assertTrue(any((x[148:] != 0).any() for x in all_history["f4"]))
        self.assertTrue(any((x[148:] != y[148:]).any()
                            for x, y in zip(all_history["f4"], all_history["corrupt"])))

    def test_gate_rejects_untrained_reference(self):
        ready, evidence = competence([{
            "lstm": {"success_rate": 0.0}, "random": {"success_rate": 0.0}
        }])
        self.assertFalse(ready)
        self.assertEqual(evidence["training_seeds"], 1)

    def test_real_short_train_six_conditions_without_opening_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "baselines"
            study = root / "ablations"
            baseline_report = baselines(1, 128, 2, raw)
            self.assertFalse(baseline_report["gate"]["ready"])
            with self.assertRaises(ValueError):
                ablations(raw, root / "rejected")
            causal = ablations(raw, study, smoke=True)
            self.assertFalse(causal["eligible_for_heldout"])
            self.assertEqual(set(causal["summary"]["policies"]), set(VARIANTS))
            for index in range(1):
                self.assertEqual(
                    causal["per_seed"][index]["lstm_f4"]["episodes"][0]["episode_seed"],
                    causal["per_seed"][index]["lstm_f4_corrupt"]["episodes"][0]["episode_seed"])
            manifest = json.loads((study / "frozen_manifest.json").read_text())
            self.assertEqual(len(manifest["models"]), 5)
            self.assertEqual(len(list((study / "checkpoints").glob("*.zip"))), 5)
            with self.assertRaises(ValueError):
                heldout(study, root / "heldout", MIN_HELDOUT_EPISODES)
            self.assertFalse((study / ".heldout_opened").exists())


if __name__ == "__main__":
    unittest.main()
