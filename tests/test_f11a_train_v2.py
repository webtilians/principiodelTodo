"""F11-A TRAIN v2 technical contracts; DEV/CONFIRM are never sampled."""
import gzip
import hashlib
import json
import random
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from observatory.f11a_train_v2 import (
    ACTIONS, CHANGE_AT, HORIZON, RECALL_K, TRAIN_BASE,
    World, Episodic, Effects, Options, OptionController,
    episode, success, train, train_seed,
)


class F11ATrainV2Contracts(unittest.TestCase):
    def test_reservation_and_bounds(self):
        starts = [train_seed(r, 0) for r in range(4)]
        self.assertEqual(starts, [61_100_000, 61_110_000, 61_120_000, 61_130_000])
        self.assertTrue(all(train_seed(r, 511) < 61_200_000 for r in range(4)))
        for replica, index in ((4, 0), (-1, 0), (0, 512), (0, -1)):
            with self.assertRaises(ValueError):
                train_seed(replica, index)

    def test_delay_requires_additional_actions(self):
        world = World(123, 0, 0, 0, 2, 0.0)
        simulator = world.reset(0)
        self.assertEqual(simulator.observe(), (0, 0, 0, 0, 0, 0))
        x, _, _ = simulator.step("A", random.Random(10))
        self.assertEqual(x[2], 0)
        self.assertEqual(x[-1], 1)
        x, _, _ = simulator.step("E", random.Random(10))
        self.assertEqual(x[2], 0)
        x, _, _ = simulator.step("E", random.Random(10))
        self.assertEqual(x[2], 1)
        self.assertEqual(x[-1], 0)
        self.assertEqual(simulator.completed_effects, 1)

    def test_same_public_input_changes_action_meaning_without_phase_tag(self):
        world = World(123, 0, 0, 0, 1, 0)
        self.assertNotEqual(world.roles(0)["charge"], world.roles(1)["charge"])
        a, b = world.reset(0), world.reset(1)
        self.assertEqual(a.observe(), b.observe())
        for sim in (a, b):
            sim.step("A", random.Random(2))
            sim.step("E", random.Random(2))
        self.assertNotEqual(a.observe(), b.observe())
        self.assertEqual(len(a.observe()), 6)  # never reveal phase or queue kinds

    def test_success_only_from_delayed_public_sequence(self):
        world = World(123, 0, 0, 0, 2, 0)
        script = iter(("A", "B", "E", "E", "D"))
        trace, resolved = episode(world, lambda obs, rng: next(script), 2, phase=0)
        self.assertTrue(success(trace))
        self.assertEqual(resolved, 2)
        self.assertEqual(trace[-1]["reward"], 1.0)

    def test_a1_stores_raw_episodes_not_policy_value_tables(self):
        world = World(123, 0, 0, 0, 2, 0)
        script = iter(("A", "B", "E", "E", "D"))
        trace, _ = episode(world, lambda obs, rng: next(script), 2, 0)
        memory = Episodic()
        memory.observe(trace)
        self.assertFalse(hasattr(memory, "by_state"))
        self.assertEqual(len(memory.episodes), 1)
        self.assertEqual(memory.choose(world.reset(0).observe(), random.Random(9)), "A")
        self.assertEqual(memory.snapshot()["retrieval_k"], RECALL_K)
        self.assertIs(memory.episodes[0][0], trace)  # references actual raw experiences

    def test_a3_keeps_selected_trace_after_guard_failure(self):
        world = World(123, 0, 0, 0, 2, 0)
        script = iter(("A", "B", "E", "E", "D"))
        trace, _ = episode(world, lambda obs, rng: next(script), 2, 0)
        options, effects = Options(), Effects()
        options.observe(trace)
        effects.observe(trace)
        controller = OptionController(options, effects)
        start = world.reset(0).observe()
        self.assertEqual(controller.choose(start, random.Random(1)), "A")
        selected = controller.selected_trace
        self.assertTrue(selected)
        controller.choose((0, 0, 1, 1, 0, 0), random.Random(2))
        self.assertTrue(controller.aborted)
        self.assertEqual(controller.selected_trace, selected)
        options.feedback(selected, False)
        self.assertEqual(options.contradicted[selected], 1)

    def test_genuine_feedback_changes_policy_preference(self):
        options = Options()
        initial = (0, 0, 0, 0, 0, 0)
        old = ((initial, "A"),)
        new = ((initial, "B"),)
        options.support[old] = 3
        options.support[new] = 1
        options.by_start[initial].update((old, new))
        self.assertEqual(options.start(initial), old)
        options.feedback(old, False)
        options.feedback(old, False)
        self.assertEqual(options.start(initial), new)

    def test_reproducible_shared_training_small_probe_and_boundaries(self):
        first = train(1, worlds=4, episodes=16, probes=2)
        again = train(1, worlds=4, episodes=16, probes=2)
        self.assertEqual(first, again)
        self.assertEqual(first["training"]["episodes"], 64)
        self.assertEqual(first["training"]["actions"], first["effects_observed_steps"])
        self.assertEqual(first["episodic"]["episodes"], 64)
        self.assertTrue(first["all_arms_saw_identical_training_experience"])
        self.assertEqual(set(first["probe_results_by_rule_phase"]), {"0", "1"})
        self.assertTrue(all(v["episodes"] == 4 for group in first["probe_results_by_rule_phase"].values()
                            for v in group.values()))
        self.assertFalse(first["seeds_dev_opened"])
        self.assertFalse(first["seeds_confirm_opened"])

    def test_artifact_hashes_and_no_privileged_fields_in_actor_trace(self):
        with TemporaryDirectory() as directory:
            out = Path(directory)
            report = train(0, worlds=3, episodes=16, probes=1, out=out)
            with gzip.open(out / "train_events.jsonl.gz", "rt", encoding="utf-8") as stream:
                raw = stream.read()
            self.assertEqual(hashlib.sha256(raw.encode()).hexdigest(), report["shared_event_sha256"])
            records = [json.loads(x) for x in raw.splitlines()]
            self.assertEqual(len(records), 48)
            self.assertEqual({r["split"] for r in records}, {"TRAIN"})
            self.assertEqual({r["phase_for_audit_only"] for r in records}, {0, 1})
            self.assertTrue(all(r["source"] in {"random", "options"} for r in records))
            self.assertTrue(all(set(s) == {"before", "action", "after", "reward", "done"}
                                for r in records for s in r["steps"]))
            self.assertTrue(all(s["action"] in ACTIONS for r in records for s in r["steps"]))
            self.assertTrue(all(TRAIN_BASE <= r["world_seed"] < 61_200_000 for r in records))
            check = json.loads((out / "checkpoint.json").read_text())
            self.assertEqual(check["status"], "TRAIN_ONLY")
            self.assertEqual(check["event_sha256"], report["shared_event_sha256"])

    def test_never_exceed_preregistered_training_budget(self):
        for kwargs in ({"replica": 4}, {"replica": 0, "worlds": 513},
                       {"replica": 0, "episodes": 31}, {"replica": 0, "probes": 3}):
            with self.assertRaises(ValueError):
                train(**kwargs)


if __name__ == "__main__":
    unittest.main()
