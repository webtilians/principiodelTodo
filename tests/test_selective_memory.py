"""F4 tests: capacity, causal time boundary, adaptation, null and repeatability."""
import unittest

from observatory.selective_memory import (
    CAPACITY, Event, FIFO, LearnedRelevance, RandomReservoir,
    Store, attempt, build_profile, evaluate, make_episode, summarize,
)


class SelectiveMemoryTests(unittest.TestCase):
    def test_capacity_and_eviction_are_explicit(self):
        m = Store(2)
        m.put(Event(0, 0))
        m.put(Event(1, 1))
        with self.assertRaisesRegex(ValueError, "explicitly choose"):
            m.put(Event(2, 0))
        m.put(Event(2, 0), evict_key=0)
        self.assertEqual(len(m.saved), 2)
        self.assertIsNone(m.recall(0))

    def test_feedback_updates_only_after_answer(self):
        learner = LearnedRelevance(CAPACITY, 5)
        values = tuple(learner.weights)
        episode = make_episode(2, 1, build_profile(2)[0], False)
        # Caller cannot pass query to consider: its signature is store, event.
        memory = learner.begin()
        for event in episode.events:
            learner.consider(memory, event)
        self.assertEqual(tuple(learner.weights), values)
        self.assertEqual(learner.feedback_count, 0)
        self.assertEqual(len(memory.saved), CAPACITY)
        attempt(learner, episode)
        self.assertEqual(learner.feedback_count, 1)
        self.assertNotEqual(tuple(learner.weights), values)

    def test_freezing_prevents_future_learning(self):
        learner = LearnedRelevance(3, 5)
        learner.feedback(6)
        learner.freeze()
        weights = tuple(learner.weights)
        learner.feedback(2)
        self.assertEqual(tuple(learner.weights), weights)

    def test_episode_is_not_a_sequence_position_cheat(self):
        ep = make_episode(11, 0, build_profile(11)[0], False)
        self.assertEqual({e.key for e in ep.events}, set(range(8)))
        self.assertEqual(len(ep.events), 8)
        self.assertIn(ep.future_query_key, set(range(8)))
        self.assertEqual({e.payload for e in ep.events}.issubset({0, 1}), True)

    def test_deterministic_under_seed(self):
        a = evaluate(19, phase_length=40, test_length=50)
        b = evaluate(19, phase_length=40, test_length=50)
        self.assertEqual(a, b)

    def test_delayed_query_learning_improves_after_shift(self):
        # Non-statistical smoke assertion; actual effect size and
        # uncertainty come from the larger predeclared workflow.
        results = [evaluate(i) for i in range(1, 13)]
        summary = summarize(results, uniform=False)
        learner = summary["strategies"]["adaptive"]["heldout_hit"]
        fifo = summary["strategies"]["fifo"]["heldout_hit"]
        frozen = summary["strategies"]["static_first_regime"]["heldout_hit"]
        self.assertGreater(learner, fifo + .15)
        self.assertGreater(learner, frozen + .15)

    def test_uniform_control_has_no_true_key_advantage(self):
        results = [evaluate(i + 2000, uniform=True) for i in range(1, 13)]
        summary = summarize(results, uniform=True)
        for x in summary["strategies"].values():
            self.assertTrue(.25 <= x["heldout_hit"] <= .5)


if __name__ == "__main__":
    unittest.main()
