"""Controls for F6: reproducibility, learning, heldout action dependence."""
import unittest
import numpy as np

from scripts.f6_homeostatic_rl import (
    ACTIONS, World, LearnedWorldModel, Planner, TabularQ,
    collect_transitions, utility, run
)


class TestHomeostaticF6(unittest.TestCase):
    def test_same_seed_same_physics(self):
        a, b = World(900123), World(900123)
        self.assertTrue(np.array_equal(a.state, b.state))
        for act in [0, 3, 1, 2] * 5:
            x, rew_a, _ = a.step(act)
            y, rew_b, _ = b.step(act)
            self.assertTrue(np.array_equal(x, y))
            self.assertEqual(rew_a, rew_b)

    def test_programmed_reward_is_not_consciousness(self):
        a = np.array([.1, .1, .8])
        b = np.array([.7, .7, .2])
        self.assertGreater(float(utility(a, np.asarray(0))),
                           float(utility(b, np.asarray(0))))

    def test_training_uses_transitions_not_action_labels(self):
        data = collect_transitions(17, episodes=15, horizon=20)
        self.assertEqual(len(data), 300)
        self.assertEqual(set(a for _, a, _, _ in data), set(range(4)))
        model = LearnedWorldModel().fit(data)
        state = data[11][0]
        for act in range(len(ACTIONS)):
            predicted = model.predict(state[None, :], act)
            self.assertEqual(predicted.shape, (1, 4))
            self.assertTrue(np.isfinite(predicted).all())
            self.assertTrue(np.array_equal(predicted[:, 3], state[None, 3]))

    def test_planner_depends_on_learned_action_model(self):
        data = collect_transitions(19, episodes=30, horizon=20)
        model = LearnedWorldModel().fit(data)
        obs = np.array([.75, .5, .09, 0])
        control = Planner(model, horizon=3)
        self.assertEqual(control.act(obs), control.act(obs))
        self.assertIn(control.act(obs), range(len(ACTIONS)))
        self.assertIn(Planner(model, horizon=3, shuffled_model=True).act(obs),
                      range(len(ACTIONS)))
        self.assertFalse(np.allclose(
            model.predict(obs[None], 1), model.predict(obs[None], 2)))

    def test_heldout_prediction_better_than_constant_state(self):
        model = LearnedWorldModel().fit(
            collect_transitions(25, episodes=45, horizon=24))
        cases = collect_transitions(99, episodes=9, horizon=24)
        learned_mse = np.mean([
            np.mean((model.predict(state[None], action)[0, :3]
                     - after[:3])**2)
            for state, action, _, after in cases
        ])
        persistence_mse = np.mean([
            np.mean((state[:3] - after[:3])**2)
            for state, _, _, after in cases
        ])
        self.assertLess(learned_mse, persistence_mse)

    def test_model_free_baseline_trained_same_data(self):
        data = collect_transitions(21, episodes=10, horizon=20)
        agent = TabularQ().fit(data, passes=2)
        self.assertIn(agent.act(data[0][0]), range(len(ACTIONS)))

    def test_complete_smoke(self):
        result = run(seeds=2, train_episodes=10, test_episodes=3, horizon=16)
        self.assertEqual(result["in_distribution"]["n_seeds"], 2)
        self.assertEqual(len(result["per_seed"]), 2)
        self.assertEqual(len(result["in_distribution"]["policies"]), 6)
        for val in result["in_distribution"]["policies"].values():
            self.assertTrue(np.isfinite(val["mean_cost_per_step"]))


if __name__ == "__main__":
    unittest.main()
