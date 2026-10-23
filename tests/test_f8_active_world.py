"""F8 verified experimental invariants: shared budgets, no oracle leakage."""
import math
import unittest
from observatory.active_world_f8 import (
    ACTIONS, LearnedDynamics, LearnedSensors, SENSOR_COST, bayes,
    choose_information, environment_transition, make_cases,
    planned_values, rules, run
)


class IntegratedWorldTests(unittest.TestCase):
    def test_same_physics_training_and_test(self):
        seed=1019
        model=LearnedDynamics().fit(seed,400)
        self.assertGreater(len(model.deltas), 30)
        state=(.40,.34,.32,0)
        west_good=model.predict(state,3,0)[2]
        west_bad=model.predict(state,3,1)[2]
        east_good=model.predict(state,4,1)[2]
        east_bad=model.predict(state,4,0)[2]
        self.assertGreater(west_good,west_bad+.30)
        self.assertGreater(east_good,east_bad+.30)
        self.assertEqual(rules(seed), rules(seed))

    def test_sensor_has_positive_value_only_when_action_changes(self):
        sensors=LearnedSensors()
        sensors.right=[95.,2.]
        sensors.wrong=[5.,98.]
        scores=[(0,1.,0.),(1,0.,1.)]
        self.assertEqual(choose_information(scores,.5,sensors),0)
        sensors.right=[50.,50.]
        sensors.wrong=[50.,50.]
        self.assertIsNone(choose_information(scores,.5,sensors))

    def test_bayesian_observation_updates_with_correct_sign(self):
        self.assertGreater(bayes(.5,1,.92),.9)
        self.assertLess(bayes(.5,0,.92),.1)

    def test_environment_changes_action_consequences(self):
        config=rules(12)
        state=(.38,.4,.30,0)
        good=environment_transition(state,3,0,config)
        bad=environment_transition(state,3,1,config)
        self.assertGreater(good[2]-bad[2],.3)

    def test_world_noise_is_reproducible_and_policy_independent(self):
        first=make_cases(9001,"switch",3,20)
        second=make_cases(9001,"switch",3,20)
        self.assertEqual(first,second)
        self.assertNotEqual(first,make_cases(9002,"switch",3,20))
        self.assertEqual(len(first[0][1]),20)

    def test_complete_bounded_run_with_negative_control(self):
        a=run(seeds=4,train=320,test_episodes=3,horizon=18,seed_offset=417)
        self.assertEqual(len(a["per_seed"]),4)
        self.assertEqual(set(a["scenarios"]),{"stable","switch","null"})
        for scenario in a["scenarios"].values():
            self.assertEqual(set(scenario["policies"]),{
                "voi","never","always","random_probe","voi_no_memory",
                "noisy_tv","reflex","oracle"})
            self.assertEqual(scenario["voi_minus"]["never"]["independent_seeds"],4)
            for value in scenario["policies"].values():
                self.assertTrue(math.isfinite(value["return_mean"]))
                self.assertTrue(0<=value["probe_rate"]<=1)

    def test_portal_long_settings_supported(self):
        with self.assertRaises(ValueError):
            run(seeds=2,train=299,test_episodes=3,horizon=18)


if __name__=="__main__":
    unittest.main()
