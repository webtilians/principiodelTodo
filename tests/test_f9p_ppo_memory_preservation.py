"""F9-P preregistered public-only PPO clock, frozen-source and new split."""
import unittest
import numpy as np

from observatory.f9p_ppo_memory_preservation import (
    ARMS,TRAIN_BASE,ACTUAL_STEPS,PERIOD,NORMAL_PREFIX,
    train_seed,scheduled,actor_transform,PPOClockedF4,
)
from observatory.f9p_confirm_memory_preservation import confirmation_seed
from observatory.f9n_stage_autonomy import autonomous_episode


class F9PContractTests(unittest.TestCase):
    def test_exogenous_long_pulse_and_equal_opportunity(self):
        self.assertEqual(ARMS,("ppo_sham","ppo_blackout"))
        self.assertEqual(PERIOD,128)
        self.assertEqual(NORMAL_PREFIX,16)
        self.assertEqual(sum(scheduled(i) for i in range(128)),112)
        self.assertEqual(sum(scheduled(i) for i in range(ACTUAL_STEPS+1)),
                         112*(ACTUAL_STEPS//128))
        for i in range(128):
            self.assertEqual(scheduled(i),i>=16)
        for invalid in (-1,1.5,"22"):
            with self.assertRaises(ValueError):
                scheduled(invalid)

    def test_actor_observation_preserved_and_sham_is_truly_sham(self):
        raw=np.zeros(660,dtype=np.float32)
        raw[:148]=.33
        raw[148:212]=.75
        for step in (0,15,16,120,128,143,144):
            sham,active_s,changed_s=actor_transform(raw,"ppo_sham",step)
            dropout,active_d,changed_d=actor_transform(raw,"ppo_blackout",step)
            self.assertEqual(active_s,active_d)
            np.testing.assert_array_equal(sham,raw)
            np.testing.assert_array_equal(dropout[:148],raw[:148])
            self.assertFalse(changed_s)
            self.assertEqual(changed_d,scheduled(step))
            np.testing.assert_array_equal(
                dropout[148:],np.zeros(512) if scheduled(step) else raw[148:])
        with self.assertRaises(ValueError):
            actor_transform(raw,"unknown",18)

    def test_actor_global_step_does_not_reset_between_episodes(self):
        for arm in ARMS:
            env=PPOClockedF4(0,arm)
            try:
                obs,_=env.reset()
                self.assertEqual(obs.shape,(660,))
                for _ in range(135):
                    obs,rew,term,trunc,info=env.step(0)
                    if term or trunc:
                        obs,_=env.reset()
                self.assertEqual(env.global_step,135)
                stats=env.metrics()
                self.assertEqual(stats["scheduled_unique_steps"],112)
                self.assertEqual(stats["native_steps"],135)
                self.assertLessEqual(stats["writer_max_occupancy"],8)
                if arm=="ppo_sham":
                    self.assertEqual(stats["effective_nonzero_unique_steps"],0)
            finally:
                env.close()

    def test_new_ppo_and_confirmation_seed_partitions_do_not_overlap(self):
        self.assertEqual(train_seed(0,0),550000)
        self.assertEqual(train_seed(3,599),580599)
        self.assertEqual(confirmation_seed(0,13,0),700000)
        self.assertEqual(confirmation_seed(3,13,39),730039)
        self.assertEqual(confirmation_seed(3,7,19),730019)
        self.assertLess(confirmation_seed(3,13,39),900000)
        for args in ((4,0),(0,600),(-1,0)):
            with self.assertRaises(ValueError):
                train_seed(*args)
        for args in ((4,13,0),(0,13,40),(0,7,20)):
            with self.assertRaises(ValueError):
                confirmation_seed(*args)

    def test_optional_hidden_reset_rejects_non_cueloss(self):
        with self.assertRaises(ValueError):
            autonomous_episode(None,0,13,550000,False,"normal",
                               reset_hidden_on_fork=True)


if __name__=="__main__":
    unittest.main()
