"""F9-M preregistration, public-only augmentation, and split isolation."""
import unittest

import numpy as np

from observatory.f9d_autonomous_navigation import checked_seed, demonstration
from observatory.f9m_temporal_dropout_training import (
    ACTIVE_END, CAPACITY, DIM, PublicTemporalBlackout,
    encoded_demo, make_model, parameter_signature, training_dataset,
)
from observatory.f9m_confirm import confirmation_seed, validate_training


class MatchedTemporalTrainingTests(unittest.TestCase):
    def test_public_only_masking_never_alters_raw_or_selector(self):
        self.assertEqual(ACTIVE_END,212)
        self.assertEqual((CAPACITY,DIM),(8,660))
        base=np.zeros(660,dtype=np.float32)
        base[148:212]=1.
        sham=PublicTemporalBlackout("sham",0,0)
        drop=PublicTemporalBlackout("dropout",0,0)
        self.assertEqual((sham.enabled,sham.length),(drop.enabled,drop.length))
        # Bypass selection probability only for direct deterministic unit tests.
        sham.enabled=drop.enabled=True
        for obj_count in (2,0,0,0,0,0):
            a=sham.transform(base[:148],base,obj_count)
            b=drop.transform(base[:148],base,obj_count)
            np.testing.assert_array_equal(a,base)
            np.testing.assert_array_equal(a[:148],b[:148])
            if 0 < drop.elapsed <= drop.length:
                self.assertFalse(np.any(b[148:]))
        self.assertEqual(sham.effective_steps,0)
        self.assertGreater(drop.effective_steps,0)
        with self.assertRaises(ValueError):
            PublicTemporalBlackout("other",0,0)
        with self.assertRaises(AssertionError):
            drop.transform(base[:148],np.zeros(659,dtype=np.float32),0)

    def test_matched_teacher_sequences_and_initial_weights(self):
        teacher=[
            demonstration(checked_seed("train",0,j),reverse=rev)
            for j in (0,1) for rev in (False,True)
        ]
        a, sham=training_dataset(teacher,"sham",0)
        b, drop=training_dataset(teacher,"dropout",0)
        self.assertEqual(sham["effective_steps"],0)
        self.assertGreater(drop["effective_steps"],0)
        self.assertEqual(sham["episodes_enabled"],drop["episodes_enabled"])
        self.assertEqual(sham["scheduled_steps"],drop["scheduled_steps"])
        for (oa,aa,_),(ob,ab,_),(original,actions,_) in zip(a,b,teacher):
            np.testing.assert_array_equal(aa,ab)
            np.testing.assert_array_equal(aa,actions)
            np.testing.assert_array_equal(oa[:,:148],original)
            np.testing.assert_array_equal(ob[:,:148],original)
            np.testing.assert_array_equal(oa[ACTIVE_END:],0.)
            np.testing.assert_array_equal(ob[ACTIVE_END:],0.)
        m1=make_model("sham",0,2)
        m2=make_model("dropout",0,2)
        try:
            self.assertEqual(parameter_signature(m1.policy),parameter_signature(m2.policy))
            self.assertEqual(m1.observation_space.shape,m2.observation_space.shape)
            self.assertEqual(m1.observation_space.shape,(660,))
            self.assertEqual(sum(p.numel() for p in m1.policy.parameters()),
                             sum(p.numel() for p in m2.policy.parameters()))
        finally:
            m1.get_env().close()
            m2.get_env().close()

    def test_confirmation_is_independent_and_never_historical_heldout(self):
        self.assertEqual(confirmation_seed(0,0),300000)
        self.assertEqual(confirmation_seed(3,19),330019)
        self.assertNotEqual(confirmation_seed(1,0),checked_seed("train",1,0))
        self.assertNotEqual(confirmation_seed(1,0),checked_seed("development",1,0))
        self.assertLess(confirmation_seed(3,19),900000)
        with self.assertRaises(ValueError):
            confirmation_seed(0,20)
        with self.assertRaises(ValueError):
            confirmation_seed(4,0)
        with self.assertRaises(ValueError):
            checked_seed("heldout",0,0)

    def test_confirmation_rejects_incomplete_training_before_any_world_is_opened(self):
        with self.assertRaises(AssertionError):
            validate_training({"seed_index":0,"heldout_opened":False},
                              __import__("pathlib").Path("."),0)


if __name__=="__main__":
    unittest.main()
