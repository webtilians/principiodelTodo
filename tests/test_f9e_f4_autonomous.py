"""F9-E: real F4 FIFO input integrity, causal readout and native transition tests."""
import unittest
import numpy as np

from observatory.f9e_f4_autonomous import (
    VisibleF4, MemoryObservation, augment_dataset, make_model, clone, rollout,
)
from observatory.f9d_autonomous_navigation import TrainingWorlds, checked_seed, demonstration
from observatory.f9b_behavior_audit import build_env


class F4NavigationIntegrity(unittest.TestCase):
    def test_f4_is_real_store_and_identity_ablation_never_changes_current_obs(self):
        from minigrid.core.constants import OBJECT_TO_IDX, COLOR_TO_IDX
        template=np.zeros(148,np.float32)
        template[147]=1/3.
        obs=template.copy()
        frame=obs[:147].reshape(7,7,3)
        frame[1,2]=(OBJECT_TO_IDX['key']/max(OBJECT_TO_IDX.values()),
                    COLOR_TO_IDX['green']/max(COLOR_TO_IDX.values()), 0)
        f4=VisibleF4('f4')
        zero=VisibleF4('zero')
        erased=VisibleF4('erase')
        swapped=VisibleF4('swap_identity')
        real=f4.encode(obs)
        disabled=zero.encode(obs)
        erased_frame=erased.encode(obs)
        altered=swapped.encode(obs)
        np.testing.assert_array_equal(real[:148],obs)
        np.testing.assert_array_equal(disabled[:148],obs)
        np.testing.assert_array_equal(erased_frame[:148],obs)
        np.testing.assert_array_equal(altered[:148],obs)
        self.assertEqual(real.shape,(160,))
        self.assertGreater(real[148:].sum(),0.)
        self.assertFalse(np.any(disabled[148:]))
        self.assertFalse(np.any(erased_frame[148:]))
        self.assertNotEqual(list(real[148:]),list(altered[148:]))
        self.assertEqual(len(f4.store.order),1)
        self.assertEqual(type(f4.store).__name__,'Store')
        f4.encode(np.zeros(148,np.float32))
        np.testing.assert_array_equal(real[148:],f4.readout())
        f4.reset()
        self.assertFalse(np.any(f4.readout()))

    def test_original_actions_and_rewards_unchanged_by_memory_wrapper(self):
        a=MemoryObservation(TrainingWorlds(0,2),'f4')
        b=TrainingWorlds(0,2)
        try:
            x,_=a.reset()
            y,_=b.reset()
            self.assertEqual(x.shape,(160,))
            np.testing.assert_array_equal(x[:148],y)
            self.assertEqual(a.action_space.n,7)
            for act in (0,1,2,0,0,2,2):
                x,ra,da,ta,_=a.step(act)
                y,rb,db,tb,_=b.step(act)
                np.testing.assert_array_equal(x[:148],y)
                self.assertEqual((ra,da,ta),(rb,db,tb))
        finally:
            a.close()
            b.close()

    def test_bc_has_same_actions_and_budgets(self):
        examples=[demonstration(checked_seed('train',0,j),reverse=r)
                  for j in (0,1) for r in (False,True)]
        f4=augment_dataset(examples,'f4')
        zero=augment_dataset(examples,'zero')
        self.assertTrue(all(np.array_equal(a[1],b[1]) for a,b in zip(f4,zero)))
        self.assertTrue(all(np.array_equal(a[0][:,:148],b[0][:,:148])
                            for a,b in zip(f4,zero)))
        self.assertTrue(all(a[0].shape[1]==160 for a in f4))
        self.assertTrue(all(np.all(z[0][:,148:]==0) for z in zero))
        self.assertTrue(any(np.any(a[0][:,148:]!=0) for a in f4))

    def test_models_have_matchable_f4_and_zero_widths(self):
        models=[]
        try:
            for mode in ('raw','zero','f4'):
                model=make_model(mode,0,2)
                models.append(model)
                self.assertEqual(model.action_space.n,7)
                self.assertEqual(model.observation_space.shape[0],
                                 148 if mode=='raw' else 160)
            self.assertEqual(sum(p.numel() for p in models[1].policy.parameters()),
                             sum(p.numel() for p in models[2].policy.parameters()))
        finally:
            for model in models:
                model.get_env().close()

    def test_real_recurrent_trajectory_and_all_f4_interventions(self):
        model=make_model('f4',0,2)
        try:
            training=[demonstration(checked_seed('train',0,j),reverse=r)
                      for j in (0,1) for r in (False,True)]
            history=clone(model,augment_dataset(training,'f4'),1,0)
            self.assertTrue(np.isfinite(history).all())
            model.learn(total_timesteps=128)
            for control in ('normal','erase','swap_identity','mask_cue'):
                a=rollout(model,'f4',7,checked_seed('development',0,0),False,control)
                self.assertTrue(all(0<=x['action']<7 for x in a['trace']))
                self.assertTrue(len(a['trace'])>0)
                self.assertTrue(all(len(x['memory_readout'])==12
                                    for x in a['trace']))
        finally:
            model.get_env().close()

    def test_reset_sealed_and_invalid_ablation_rejected(self):
        with self.assertRaises(ValueError):
            checked_seed('heldout',0,0)
        with self.assertRaises(ValueError):
            VisibleF4('oracle')
        with self.assertRaises(ValueError):
            rollout(None,'zero',7,checked_seed('development',0,0),control='erase')


if __name__=='__main__':
    unittest.main()
