"""F9-H contract: matched policy weights, observation-only F4, native actions."""
import unittest

import numpy as np

from observatory.f9d_autonomous_navigation import (
    TrainingWorlds, checked_seed, demonstration,
)
from observatory.f9h_context_memory_navigation import (
    DIM, make_model, augment_demonstrations, paired_rollout,
)
from observatory.f9g_context_memory import ContextF4, ContextObservation
from observatory.f9b_behavior_audit import build_env


class ConstantNativeActor:
    def predict(self, obs, **kwargs):
        assert np.shape(obs) == (DIM,)
        return np.array(2), None


class MatchedF9HTests(unittest.TestCase):
    def test_contextual_F4_zero_erase_and_swap_same_raw_current_frame(self):
        env=build_env(7)
        try:
            raw,_=env.reset(seed=checked_seed("development",0,0))
            outputs={}
            for mode in ("f4","zero","erase","swap"):
                outputs[mode]=ContextF4(mode).encode(raw)
                self.assertEqual(outputs[mode].shape,(DIM,))
                np.testing.assert_array_equal(outputs[mode][:148],raw)
            self.assertFalse(np.any(outputs["zero"][148:]))
            self.assertFalse(np.any(outputs["erase"][148:]))
            # First view sometimes contains no visible objects. Inspect the
            # native demonstration's cue frame, not assume the start is visible.
            obs,_,_=demonstration(checked_seed("train",0,0))
            records={}
            for mode in ("f4","zero","erase","swap"):
                writer=ContextF4(mode)
                records[mode]=[writer.encode(x) for x in obs]
            self.assertTrue(any(np.any(x[148:]) for x in records["f4"]))
            self.assertTrue(all(not np.any(x[148:]) for x in records["zero"]))
            self.assertTrue(all(not np.any(x[148:]) for x in records["erase"]))
            self.assertTrue(any(not np.array_equal(x[148:],y[148:])
                                for x,y in zip(records["swap"],records["f4"])))
            for mode in records:
                for i in range(len(obs)):
                    np.testing.assert_array_equal(records[mode][i][:148],obs[i])
        finally:
            env.close()

    def test_wrapped_seven_actions_and_rewards_match_original(self):
        for mode in ("f4","zero"):
            a=ContextObservation(TrainingWorlds(0,2),mode)
            b=TrainingWorlds(0,2)
            try:
                first,_=a.reset()
                raw,_=b.reset()
                self.assertEqual(first.shape,(DIM,))
                np.testing.assert_array_equal(first[:148],raw)
                self.assertEqual(a.action_space.n,7)
                for act in range(7):
                    x,rx,dx,tx,ix=a.step(act)
                    y,ry,dy,ty,iy=b.step(act)
                    np.testing.assert_array_equal(x[:148],y)
                    self.assertEqual((rx,dx,tx,ix),(ry,dy,ty,iy))
                    # End resets not needed here: action sequence doesn't
                    # step beyond the very short starting room.
            finally:
                a.close()
                b.close()

    def test_imitation_examples_match_action_sequences_and_dimensions(self):
        originals=[demonstration(checked_seed("train",0,j),reverse=rev)
                   for j in (0,1) for rev in (False,True)]
        f4=augment_demonstrations(originals,"f4")
        zero=augment_demonstrations(originals,"zero")
        self.assertEqual(len(f4),len(zero))
        for a,b,o in zip(f4,zero,originals):
            np.testing.assert_array_equal(a[1],b[1])
            np.testing.assert_array_equal(a[0][:,:148],b[0][:,:148])
            np.testing.assert_array_equal(a[0][:,:148],o[0])
            self.assertEqual(a[0].shape[1],DIM)
            self.assertTrue(np.all(b[0][:,148:]==0))
        with self.assertRaises(ValueError):
            checked_seed("heldout",0,0)

    def test_equal_model_architecture_budget_and_trainability(self):
        models=[make_model(mode,0,2) for mode in ("f4","zero")]
        try:
            self.assertEqual(
                sum(p.numel() for p in models[0].policy.parameters()),
                sum(p.numel() for p in models[1].policy.parameters()))
            self.assertEqual(models[0].observation_space.shape,(DIM,))
            self.assertEqual(models[1].observation_space.shape,(DIM,))
            self.assertEqual(models[0].action_space.n,7)
            self.assertEqual(models[1].action_space.n,7)
        finally:
            for m in models:
                m.get_env().close()

    def test_native_rollout_all_memory_interventions(self):
        # No training needed for the integrity of the boundary.
        model=ConstantNativeActor()
        seed=checked_seed("development",0,0)
        for control in ("normal","erase","swap","mask_cue"):
            r=paired_rollout(model,"f4",7,seed,control=control)
            self.assertTrue(all(0<=t["action"]<7 for t in r["trace"]))
            self.assertEqual(len(r["trace"]),r["length"])
            self.assertIn(r["chosen_side"],("upper","lower","none"))
        with self.assertRaises(ValueError):
            paired_rollout(model,"zero",7,seed,control="swap")


if __name__ == "__main__":
    unittest.main()
