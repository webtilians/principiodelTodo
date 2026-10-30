"""F9-S contract tests on fresh 890000 calibration worlds, never 895000 confirm."""
import unittest
import numpy as np

from observatory.f9o_long_blackout_training import make_model
from observatory.f9q_causal_fork import matched_pair
from observatory.f9s_train_cue_axes import train_seed,confirm_seed,CONTROL_SEED
from observatory.f9s_cue_subspace import (
    normalize,orthogonal_control,patch_axis,validate_axes,
    construct_states,CONDITIONS,MODES,
    analyze_world,summarize,
)


class F9SContracts(unittest.TestCase):
    def test_disjoint_worlds_and_historical_holdout_untouched(self):
        self.assertEqual(train_seed(0,0),890000)
        self.assertEqual(train_seed(3,79),893079)
        self.assertEqual(confirm_seed(0,0),895000)
        self.assertEqual(confirm_seed(3,39),898039)
        train={train_seed(i,j) for i in range(4) for j in range(80)}
        confirm={confirm_seed(i,j) for i in range(4) for j in range(40)}
        self.assertEqual(len(train),320)
        self.assertEqual(len(confirm),160)
        self.assertFalse(train&confirm)
        self.assertLess(max(confirm),900000)
        for fn,invalid in ((train_seed,((4,0),(0,80))),
                           (confirm_seed,((4,0),(0,40)))):
            for args in invalid:
                with self.assertRaises(ValueError):
                    fn(*args)

    def test_one_dimensional_cue_patch_versus_equal_l2_orthogonal_control(self):
        cue=normalize(np.arange(1,65,dtype=np.float32))
        u=orthogonal_control(cue,CONTROL_SEED)
        axis={"cue":cue.tolist(),"orthogonal":u.tolist(),
              "paired_train_worlds":80,
              "fit_method":"mean_ball_minus_key_at_last"}
        validate_axes({"H":axis,"C":axis})
        target=np.zeros((1,1,64),np.float32)
        source=np.arange(1,65,dtype=np.float32).reshape(1,1,64)
        along=patch_axis(target,source,axis)
        orth=patch_axis(target,source,axis,orthogonal=True)
        np.testing.assert_allclose(np.linalg.norm(along-target),
                                   np.linalg.norm(orth-target),rtol=1.e-6)
        self.assertGreater(float(np.dot(along.flatten(),cue)),0)
        self.assertAlmostEqual(float(np.dot(orth.flatten(),cue)),0,places=4)
        np.testing.assert_array_equal(target,0)
        np.testing.assert_array_equal(patch_axis(source,source,axis),source)

    def test_mixed_history_and_causal_conditions_on_train_only(self):
        model=make_model("continuous6",0)
        try:
            pair=matched_pair(model,0,train_seed(0,0))
            w=normalize(np.arange(1,65,dtype=np.float32))
            u=orthogonal_control(w,CONTROL_SEED)
            axes={name:{"cue":w.tolist(),"orthogonal":u.tolist(),
                        "paired_train_worlds":80,
                        "fit_method":"mean_ball_minus_key_at_last"}
                  for name in ("H","C")}
            states,audits=construct_states(model,0,pair,axes)
            self.assertEqual(set(states),set(CONDITIONS))
            self.assertEqual(MODES,("intact","swap_HC"))
            for name,items in states.items():
                self.assertEqual(len(items),2)
                for h,c in items:
                    self.assertEqual(h.shape,(1,1,64))
                    self.assertEqual(c.shape,(1,1,64))
                if name=="normal_reference":
                    self.assertEqual(audits[name]["effective"],0)
                else:
                    self.assertEqual(audits[name]["effective"],
                                     pair["times"]["fork"]-pair["times"]["post_cue"])
            # The identity of key vs ball must not select geometry at fork.
            np.testing.assert_array_equal(
                pair["frames"][0][pair["times"]["fork"]],
                pair["frames"][1][pair["times"]["fork"]])
            result=analyze_world(model,0,train_seed(0,0),axes)
            summary=summarize([result])
            self.assertEqual(summary["pairs"],1)
            self.assertEqual(set(summary["conditions"]),set(CONDITIONS))
            self.assertTrue(result["fork_public_identical"])
            self.assertTrue(result["F4_fork_and_continuation_zero"])
            self.assertIn("cue_axis_causal_flip_pairs",summary)
            for c in result["conditions"].values():
                self.assertEqual(set(c["runs"]),set(MODES))
                self.assertTrue(all(r["F4_zero_throughout_decision"]
                                    for p in c["runs"].values() for r in p))
        finally:
            model.get_env().close()


if __name__=="__main__":
    unittest.main()
