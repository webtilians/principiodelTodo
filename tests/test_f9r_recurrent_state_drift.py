"""F9-R preconfirm integrity tests. All fixtures are in disjoint 840000+ only."""
import unittest
import numpy as np

from observatory.f9o_long_blackout_training import make_model
from observatory.f9q_causal_fork import matched_pair
from observatory.f9r_confirm_state_drift import fixture_seed,confirm_seed
from observatory.f9r_recurrent_state_drift import (
    CONDITIONS,SWAPS,trace_history,prepare_matched_conditions,
    first_choice_metrics,analyze_world,summarize_worlds,
)


class F9RContracts(unittest.TestCase):
    def test_all_seed_ranges_disjoint_and_no_900000(self):
        self.assertEqual(fixture_seed(0,0),840000)
        self.assertEqual(fixture_seed(3,9),843009)
        self.assertEqual(confirm_seed(0,0),850000)
        self.assertEqual(confirm_seed(3,39),880039)
        self.assertLess(confirm_seed(3,39),900000)
        a={fixture_seed(i,j) for i in range(4) for j in range(10)}
        b={confirm_seed(i,j) for i in range(4) for j in range(40)}
        self.assertEqual(len(a),40)
        self.assertEqual(len(b),160)
        self.assertFalse(a&b)
        self.assertTrue(all(seed>=840000 and seed<844000 for seed in a))
        self.assertTrue(all(seed>=850000 and seed<890000 for seed in b))
        for idx,j in ((0,10),(4,0),(-1,0)):
            with self.assertRaises(ValueError):
                fixture_seed(idx,j)
        for idx,j in ((0,40),(4,0),(-1,0)):
            with self.assertRaises(ValueError):
                confirm_seed(idx,j)

    def test_freeze_and_causal_replay_all_conditions_on_new_fixture(self):
        model=make_model("continuous6",0)
        try:
            world=matched_pair(model,0,fixture_seed(0,0))
            f=world["times"]["fork"]
            p=world["times"]["post_cue"]
            self.assertGreaterEqual(f-p,4)
            self.assertTrue(np.array_equal(world["frames"][0][f],world["frames"][1][f]))
            states=prepare_matched_conditions(model,0,world)
            self.assertEqual(set(CONDITIONS),set(states))
            self.assertEqual(SWAPS,("intact","swap_HC"))
            for cue in range(2):
                ref=states["normal_reference"][cue]
                blackout=states["blackout_update"][cue]
                frozen=states["blackout_freeze_HC"][cue]
                natural_at_last_cue=ref["natural_snapshots"][p]
                self.assertEqual(ref["scheduled_prefix_masks"],0)
                self.assertEqual(blackout["scheduled_prefix_masks"],f-p)
                self.assertEqual(blackout["effective_prefix_masks"],f-p)
                self.assertEqual(frozen["scheduled_prefix_masks"],f-p)
                for a,b in zip(frozen["state"],natural_at_last_cue):
                    np.testing.assert_array_equal(a,b)
                for name in CONDITIONS:
                    state=states[name][cue]
                    self.assertEqual(state["state"][0].shape,(1,1,64))
                    self.assertEqual(state["state"][1].shape,(1,1,64))
                    np.testing.assert_array_equal(state["raw"],ref["raw"])
                    if name=="normal_reference":
                        self.assertEqual(state["effective_prefix_masks"],0)
                    else:
                        self.assertEqual(state["effective_prefix_masks"],f-p)
                for a,b in zip(ref["state"],states["blackout_patch_normal_fork_HC"][cue]["state"]):
                    np.testing.assert_array_equal(a,b)
            metrics=first_choice_metrics(model,0,world,states)
            self.assertEqual(set(metrics),set(CONDITIONS))
            for row in metrics.values():
                self.assertEqual(set(row["runs"]),set(SWAPS))
                for behavior in row["runs"].values():
                    self.assertEqual(len(behavior),2)
                    self.assertTrue(all(r["F4_zero_throughout_decision"] for r in behavior))
                    self.assertTrue(all(r["first_action"] in range(7) for r in behavior))
            summary=summarize_worlds([{"conditions":metrics}])
            self.assertEqual(summary["pairs"],1)
            self.assertEqual(set(summary["conditions"]),set(CONDITIONS))
            self.assertEqual(
                summary["conditions"]["normal_reference"]["both_intact_first_arms_correct"],
                summary["conditions"]["blackout_patch_normal_fork_HC"]["both_intact_first_arms_correct"])
        finally:
            model.get_env().close()


if __name__=="__main__":
    unittest.main()
