"""F9-K: integrity, no leakage, paired metric, and native MiniGrid tests."""
import unittest

import numpy as np

from observatory.f9d_autonomous_navigation import checked_seed
from observatory import f9i_selective_navigation as nav
from observatory.f9k_temporal_memory_probes import (
    STAGES, TRAIN_PAIRS, DEV_PAIRS, build_dataset, fit_linear_probe,
    flatten_hidden, matrix, paired_snapshots, predict_probe, scores,
    stage_indices,
)


class FrozenTemporalProbeTests(unittest.TestCase):
    def test_no_reserved_split_and_preassigned_budgets(self):
        with self.assertRaises(ValueError):
            checked_seed("heldout", 0, 0)
        with self.assertRaises(ValueError):
            build_dataset(None, "context", 0, 7, "heldout", 1)
        with self.assertRaises(ValueError):
            build_dataset(None, "context", 0, 7, "train", TRAIN_PAIRS+1)
        with self.assertRaises(ValueError):
            build_dataset(None, "context", 0, 13, "development", DEV_PAIRS+1)

    def test_indices_derived_from_teacher_visibility_not_model_predictions(self):
        a = np.array([0, 2, 2, 1, 2, 0, 1])
        b = np.array([0, 2, 2, 1, 2, 2, 1])
        visibility = [False, True, True, False, False, False, False]
        self.assertEqual(stage_indices(a,b,visibility,visibility),
                         {"post_cue":3, "halfway":4, "fork":5})
        with self.assertRaises(AssertionError):
            stage_indices(a,b,[False]*7,[False]*7)
        with self.assertRaises(AssertionError):
            stage_indices(a,a,visibility,visibility)

    def test_probe_recovers_only_signal_with_pairwise_disjoint_evaluation(self):
        def dataset(start, count):
            rows=[]
            for j in range(start,start+count):
                noise = float(j)/100
                a={"H":np.array([-5.0, noise]),
                   "M":np.zeros(2), "O":np.zeros(2)}
                b={"H":np.array([5.0, noise]),
                   "M":np.zeros(2), "O":np.zeros(2)}
                rows.append({"a":a,"b":b,"raw_equal":True,
                             "f4_equal":True,"hidden_distance":10.0,
                             "cue_gap":3})
            return rows
        training=dataset(0,30)
        evaluation=dataset(30,10)
        x,y=matrix(training,"H")
        self.assertEqual(x.shape,(60,2))
        probe=fit_linear_probe(x,y)
        evaluated=scores(probe,evaluation,"H")
        self.assertEqual(evaluated["accuracy"],1.0)
        self.assertEqual(evaluated["both_correct"],10)
        self.assertEqual(evaluated["paired_flips"],10)
        # Identical features cannot decode inversions, even with balanced labels.
        x_zero,y_zero=matrix(training,"M")
        control=fit_linear_probe(x_zero,y_zero)
        evaluated_zero=scores(control,evaluation,"M")
        self.assertEqual(evaluated_zero["accuracy"],0.5)
        self.assertEqual(evaluated_zero["both_correct"],0)
        np.testing.assert_array_equal(predict_probe(control,np.zeros((2,2))),
                                      np.ones(2))

    def test_hidden_width_and_rejects_drift(self):
        self.assertEqual(flatten_hidden(None).shape, (128,))
        with self.assertRaises(AssertionError):
            flatten_hidden((np.zeros((1,1,65)),np.zeros((1,1,64))))

    def test_public_observation_is_matched_after_cue_native_task(self):
        old_capacity,old_seed=nav.EVENT_CAPACITY,nav.SELECTOR_SEED
        nav.EVENT_CAPACITY,nav.SELECTOR_SEED=8,81000
        model=nav.make_model("context",0,2)
        try:
            for size in (7,13):
                pair=paired_snapshots(
                    model,"context",0,size,checked_seed("development",0,0))
                self.assertEqual(set(pair),set(STAGES))
                for checkpoint in pair.values():
                    self.assertTrue(checkpoint["raw_equal"])
                    self.assertGreaterEqual(checkpoint["cue_gap"],1)
                    self.assertEqual(checkpoint["a"]["H"].shape,(128,))
                    self.assertEqual(checkpoint["a"]["M"].shape,(512,))
                    self.assertEqual(checkpoint["a"]["O"].shape,(148,))
        finally:
            model.get_env().close()
            nav.EVENT_CAPACITY,nav.SELECTOR_SEED=old_capacity,old_seed


if __name__ == "__main__":
    unittest.main()
