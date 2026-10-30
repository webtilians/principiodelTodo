"""F9-N contracts tested without opening the prospective 400000+ worlds."""
import unittest

import numpy as np

from observatory.f9d_autonomous_navigation import checked_seed
from observatory.f9m_temporal_dropout_training import make_model
from observatory.f9n_stage_autonomy import (
    CLOCK_TIMES, CONDITIONS, StageObserver, autonomous_episode,
    confirmation_seed, mask_actor, pair_summary,
)


class F9NStageTests(unittest.TestCase):
    def test_locked_new_partition_and_no_historical_holdout(self):
        self.assertEqual(confirmation_seed(0,13,0),400000)
        self.assertEqual(confirmation_seed(3,13,39),430039)
        self.assertEqual(confirmation_seed(3,7,19),430019)
        self.assertNotEqual(confirmation_seed(0,13,0),
                            checked_seed("development",0,0))
        self.assertNotIn(confirmation_seed(0,13,0),range(300000,340021))
        self.assertLess(confirmation_seed(3,13,39),900000)
        for i,s,j in ((4,13,0),(0,13,40),(0,7,20),(0,8,0)):
            with self.assertRaises(ValueError):
                confirmation_seed(i,s,j)
        with self.assertRaises(ValueError):
            checked_seed("heldout",0,0)

    def test_clock_and_cue_loss_keep_public_frame(self):
        x=np.zeros(660,dtype=np.float32)
        x[:148]=0.25
        x[148:212]=0.75
        self.assertEqual(CLOCK_TIMES,frozenset((3,4,5,6)))
        self.assertEqual(CONDITIONS,("normal","cue_lost_continuous","fixed_4"))
        for t in range(12):
            masked, scheduled, effect=mask_actor(x,"fixed_4",t,False,False)
            self.assertEqual(scheduled,t in CLOCK_TIMES)
            self.assertEqual(effect,t in CLOCK_TIMES)
            np.testing.assert_array_equal(masked[:148],x[:148])
            np.testing.assert_array_equal(masked[148:],0 if scheduled else x[148:])
        for seen, visible, expected in (
            (False,False,False),(True,True,False),(True,False,True)):
            _, scheduled, _ = mask_actor(x,"cue_lost_continuous",5,seen,visible)
            self.assertEqual(scheduled,expected)
        y,scheduled,changed=mask_actor(x,"normal",5,True,False)
        np.testing.assert_array_equal(y,x)
        self.assertFalse(scheduled or changed)
        with self.assertRaises(ValueError):
            mask_actor(x,"invalid",2,True,False)

    def test_stage_observer_first_commit_not_final_success(self):
        labels={"success_position":[5,1],"correct_side":"upper"}
        o=StageObserver(labels,7)
        o.observe((1,3),0)
        o.observe((5,3),6)
        self.assertTrue(o.summary()["junction_arrived"])
        self.assertFalse(o.summary()["first_committed"])
        o.observe((5,2),7)
        o.observe((5,4),8)   # first decision remains upper, no reclassification
        r=o.summary()
        self.assertTrue(r["first_committed"])
        self.assertEqual(r["first_committed_side"],"upper")
        self.assertEqual(r["first_choice_correct"],1)
        self.assertEqual(r["first_commit_step"],7)
        d=StageObserver(labels,7)
        d.observe((5,2),3)
        self.assertTrue(d.summary()["arm_without_center"])
        self.assertFalse(d.summary()["first_committed"])

    def test_pair_summary_all_denominators_and_no_selection(self):
        common=dict(cue_seen=True,junction_arrived=True,first_committed=True,
                    first_choice_wrong=0,timeout=False,arm_without_center=False,
                    terminal_without_center=False,scheduled_masks=4,
                    effective_masks=4,intervention_applied=True,
                    clock_4_complete=True,length=10)
        a=common|dict(correct_side="upper",cue="key",first_committed_side="upper",
                      first_choice_correct=1,success=1)
        b=common|dict(correct_side="lower",cue="ball",first_committed_side="lower",
                      first_choice_correct=1,success=1)
        s=pair_summary([(a,b)])
        self.assertEqual(s["pairs"],1)
        self.assertEqual(s["episodes"],2)
        self.assertEqual(s["pair_counts"]["both_first_correct"],1)
        self.assertEqual(s["conditional"]["correct_given_commit"],1)
        self.assertEqual(s["rates"]["junction_arrived"],1)
        with self.assertRaises(AssertionError):
            pair_summary([(a,a)])

    def test_one_untrained_policy_on_train_worlds_only(self):
        # This is a software smoke test; never use the 400000 confirm partition.
        model=make_model("sham",0,2)
        try:
            seed=checked_seed("train",0,0)
            a=autonomous_episode(model,0,7,seed,False,"normal")
            b=autonomous_episode(model,0,7,seed,True,"fixed_4")
            self.assertEqual(a["scheduled_masks"],0)
            self.assertEqual(b["scheduled_masks"],len(
                CLOCK_TIMES.intersection(range(b["length"]))))
            self.assertEqual(a["success"] in (0,1),True)
            self.assertEqual(a["cue"]!=b["cue"],True)
            self.assertEqual(a["correct_side"]!=b["correct_side"],True)
            self.assertEqual(b["first_choice_correct"] in (0,1),True)
        finally:
            model.get_env().close()


if __name__=="__main__":
    unittest.main()
