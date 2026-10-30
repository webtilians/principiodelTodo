"""Small deterministic unit tests of the F9-F readout forensic rules."""
import unittest

from observatory.f9f_memory_content_audit import (
    readout_types, assess_pair, summarize,
)


def visible_readout():
    # The memory contains a key AND a ball but has no role/cue annotation.
    return [1.,0.,.2,1.,0.,1.,.2,1.,0.,0.,0.,0.]


class MemoryContentTests(unittest.TestCase):
    def test_f4_slots_have_object_id_but_not_cue_role(self):
        self.assertEqual(readout_types(visible_readout()),["key","ball"])
        with self.assertRaises(ValueError):
            readout_types([1.,1.,0.,1.]+[0.]*8)

    def test_paired_identical_memory_is_not_counterfactual_cue(self):
        base={
            "seed":100000,"side":"upper","success":1,
            "correct_side":"upper","cue":"ball",
            "trace":[
                {"action":2,"position":[1,3],"cue_visible":True,
                 "memory_readout":visible_readout()},
                {"action":0,"position":[5,3],"cue_visible":False,
                 "memory_readout":visible_readout()},
            ],
        }
        changed={
            **base,"correct_side":"lower","cue":"key","success":0,
            "trace":[dict(t) for t in base["trace"]],
        }
        item=assess_pair({"pair":[base,changed]})
        self.assertTrue(item["both_cues_observed"])
        self.assertTrue(item["same_f4_readout_at_first_cue"])
        self.assertTrue(item["both_object_types_in_f4_at_first_cue"])
        self.assertTrue(item["same_f4_readout_before_final_action"])
        self.assertTrue(item["same_complete_f4_readout_sequence"])
        self.assertTrue(item["same_action_and_position_sequence"])
        self.assertFalse(item["both_correct"])
        s=summarize([item])
        self.assertEqual(s["paired_worlds"],1)
        self.assertEqual(s["both_object_types_present_at_first_cue"],1)
        self.assertEqual(s["both_conditions_successful"],0)

    def test_pair_consistency_and_changed_readout_detected(self):
        a={"seed":100000,"cue":"ball","correct_side":"upper",
           "side":"upper","success":1,
           "trace":[{"action":0,"position":[1,3],
                     "cue_visible":True,"memory_readout":visible_readout()}]}
        different=visible_readout()
        different[0:4]=[0.,1.,.2,1.]
        b={**a,"cue":"key","correct_side":"lower","side":"lower",
           "trace":[{**a["trace"][0],"memory_readout":different}]}
        result=assess_pair({"pair":[a,b]})
        self.assertFalse(result["same_f4_readout_at_first_cue"])
        self.assertFalse(result["same_f4_readout_before_final_action"])
        self.assertFalse(result["same_complete_f4_readout_sequence"])
        bad={**b,"correct_side":"upper"}
        with self.assertRaises(AssertionError):
            assess_pair({"pair":[a,bad]})


if __name__=="__main__":
    unittest.main()
