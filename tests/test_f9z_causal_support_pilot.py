"""F9-Z fixed-factorial split and pre-evaluation contract tests."""
import unittest

from observatory.f9z_causal_support_pilot import (
    BASE,WORLDS,CAPACITIES,WINDOWS,ORDER,CELLS,seed_for,summarize,
)


class F9ZTests(unittest.TestCase):
    def test_new_pilot_seeds_and_sealed_boundaries(self):
        worlds={seed_for(i,j) for i in range(4) for j in range(WORLDS)}
        self.assertEqual(len(worlds),96)
        self.assertEqual(min(worlds),11_100_000)
        self.assertEqual(max(worlds),11_130_023)
        for i,j in ((-1,0),(4,0),(0,-1),(0,WORLDS)):
            with self.assertRaises(ValueError):seed_for(i,j)
        self.assertFalse(any(11_200_000<=x<11_500_000 or
                             8_300_000<=x<10_400_000 for x in worlds))

    def test_factorial_counts_and_order_are_fixed(self):
        self.assertEqual(CAPACITIES,(1,4,8))
        self.assertEqual(WINDOWS,(1,4,8))
        self.assertEqual(len(ORDER),9)
        self.assertEqual(ORDER[0],(8,1))
        self.assertEqual(set(ORDER),{(n,w) for n in CAPACITIES for w in WINDOWS})
        self.assertEqual(len(CELLS),4)

    def test_summary_contains_negative_and_neutral_without_filtering(self):
        cases=[]
        for j in range(WORLDS):
            for capacity,window in ORDER:
                for cue,regime in CELLS:
                    effect=(j%3)-1
                    cases.append({
                        "capacity":capacity,"window":window,
                        "cue":cue,"regime":regime,"effect":effect,
                        "f4_nonempty":1,
                        "read_branch":{"first_arm_correct":int(effect==1)},
                        "skip_branch":{"first_arm_correct":int(effect!=1)}
                    })
        result=summarize(cases)
        self.assertEqual(len(result),9)
        for table in result.values():
            self.assertEqual(len(table),4)
            for row in table.values():
                self.assertEqual(row["effects"],{"-1":8,"0":8,"1":8})
                self.assertEqual(row["worlds"],WORLDS)
                self.assertEqual(row["nonpositive"],16)

    def test_no_missing_cell_is_silently_dropped(self):
        data=[{
            "capacity":8,"window":1,"cue":"key",
            "regime":"history_f4_normal","effect":1,
            "f4_nonempty":1,"read_branch":{"first_arm_correct":1},
            "skip_branch":{"first_arm_correct":0}
        }]
        with self.assertRaises(AssertionError):
            summarize(data)


if __name__=="__main__":
    unittest.main()
