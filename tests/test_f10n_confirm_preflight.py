"""F10-N-C PRE-CONFIRM ONLY: never instantiate a 26300000+ world here."""
from __future__ import annotations

import unittest
import numpy as np

from observatory import f10n_worlds as original
from observatory.f10n_confirm_worlds import (
    build_world,old_ood_regression_cases,CONFIRM_BASE,OOD_BASE,N_WORLDS
)

class F10NConfirmPreflight(unittest.TestCase):
    def test_confirm_is_sealed_for_original_f10n_and_namespace_is_fixed(self):
        self.assertEqual(CONFIRM_BASE,26300000)
        self.assertEqual(OOD_BASE,26400000)
        self.assertEqual(N_WORLDS,128)
        for args in (
            ("confirm",0,0,("MEMORY",1)),
            ("train",0,0,("MEMORY",1)),
            ("dev",0,0,("MEMORY",1)),
        ):
            with self.assertRaises(ValueError):
                original.seed_for(*args)
        for params in (
            (26300000,0,0,"ood"),
            (26400000,0,0,"confirm"),
            (26400000,4,0,"ood"),
            (26400000,0,128,"ood"),
            (26400000,0,0,"train"),
        ):
            with self.assertRaises(ValueError):
                build_world(*params)

    def test_all_2048_already_open_ood_cases_identical_byte_by_byte(self):
        count=0
        fingerprints=set()
        for rep in range(4):
            reference=original.dataset("ood",rep)
            duplicate=old_ood_regression_cases(rep)
            self.assertEqual(len(reference),512)
            self.assertEqual(len(duplicate),512)
            for a,b in zip(reference,duplicate):
                self.assertEqual(set(a),set(b))
                for key in a:
                    if isinstance(a[key],np.ndarray):
                        self.assertEqual(a[key].dtype,b[key].dtype)
                        np.testing.assert_array_equal(a[key],b[key])
                        self.assertEqual(a[key].tobytes(),b[key].tobytes())
                    else:
                        self.assertEqual(a[key],b[key],(rep,key))
                self.assertNotIn(a["case_sha"],fingerprints)
                fingerprints.add(a["case_sha"])
                self.assertEqual(a["split"],"ood")
                self.assertEqual(a["task"],"MEMORY")
                self.assertEqual(a["flip"],1)
                count+=1
        self.assertEqual(count,2048)
        self.assertEqual(len(fingerprints),2048)

if __name__=="__main__":
    unittest.main()
