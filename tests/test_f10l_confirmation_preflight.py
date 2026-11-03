"""Pre-confirm PRE-FLIGHT: must NEVER instantiate any 23300000+ worlds.

Validates byte-for-byte equivalence of NEW confirmation generator against
the FROZEN original algorithm when fed ALREADY-OPEN OOD 23400000+ seeds.
"""
from __future__ import annotations
import unittest
import numpy as np

from observatory.f10l_worlds import select_world,VARIANTS
from observatory.f10l_confirm_worlds import build_world,CONFIRM_BASE,N_WORLDS

class ConfirmationGeneratorIntegrity(unittest.TestCase):
    def test_namespace_reserved_only_for_confirmation(self):
        self.assertEqual(CONFIRM_BASE,23_300_000)
        self.assertEqual(N_WORLDS,128)
        self.assertEqual(set(VARIANTS),
                         {"U_before","U_after","S_before","S_after"})
        for request in ((-1,0,"ood"),(0,128,"ood"),(4,0,"ood"),
                        (0,0,"train"),(0,0,"dev"),(0,0,"invalid")):
            rep,j,split=request
            with self.assertRaises(ValueError):
                build_world(23400000,rep,j,split)

    def test_bit_exact_original_old_ood_all_worlds_and_replicas(self):
        # The only seeds used below are *already opened* F10-L OOD,
        # NEVER original 23300000 confirmation namespace.
        keys=("task","split","replica","seed","base_world","variant",
              "source_mode","n","signed_count","winner","status","hash")
        count=0
        for rep in range(4):
            for j in range(128):
                original=select_world("ood",rep,j)
                old_seed=23_400_000+10_000*rep+j
                duplicate=build_world(old_seed,rep,j,"ood")
                self.assertEqual(len(original),len(duplicate))
                for a,b in zip(original,duplicate):
                    for key in keys:
                        self.assertEqual(a[key],b[key],(rep,j,key))
                    for key in ("features","contents","mask"):
                        self.assertEqual(a[key].dtype,b[key].dtype)
                        np.testing.assert_array_equal(a[key],b[key])
                    count+=1
        self.assertEqual(count,2048)

if __name__=="__main__":
    unittest.main()
