"""Pre-confirm gate: sources immutable, new seeds and no retraining."""
import unittest
from unittest.mock import patch
from pathlib import Path

from observatory.f9v_confirm_lexicographic_causal_f4 import (
    confirmation_seed,CONFIRM_BASE,CONFIRM_PAIRS,validate_lock,
)


class F9VConfirmationContracts(unittest.TestCase):
    def test_sealed_seed_range_disjoint(self):
        seeds={confirmation_seed(i,j) for i in range(4) for j in range(CONFIRM_PAIRS)}
        self.assertEqual(len(seeds),160)
        self.assertTrue(all(8_300_000<=s<8_340_000 for s in seeds))
        for index,j in ((4,0),(-1,0),(0,-1),(0,40)):
            with self.assertRaises(ValueError):
                confirmation_seed(index,j)

    def test_lock_has_all_frozen_checkpoint_ids_and_sha(self):
        import json
        obj=json.loads(Path(
            "docs/INFINITO_F9V_CONFIRMATION_LOCK.json").read_text())
        self.assertEqual(obj["dev_gate_passed"],True)
        self.assertEqual(obj["confirm_cannot_train"],True)
        self.assertEqual(obj["split"],{"base":8_300_000,"per_replica":40,"replicas":4})
        self.assertEqual(set(obj["weights"]),{"0","1","2","3"})
        for i in range(4):
            item=obj["weights"][str(i)]
            self.assertEqual(set(item["heads"]),{"lex_hc","lex_public","scalar_hc"})
            for digest in (item["actor_sha256"],item["report_sha256"],*item["heads"].values()):
                self.assertEqual(len(digest),64)
                self.assertTrue(all(c in "0123456789abcdef" for c in digest))

    def test_lock_gate_rejects_wrong_original_actor(self):
        with self.assertRaises(AssertionError):
            validate_lock(0,Path("/nonexistent"),"0"*64)

if __name__=="__main__":
    unittest.main()
