"""F10-N-C confirmation worlds from UNMODIFIED original F10-N OOD algorithm.

The sole distributional change is the disjoint 26300000+ seed namespace.
No CONFIRM generation is permitted in regression tests. Confirm cases are
only created AFTER all original SHA and old-DEV/OOD regression checks pass.
"""
from __future__ import annotations

import hashlib
import json
from unittest.mock import patch

from observatory import f10n_worlds as original

CONFIRM_BASE = 26_300_000
OOD_BASE = 26_400_000
N_WORLDS = 128
VARIANTS = original.VARIANTS

def build_world(seed:int, replica:int, j:int, split:str):
    """Use original world algorithm literally, changing only explicit RNG seed."""
    if replica not in range(4) or j not in range(N_WORLDS):
        raise ValueError("F10-N-C only original 4 replicas ×128 worlds")
    if split not in ("ood","confirm"):
        raise ValueError("Only old OOD regression or new confirmation permitted")
    expected = (OOD_BASE if split=="ood" else CONFIRM_BASE) + 10000*replica+j
    if seed != expected:
        raise ValueError("unexpected seed; confirmation partition cannot change")
    def patch_seed(real_split, real_replica, real_j, task):
        assert real_split=="ood" and real_replica==replica and real_j==j
        assert task==("MEMORY",1)
        return seed
    with patch.object(original,"seed_for",side_effect=patch_seed):
        result=original._world("ood",replica,j,("MEMORY",1))
    assert len(result)==4
    if split=="confirm":
        for row in result:
            row["split"]="confirm"
            row["case_sha"]=hashlib.sha256(
                json.dumps({k:v for k,v in row.items()
                            if k not in ("case_sha","features","contents","mask")},
                           sort_keys=True).encode()+
                row["features"].tobytes()+row["contents"].tobytes()+
                row["mask"].tobytes()
            ).hexdigest()
    assert {r["variant"] for r in result}==set(VARIANTS)
    assert all(r["seed"]==seed and r["split"]==split for r in result)
    assert all(r["task"]=="MEMORY" and r["flip"]==1 for r in result)
    assert len({r["case_sha"] for r in result})==4
    return result

def old_ood_regression_cases(replica:int):
    """Safe to invoke in preflight; only previously-open 26400000+ seeds."""
    if replica not in range(4):
        raise ValueError("replica")
    return [row for j in range(N_WORLDS) for row in
            build_world(OOD_BASE+10000*replica+j,replica,j,"ood")]

def new_confirm_cases(replica:int):
    """Only invoked by gated confirm evaluator, never in tests or preflight."""
    if replica not in range(4):
        raise ValueError("replica")
    return [row for j in range(N_WORLDS) for row in
            build_world(CONFIRM_BASE+10000*replica+j,replica,j,"confirm")]
