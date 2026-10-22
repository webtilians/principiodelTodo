"""Anonymize run identifiers before feature inspection.

The private mapping is written with permissions 0600; never commit it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from pathlib import Path
from typing import Mapping, Sequence

from .core import Point, save_json


def _write_private(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            json.dump(payload, out, indent=2, ensure_ascii=False)
            out.write("\n")
    except BaseException:
        target.unlink(missing_ok=True)
        raise


def blind_runs(
    runs: Mapping[str, Sequence[Point]],
    *,
    output_path: str | Path,
    key_file: str | Path,
) -> dict:
    """Release only aliased IDs, steps, and metrics.

    No conditions, seeds, source paths, free-text notes, or metadata are exported.
    Caller must ensure metric names themselves do not expose experiment conditions.
    """
    output_path, key_file = Path(output_path), Path(key_file)
    if output_path.resolve() == key_file.resolve():
        raise ValueError("output and private mapping must be different paths")
    if output_path.exists() or key_file.exists():
        raise FileExistsError("refusing to overwrite previous blind data or mapping")
    key = secrets.token_bytes(32)
    aliases = {
        run_id: "case_" + hmac.new(key, run_id.encode(), hashlib.sha256).hexdigest()[:16]
        for run_id in sorted(runs)
    }
    if len(set(aliases.values())) != len(aliases):
        raise ValueError("case identifier collision")
    payload = {
        "schema": "emergence_observatory_blind_v0.1",
        "secret_hex": key.hex(),
        "alias_to_original": {alias: run_id for run_id, alias in aliases.items()},
        "instructions": "KEEP PRIVATE. Decode only after the analysis protocol and findings are frozen.",
    }
    _write_private(key_file, payload)
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("x", encoding="utf-8") as out:
            for run_id, points in sorted(runs.items()):
                for point in points:
                    out.write(json.dumps(
                        {"run_id": aliases[run_id], "step": point.step, "metrics": point.metrics},
                        sort_keys=True, ensure_ascii=False,
                    ) + "\n")
    except BaseException:
        output_path.unlink(missing_ok=True)
        key_file.unlink(missing_ok=True)
        raise
    return {"cases": len(aliases), "observations": sum(map(len, runs.values()))}
