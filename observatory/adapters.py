"""Read historical INFINITO logs or scalar CSVs without importing simulations."""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import List

from .core import parse_points

_FLOAT = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
_SIMPLE = re.compile(
    r"Recursion\s+(\d+):\s+Clusters\s+(\d+),\s+Entropy\s+(" + _FLOAT +
    r"),\s+Loss\s+(" + _FLOAT + r")"
)
_OPTIMIZED = re.compile(
    r"R\s*(\d+):\s*C\s*(\d+)\s*\|\s*E\s*(" + _FLOAT + r")\s*\|\s*L\s*(" + _FLOAT + r")"
)


def parse_legacy_log(source: str, *, run_id: str) -> List[dict]:
    """Extract logged scalar values only; no semantic reinterpretation."""
    rows = []
    for line in source.splitlines():
        match = _SIMPLE.search(line) or _OPTIMIZED.search(line)
        if not match:
            continue
        step, clusters, ent, loss = match.groups()
        rows.append({
            "run_id": run_id,
            "step": int(step),
            "metrics": {
                "clusters_reported": float(clusters),
                "entropy_reported": float(ent),
                "loss_reported": float(loss),
            },
        })
    if not rows:
        raise ValueError("no recognizable recursion log lines found")
    # Reject repeated steps: the log may contain several concatenated runs.
    parse_points(rows)
    return rows


def read_scalar_csv(path: str | Path, *, run_id: str, step_column: str = "") -> List[dict]:
    """Convert one time-series CSV to an observer-friendly JSONL schema."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        names = reader.fieldnames or []
        column = step_column or ("step" if "step" in names else "recursion")
        if column not in names:
            raise ValueError("CSV requires a step/recursion column, or --step-column")
        rows = []
        for number, row in enumerate(reader, 2):
            try:
                step = int(row[column])
            except (TypeError, ValueError) as exc:
                raise ValueError("invalid step on CSV line %d" % number) from exc
            metrics = {}
            for metric, value in row.items():
                if metric == column or not metric or value is None or not value.strip():
                    continue
                try:
                    metrics[metric] = float(value)
                except ValueError:
                    # Textual columns are metadata, not measurements.
                    pass
            if metrics:
                rows.append({"run_id": run_id, "step": step, "metrics": metrics})
    if not rows:
        raise ValueError("CSV has no numeric observations")
    parse_points(rows)
    return rows


def export_jsonl(path: str | Path, rows: List[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as dest:
        for row in rows:
            dest.write(json.dumps(row, sort_keys=True) + "\n")
