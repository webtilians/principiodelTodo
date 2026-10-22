"""INFINITO emergence observatory: preregistered, descriptive time-series probes.

This module measures candidate temporal transitions, not consciousness.
No SciPy/PyTorch dependencies and no mutation of the original experiments.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

SCHEMA_VERSION = "emergence_observatory_v0.1"


@dataclass(frozen=True)
class Point:
    run_id: str
    step: int
    metrics: Dict[str, float]


@dataclass(frozen=True)
class Candidate:
    run_id: str
    metric: str
    step: int
    delta: float
    effect_size: float
    p_value: float
    q_value: float
    flagged: bool
    label: str = "unclassified_transition"


def parse_points(rows: Iterable[Mapping]) -> Dict[str, List[Point]]:
    """Validate scalar observations. A row is a snapshot, not a diagnosis."""
    runs: Dict[str, List[Point]] = defaultdict(list)
    for number, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("row %d must be an object" % number)
        name = row.get("run_id")
        step = row.get("step")
        metrics = row.get("metrics")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("row %d: invalid run_id" % number)
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError("row %d: step must be a nonnegative integer" % number)
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("row %d: metrics must be a nonempty object" % number)
        clean = {}
        for metric, value in metrics.items():
            if not isinstance(metric, str) or not metric.strip():
                raise ValueError("row %d: invalid metric name" % number)
            if isinstance(value, bool) or not isinstance(value, (float, int)):
                raise ValueError("row %d: %s must be numeric" % (number, metric))
            if not math.isfinite(value):
                raise ValueError("row %d: %s must be finite" % (number, metric))
            clean[metric] = float(value)
        runs[name].append(Point(name, step, clean))
    for name, points in runs.items():
        points.sort(key=lambda point: point.step)
        steps = [point.step for point in points]
        if len(steps) != len(set(steps)):
            raise ValueError("run %s has duplicate steps" % name)
    return dict(sorted(runs.items()))


def load_jsonl(path: str | Path) -> Dict[str, List[Point]]:
    with Path(path).open("r", encoding="utf-8") as source:
        rows = []
        for number, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError("invalid JSON on line %d" % number) from exc
    return parse_points(rows)


def _max_window_shift(values: Sequence[float], window: int):
    """Maximum adjacent-window mean difference; scans all eligible boundaries."""
    if len(values) < 2 * window:
        return None
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    best = (0.0, window, 0.0)
    for split in range(window, len(values) - window + 1):
        before = (prefix[split] - prefix[split - window]) / window
        after = (prefix[split + window] - prefix[split]) / window
        delta = after - before
        if abs(delta) > best[0]:
            best = (abs(delta), split, delta)
    return best


def _adjust_bh(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg adjustment across metrics within a run."""
    n = len(p_values)
    result = [1.0] * n
    minimum = 1.0
    for rank in range(n, 0, -1):
        index = sorted(range(n), key=lambda i: (p_values[i], i))[rank - 1]
        minimum = min(minimum, p_values[index] * n / rank)
        result[index] = minimum
    return result


def analyze(
    runs: Mapping[str, Sequence[Point]],
    *,
    window: int = 6,
    permutations: int = 199,
    alpha: float = 0.05,
    min_effect: float = 1.0,
    seed: int = 42,
    recurrence_tolerance: int = 2,
    min_replicates: int = 2,
) -> dict:
    """Flag candidate shifts against a shuffled-time *descriptive* null.

    Shuffling destroys temporal order; it does not preserve autocorrelation.
    Significant scores do not establish causality, adaptation, or consciousness.
    """
    if window < 2 or permutations < 19 or not (0 < alpha < 1):
        raise ValueError("window >= 2, permutations >= 19 and 0 < alpha < 1 required")
    if min_effect < 0 or recurrence_tolerance < 0 or min_replicates < 2:
        raise ValueError("invalid effect/recurrence parameters")
    findings: List[Candidate] = []
    excluded = []
    for run_id, points in sorted(runs.items()):
        metrics = sorted(set.intersection(*(set(point.metrics) for point in points))) if points else []
        if len(points) < 2 * window:
            excluded.append({"run_id": run_id, "reason": "too_short"})
            continue
        provisional = []
        for metric in metrics:
            values = [point.metrics[metric] for point in points]
            best = _max_window_shift(values, window)
            if best is None:
                continue
            score, split, delta = best
            spread = statistics.pstdev(values)
            effect = score / spread if spread > 0 else 0.0
            salt = hashlib.sha256(("%s|%s|%s" % (seed, run_id, metric)).encode()).digest()
            rng = random.Random(int.from_bytes(salt[:8], "big"))
            exceedances = 0
            for _ in range(permutations):
                shuffled = values.copy()
                rng.shuffle(shuffled)
                null_best = _max_window_shift(shuffled, window)
                if null_best is not None and null_best[0] >= score - 1e-12:
                    exceedances += 1
            p_value = (1 + exceedances) / (1 + permutations)
            provisional.append((metric, points[split].step, delta, effect, p_value))
        q_values = _adjust_bh([row[4] for row in provisional])
        for row, q_value in zip(provisional, q_values):
            metric, at_step, delta, effect, p_value = row
            findings.append(Candidate(
                run_id=run_id, metric=metric, step=at_step, delta=delta,
                effect_size=effect, p_value=p_value, q_value=q_value,
                flagged=q_value <= alpha and effect >= min_effect,
            ))
    recurring = _recurrence(findings, recurrence_tolerance, min_replicates)
    return {
        "schema_version": SCHEMA_VERSION,
        "method": "maximum_adjacent_window_mean_shift_vs_within_run_time_shuffle",
        "protocol": {
            "window": window, "permutations": permutations, "alpha": alpha,
            "min_effect": min_effect, "seed": seed,
            "recurrence_tolerance": recurrence_tolerance,
            "min_replicates": min_replicates,
            "multiple_testing": "BH per run across observed metrics",
        },
        "run_count": len(runs),
        "observations": sum(len(points) for points in runs.values()),
        "candidates": [asdict(candidate) for candidate in findings],
        "recurrences": recurring,
        "excluded": excluded,
        "interpretation": (
            "Candidate temporal shifts only. Permutation null breaks serial dependence; "
            "autocorrelation, interventions, held-out runs and independent replication "
            "must be assessed before attributing mechanisms. Not evidence of consciousness."
        ),
    }


def _recurrence(findings: Sequence[Candidate], tolerance: int, min_replicates: int) -> List[dict]:
    """Descriptive grouping of flagged events by metric, direction and step."""
    buckets = defaultdict(list)
    for item in findings:
        if item.flagged:
            buckets[(item.metric, "up" if item.delta > 0 else "down")].append(item)
    groups = []
    for (metric, direction), records in sorted(buckets.items()):
        records = sorted(records, key=lambda item: (item.step, item.run_id))
        clusters: List[List[Candidate]] = []
        for item in records:
            if clusters and item.step - clusters[-1][0].step <= tolerance:
                clusters[-1].append(item)
            else:
                clusters.append([item])
        for cluster in clusters:
            ids = sorted(set(item.run_id for item in cluster))
            if len(ids) >= min_replicates:
                groups.append({
                    "metric": metric, "direction": direction,
                    "runs": ids, "n_runs": len(ids),
                    "steps": sorted(set(item.step for item in cluster)),
                    "status": "recurrent_candidate_not_causal",
                })
    return groups


def save_json(path: str | Path, data: Mapping):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as out:
        json.dump(data, out, indent=2, ensure_ascii=False, sort_keys=True)
        out.write("\n")
