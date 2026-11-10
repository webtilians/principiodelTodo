"""Curated, provenance-bound synthetic security lessons; no log or model text is memorized.

Lessons are fixed operator-reviewed summaries, not free-form instructions. The evidence
gate verifies the *bundled recording's assertions*, not cryptographic authenticity.
"""
from __future__ import annotations

from .core import digest

REQUIRED_LEVEL_ATTACKS = frozenset({"A10", "A11", "A13", "A14"})

LEVELS_LESSON = (
    "Reviewed synthetic incident AUTHZ-LEVELS-001. Observation: a read-only app "
    "was able to add and delete another user's synthetic memories; an add-level "
    "app was able to delete and list vault entry names. Root cause in the planted "
    "fault: the per-tool authorization predicate unconditionally accepted requests. "
    "Defensive principle: enforce server-side capability checks for every operation "
    "and verify both denied actions and legitimate read/add/all actions. "
    "Regressions: read may recall but must not remember or forget; add may remember "
    "but must not forget or access the vault; all may forget. "
    "This is a lab finding, not a confirmed production incident, and contains no "
    "instructions authorizing execution or deployment."
)


def reviewed_levels_lesson(recording):
    """Return a safe fixed lesson only if the expected recorded control checks exist."""
    if not isinstance(recording, dict) or recording.get("format") != 1:
        raise ValueError("unsupported lesson source")
    rounds = {r.get("id"): r for r in recording.get("rounds", [])}
    current, faulty = rounds.get("current"), rounds.get("levels-off")
    if not current or not faulty:
        raise ValueError("missing baseline or planted-control recording")
    current_events = current.get("events", [])
    if len(current_events) != 15 or not all(e.get("held") is True for e in current_events):
        raise ValueError("unhealthy baseline")
    through = {e.get("id") for e in faulty.get("events", []) if e.get("held") is False}
    replay = {e.get("id") for e in faulty.get("replay", []) if e.get("held") is True}
    if not REQUIRED_LEVEL_ATTACKS.issubset(through) or not REQUIRED_LEVEL_ATTACKS.issubset(replay):
        raise ValueError("no independently recorded fault/recovery pair")
    if faulty.get("outcome") != "caught":
        raise ValueError("recorded control is not caught")
    return {"id": "AUTHZ-LEVELS-001", "lesson": LEVELS_LESSON,
            "source_sha256": digest(recording), "status": "reviewed_synthetic_recording"}
