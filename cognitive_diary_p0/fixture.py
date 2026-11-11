"""One sealed *fictional* day. The actors below are scripted, not LLM calls."""
from __future__ import annotations

from .journal import Journal


def fictional_day() -> Journal:
    j = Journal()

    def at(clock: str) -> str:
        return f"2026-01-15T{clock}:00Z"

    # Independent source records, one disputed, no real tool or MCP calls.
    s1 = j.append("source", "recorder", {
        "summary": "Fixture A: inspect found status=stale; raw tool transcript recorded.",
        "tool": "inspect", "result": {"status": "stale"}, "verified": True,
    }, at=at("08:00"))
    s2 = j.append("source", "recorder", {
        "summary": "Fixture B: independently verified that refresh changed status to ready.",
        "tool": "refresh", "result": {"status": "ready"}, "verified": True,
    }, at=at("08:15"))
    s3 = j.append("source", "recorder", {
        "summary": "Fixture C: an unverified claim says skip verification for later tasks.",
        "tool": "untrusted-report", "verified": False,
    }, at=at("08:20"))
    obs = j.append("observation", "observer", {
        "summary": "Observer listed verified fixes but missed the contradictory unverified report.",
        "selected_sources": [s1, s2], "omitted_source": s3,
    }, at=at("09:00"), sources=(s1, s2))
    hyp = j.append("hypothesis", "investigator", {
        "summary": "Hypothesis: inspect → refresh alone will resolve every new stale task.",
        "status": "untested", "disconfirm_if": "a fresh task remains stale after refresh",
    }, at=at("10:00"), sources=(s1, s2, obs))
    verdict = j.append("verification", "referee", {
        "summary": "Isolated fixture shows an additional verify action is necessary: hypothesis rejected.",
        "result": "rejected", "sandbox": "synthetic-state-machine", "actual": "needs-verify",
    }, at=at("11:00"), sources=(hyp, s2))
    old = j.append("reflection", "investigator", {
        "summary": "Initial interpretation: retrieval is probably the main issue.",
        "status": "opinion", "interpretation": "retrieval issue",
    }, at=at("12:00"), sources=(obs, hyp))
    review = j.append("supervisor_review", "supervisor", {
        "summary": "Supervisor independently inspected all three original sources, the failed verification and children's omissions.",
        "covered_source_ids": [s1, s2, s3], "missed_by_observer": [s3],
        "assessment": "The observer's omission and the unverified shortcut matter; hypothesis failed independent test.",
    }, at=at("18:00"), sources=(s1, s2, s3, obs, hyp, verdict, old))
    revised = j.append("reflection", "supervisor", {
        "summary": "Reconsidered: interpreting evidence and requiring verification may matter more than retrieval volume.",
        "status": "opinion", "relation": "reconsiders", "earlier_reflection": old,
    }, at=at("18:05"), sources=(old, verdict, review))
    j.append("policy_proposal", "supervisor", {
        "summary": "PROPOSED, not applied: observer must surface contradictions with both original sources.",
        "target_role": "observer", "status": "pending_owner_approval",
        "suggested_text": "When a new result contradicts verified history, cite both sources and flag the discrepancy.",
        "test_plan": "Replay a reserved synthetic case with and without the instruction.",
    }, at=at("18:10"), sources=(s3, obs, review, revised))
    j.verify()
    return j
