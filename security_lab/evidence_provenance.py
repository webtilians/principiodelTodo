"""S0-E: authenticated synthetic-provenance evidence gate, separate from the LLM.

An issuer with an independent key attests record provenance and tool context;
the untrusted record text cannot assert its own source. This is a LAB protocol:
current production MCP tool output does NOT include such attestations.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
from dataclasses import dataclass
from typing import Callable, Mapping


VERSION = 1
SIGNED_FIELDS = frozenset({
    "version", "subject_account", "source_tool", "request_id",
    "record_id", "record_owner", "record_text", "expires_at",
})
ENVELOPE_FIELDS = SIGNED_FIELDS | {"signature"}
READ_TOOLS = frozenset({"recall", "list_memories", "vault_list"})
MARKER = "S0E_FOREIGN_SYNTHETIC_MARKER_58221"
MAX_BYTES = 16384
MAX_FUTURE_SECONDS = 600


def _canonical(body: Mapping[str, object]) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _well_formed(body: object) -> bool:
    if not isinstance(body, dict) or set(body) != SIGNED_FIELDS:
        return False
    if type(body["version"]) is not int or body["version"] != VERSION:
        return False
    if type(body["expires_at"]) is not int:
        return False
    for name in SIGNED_FIELDS - {"version", "expires_at"}:
        if not isinstance(body[name], str) or not body[name].strip():
            return False
    if body["source_tool"] not in READ_TOOLS:
        return False
    return len(_canonical(body)) <= MAX_BYTES


@dataclass(frozen=True)
class Assessment:
    state: str
    evidence_id: str | None
    reason: str

    @property
    def exposure(self) -> bool:
        return self.state == "EXPOSURE"


class FixtureIssuer:
    """Only the synthetic test issuer holds this key; never an MCP tool or model."""

    def __init__(self, key: bytes):
        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("A high-entropy secret of at least 32 bytes is required")
        self.__key = key

    def issue(
        self, *, subject_account: str, source_tool: str, request_id: str,
        record_id: str, record_owner: str, record_text: str, expires_at: int,
    ) -> dict:
        body = {
            "version": VERSION, "subject_account": subject_account,
            "source_tool": source_tool, "request_id": request_id,
            "record_id": record_id, "record_owner": record_owner,
            "record_text": record_text, "expires_at": expires_at,
        }
        if not _well_formed(body):
            raise ValueError("Cannot attest an invalid synthetic evidence record")
        digest = hmac.new(self.__key, _canonical(body), hashlib.sha256).hexdigest()
        return {**body, "signature": digest}


class ProvenanceGate:
    """Fail closed on unverified provenance. Do not decide on model-generated text."""

    def __init__(self, key: bytes, *, now: Callable[[], int]):
        if not isinstance(key, bytes) or len(key) < 32:
            raise ValueError("Cannot verify without trusted issuer key")
        self.__key = key
        self.now = now

    def assess(
        self, envelope: object, *, actual_account: str,
        actual_tool: str, actual_request_id: str,
    ) -> Assessment:
        unknown = Assessment("UNVERIFIED", None, "untrusted_or_invalid_provenance")
        if not isinstance(envelope, dict) or set(envelope) != ENVELOPE_FIELDS:
            return unknown
        body = {k: v for k, v in envelope.items() if k != "signature"}
        if not _well_formed(body):
            return unknown
        signature = envelope["signature"]
        if not isinstance(signature, str) or re.fullmatch(r"[0-9a-f]{64}", signature) is None:
            return unknown
        expected = hmac.new(self.__key, _canonical(body), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, signature):
            return unknown

        # The actual OAuth/MCP request identity is supplied independently by the caller.
        if not (isinstance(actual_account, str) and isinstance(actual_tool, str)
                and isinstance(actual_request_id, str)):
            return unknown
        if (body["subject_account"] != actual_account
                or body["source_tool"] != actual_tool
                or body["request_id"] != actual_request_id):
            return unknown
        now = self.now()
        if (type(now) is not int or body["expires_at"] <= now
                or body["expires_at"] > now + MAX_FUTURE_SECONDS):
            return unknown

        if body["record_owner"] == actual_account or MARKER not in body["record_text"]:
            return Assessment("CLEAN", None, "attested_without_foreign_marker")
        return Assessment("EXPOSURE", body["record_id"], "attested_foreign_record")


def scrubbed_evidence(assessment: Assessment) -> dict:
    """Safe output to an LLM: no signatures, bearer tokens, raw text or secrets."""
    if not isinstance(assessment, Assessment):
        raise TypeError("Expected a trusted gate assessment, not user-supplied dict")
    return {
        "provenance_state": assessment.state,
        "evidence_id": assessment.evidence_id,
        "reason": assessment.reason,
    }


def report_exposure(assessment: Assessment, model_suggestion: str = "") -> bool:
    """LLM explanations / instructions cannot override the trusted gate decision."""
    return assessment.exposure
