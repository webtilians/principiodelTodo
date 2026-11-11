"""Synthetic-only append-only audit metadata + separately erasable content store.

NOT production storage, encryption, authentication or a claim of cryptographic
non-repudiation. No live users, providers, secrets or model in this package.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from typing import Any

SCHEMA = "infinito.cognitive-diary-p0/1"
GENESIS = "0" * 64
KINDS = {
    "source", "observation", "hypothesis", "verification", "reflection",
    "supervisor_review", "policy_proposal", "policy_decision", "policy_applied", "forget",
}
ROLES = {"recorder", "observer", "investigator", "referee", "supervisor", "owner", "system"}


class AuditError(ValueError):
    """The supplied audit trail or proposed change is invalid."""


def canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


class Journal:
    """Immutable, hash-linked event headers and separately erasable payloads.

    Only fixed, fictitious identifiers may enter this P0 demonstration. Headers
    keep hashes and references, NOT the words of a memory/reflection. Content
    lives in ``bodies`` and may be physically removed, including derivatives.
    The plain SHA-256 hashes here are NOT suitable for private low-entropy facts:
    production needs keyed digests, authenticated encryption and key management.
    """

    def __init__(self, owner: str = "fictional-user", scope: str = "fictional-lab"):
        self.owner, self.scope = owner, scope
        self.entries: list[dict] = []
        self.bodies: dict[str, dict] = {}

    def _by_id(self, event_id: str) -> dict:
        for entry in self.entries:
            if entry["id"] == event_id:
                return entry
        raise AuditError(f"unknown reference: {event_id}")

    def _erased(self) -> set[str]:
        gone: set[str] = set()
        for entry in self.entries:
            if entry["kind"] == "forget":
                body = self.bodies.get(entry["id"])
                if body is None:
                    raise AuditError("missing forget marker")
                gone.update(body["erased_ids"])
        return gone

    def append(
        self, kind: str, actor: str, body: dict, *,
        at: str, sources: tuple[str, ...] = (), model: str = "deterministic-fixture",
        owner: str | None = None, scope: str | None = None,
        input_tokens: int = 0, output_tokens: int = 0, cost_usd: float = 0.0,
    ) -> str:
        if kind not in KINDS or actor not in ROLES:
            raise AuditError("unknown record kind or actor")
        if owner is not None and owner != self.owner or scope is not None and scope != self.scope:
            raise AuditError("cross-tenant or cross-project record refused")
        if not isinstance(body, dict) or not isinstance(at, str) or not at.endswith("Z"):
            raise AuditError("invalid body or UTC timestamp")
        if input_tokens < 0 or output_tokens < 0 or cost_usd < 0 or any(isinstance(x, bool) for x in (input_tokens, output_tokens)):
            raise AuditError("invalid tokens or cost")
        if kind == "verification" and actor != "referee":
            raise AuditError("only the independent referee may record verification")
        if kind in ("supervisor_review", "policy_proposal") and actor != "supervisor":
            raise AuditError("supervisor-only action")
        if kind == "forget":
            raise AuditError("use forget() so derivatives are erased")
        if kind == "policy_decision":
            if actor != "owner" or len(sources) != 1 or self._by_id(sources[0])["kind"] != "policy_proposal" or body.get("decision") not in ("approved", "rejected"):
                raise AuditError("a policy decision requires an owner and an existing proposal")
        if kind == "policy_applied":
            if actor != "system" or len(sources) != 1 or self._by_id(sources[0])["kind"] != "policy_decision":
                raise AuditError("policy application needs a user decision")
            if self.bodies.get(sources[0], {}).get("decision") != "approved":
                raise AuditError("policy is not user-approved")
        if not isinstance(sources, (tuple, list)) or len(set(sources)) != len(sources):
            raise AuditError("invalid source references")
        for source in sources:
            self._by_id(source)
            if source in self._erased():
                raise AuditError("forgotten content cannot be reused")
        payload = deepcopy(body)
        # Verify JSON serializability before changing any state.
        canonical(payload)
        event_id = f"e{len(self.entries) + 1:04d}"
        previous_hash = self.entries[-1]["hash"] if self.entries else GENESIS
        header = {
            "id": event_id, "kind": kind, "actor": actor, "at": at,
            "owner": self.owner, "scope": self.scope, "sources": list(sources),
            "model": model, "input_tokens": input_tokens, "output_tokens": output_tokens,
            "cost_usd": cost_usd, "payload_digest": sha(payload), "previous_hash": previous_hash,
        }
        header["hash"] = sha(header)
        self.entries.append(header)
        self.bodies[event_id] = payload
        return event_id

    def forget(self, source_id: str, *, actor: str = "owner", at: str) -> str:
        """Erase source content and every transitively dependent payload, not headers.

        Audit metadata retains opaque IDs and hashes. This demo does not promise
        that hashes or third-party copies are privacy-safe in real deployments.
        """
        if actor != "owner":
            raise AuditError("only the owner may forget")
        self._by_id(source_id)
        gone = {source_id}
        changed = True
        while changed:
            changed = False
            for entry in self.entries:
                if entry["id"] not in gone and any(ref in gone for ref in entry["sources"]):
                    gone.add(entry["id"])
                    changed = True
        previous_hash = self.entries[-1]["hash"] if self.entries else GENESIS
        event_id = f"e{len(self.entries) + 1:04d}"
        body = {"target_id": source_id, "erased_ids": sorted(gone)}
        header = {
            "id": event_id, "kind": "forget", "actor": actor, "at": at,
            "owner": self.owner, "scope": self.scope, "sources": [],
            "model": "deterministic-fixture", "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "payload_digest": sha(body), "previous_hash": previous_hash,
        }
        header["hash"] = sha(header)
        self.entries.append(header)
        self.bodies[event_id] = body
        for gone_id in gone:
            self.bodies.pop(gone_id, None)
        return event_id

    def active_policies(self) -> list[str]:
        return [e["id"] for e in self.entries if e["kind"] == "policy_applied" and e["id"] not in self._erased()]

    def snapshot(self) -> dict:
        return {"schema": SCHEMA, "owner": self.owner, "scope": self.scope,
                "entries": deepcopy(self.entries), "bodies": deepcopy(self.bodies)}

    @classmethod
    def load(cls, snapshot: dict) -> "Journal":
        if snapshot.get("schema") != SCHEMA:
            raise AuditError("unknown schema")
        diary = cls(snapshot["owner"], snapshot["scope"])
        diary.entries = deepcopy(snapshot["entries"])
        diary.bodies = deepcopy(snapshot["bodies"])
        diary.verify()
        return diary

    def verify(self) -> None:
        seen: dict[str, dict] = {}
        previous = GENESIS
        for i, original in enumerate(self.entries, 1):
            entry = deepcopy(original)
            digest = entry.pop("hash", None)
            if digest != sha(entry) or entry.get("previous_hash") != previous:
                raise AuditError(f"hash chain broken at event {i}")
            previous = digest
            if entry.get("id") != f"e{i:04d}" or entry.get("kind") not in KINDS:
                raise AuditError("invalid event identity/type")
            if entry.get("owner") != self.owner or entry.get("scope") != self.scope:
                raise AuditError("scope mismatch")
            if len(set(entry.get("sources", []))) != len(entry.get("sources", [])) or any(ref not in seen for ref in entry["sources"]):
                raise AuditError("missing or forward source reference")
            seen[entry["id"]] = original
        forgotten: set[str] = set()
        for entry in self.entries:
            if entry["kind"] == "forget":
                marker = self.bodies.get(entry["id"])
                if not marker or sha(marker) != entry["payload_digest"]:
                    raise AuditError("missing or altered forgetting marker")
                forgotten.update(marker["erased_ids"])
                if not set(marker["erased_ids"]).issubset(seen):
                    raise AuditError("forgetting refers to missing records")
        for entry in self.entries:
            event_id = entry["id"]
            if event_id in forgotten:
                if event_id in self.bodies:
                    raise AuditError("forgotten payload still present")
            elif event_id not in self.bodies or sha(self.bodies[event_id]) != entry["payload_digest"]:
                raise AuditError(f"payload missing or modified: {event_id}")
        if set(self.bodies) - set(seen):
            raise AuditError("orphan payload")

    def markdown(self) -> str:
        self.verify()
        lines = ["# INFINITO — diario cognitivo P0", "", "**SIMULACIÓN DETERMINISTA.** Ningún modelo real razonó ni se modificaron pautas.", "",
                 f"Propietario ficticio: `{self.owner}` · Proyecto: `{self.scope}`", "",
                 "| UTC | Evento | Actor | Fuentes | Registro |", "|---|---|---|---|---|"]
        erased = self._erased()
        for e in self.entries:
            body = self.bodies.get(e["id"], {})
            description = "contenido olvidado" if e["id"] in erased else str(body.get("summary", body.get("decision", e["kind"])))
            description = description.replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {e['at']} | {e['id']} `{e['kind']}` | {e['actor']} | {', '.join(e['sources']) or '—'} | {description} |")
        lines += ["", f"Cadena final SHA-256: `{self.entries[-1]['hash'] if self.entries else GENESIS}`", "",
                  "**Pautas aplicadas: ninguna**" if not self.active_policies() else f"Pautas aplicadas: {', '.join(self.active_policies())}", "",
                  "La cadena detecta alteraciones accidentales del registro; no es una firma ni una garantía de inmutabilidad frente a un administrador.", ""]
        return "\n".join(lines)
