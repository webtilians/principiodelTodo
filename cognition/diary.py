"""The engine's diary: everything it did and why, append-only, so the person can audit it and see it change over time.

Each entry has a header in diary.jsonl and a body in bodies/<seq>.json. The header holds what can be audited without
any content: when, what type, which role and model, which memory ids, the cost, and the hash of the body. Headers are
chained (each holds the hash of the one before), so a changed or removed header shows up in verify(). A body holds
what the models said; it can be purged when the person forgets a memory it cites, and the chain stays whole. Each body
carries a random salt, so the hash left behind cannot be matched by guessing what the body said.

No memory text, prompt or key is written here: only ids, and the models' own words.
"""
import hashlib
import json
import os
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, List, Optional, Sequence

GENESIS = "0" * 64


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _digest(data: dict) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


class Diary:
    def __init__(self, folder: Path, clock=None):
        self.folder, self.clock = folder, clock or _now
        self.headers = folder / "diary.jsonl"
        self.bodies = folder / "bodies"

    def _all(self) -> List[dict]:
        try:
            lines = self.headers.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            return []
        return [json.loads(line) for line in lines if line.strip()]

    def append(self, kind: str, *, ids: Sequence[str] = (), at: Optional[str] = None, body: Optional[dict] = None,
               **header) -> dict:
        self.bodies.mkdir(parents=True, exist_ok=True)
        entries = self._all()
        seq = len(entries) + 1
        prev = entries[-1]["hash"] if entries else GENESIS
        content = {"salt": secrets.token_hex(16), **(body or {})}
        (self.bodies / f"{seq}.json").write_text(json.dumps(content, ensure_ascii=False, indent=1), encoding="utf-8")
        entry = {"seq": seq, "at": at or self.clock(), "type": kind, "ids": sorted(set(ids)), **header,
                 "body_sha256": _digest(content), "prev": prev}
        entry["hash"] = _digest(entry)
        with self.headers.open("a", encoding="utf-8") as out:
            out.write(json.dumps(entry, ensure_ascii=False) + "\n")
            out.flush()
            os.fsync(out.fileno())
        return entry

    def body(self, entry: dict) -> Optional[dict]:
        try:
            return json.loads((self.bodies / f"{entry['seq']}.json").read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None

    def entries(self, kind: Optional[str] = None) -> Iterator[dict]:
        for entry in self._all():
            if kind is None or entry["type"] == kind:
                yield entry

    def last(self, kind: str, **match) -> Optional[dict]:
        found = None
        for entry in self.entries(kind):
            if all(entry.get(k) == v for k, v in match.items()):
                found = entry
        return found

    def verify(self) -> List[str]:
        """What is wrong with the diary: a broken chain or a body that does not match its header. Empty is good.
        A purged body is not a fault: its header says so."""
        problems, prev = [], GENESIS
        for entry in self._all():
            claimed = dict(entry)
            stored = claimed.pop("hash")
            if claimed["prev"] != prev:
                problems.append(f"entry {entry['seq']}: chain broken")
            if _digest(claimed) != stored:
                problems.append(f"entry {entry['seq']}: header changed")
            body = self.body(entry)
            if body is None and not entry.get("purged"):
                problems.append(f"entry {entry['seq']}: body missing")
            elif body is not None and _digest(body) != entry["body_sha256"]:
                problems.append(f"entry {entry['seq']}: body changed")
            prev = stored
        return problems

    def purge(self, memory_ids: Sequence[str]) -> int:
        """Delete every body that cites one of these memories, and say so in a new entry. Headers keep the ids and the
        hash, so the chain stays whole; a purged body is listed in that entry so verify() knows it is gone on purpose."""
        gone = set(memory_ids)
        purged = []
        for entry in self._all():
            if gone & set(entry.get("ids", [])) and self.body(entry) is not None:
                (self.bodies / f"{entry['seq']}.json").unlink()
                purged.append(entry["seq"])
        if purged:
            self.append("purge", purged_seqs=purged, ids=sorted(gone))
        return len(purged)

    def purged(self) -> set:
        seqs = set()
        for entry in self.entries("purge"):
            seqs.update(entry.get("purged_seqs", []))
        return seqs


def _verify_with_purges(diary: Diary) -> List[str]:
    purged = diary.purged()
    return [p for p in diary.verify() if not (p.endswith("body missing") and int(p.split()[1].rstrip(":")) in purged)]


def check(diary: Diary) -> List[str]:
    """verify(), knowing which bodies were purged on purpose."""
    return _verify_with_purges(diary)


# Guidelines the supervisor proposes for a role, and the person's word on them.

def proposals(diary: Diary) -> List[dict]:
    """Every proposal with its status: pending, approved, rejected or revoked (the person's latest word wins)."""
    decided = {}
    for entry in diary.entries("policy_decision"):
        decided[entry["proposal"]] = entry["decision"]
    found = []
    for entry in diary.entries("policy_proposal"):
        body = diary.body(entry) or {}
        found.append({"id": entry["proposal"], "at": entry["at"], "target_role": entry["target_role"],
                      "guideline": body.get("guideline", "(purged)"), "rationale": body.get("rationale", ""),
                      "test": body.get("test", ""), "status": decided.get(entry["proposal"], "pending")})
    return found


def approved_guidelines(diary: Diary, role: str) -> List[dict]:
    return [p for p in proposals(diary) if p["target_role"] == role and p["status"] == "approved"]
