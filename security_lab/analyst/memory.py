"""Incident ledger using INFINITO, always in a new explicit local lab directory."""
import re
from pathlib import Path

from src.infinito3.mcp_server import AgentMemory

PROJECT = "Arena analyst synthetic incidents"
KINDS = {"incident": "fact", "hypothesis": "assumption", "failure": "fact", "verified_defence": "fact"}


class IncidentMemory:
    def __init__(self, directory):
        directory = Path(directory).resolve()
        # Creating a new directory prevents opening an existing personal database.
        directory.mkdir(parents=True, exist_ok=False)
        self.agent = AgentMemory(str(directory / "incidents.sqlite3"))

    def record(self, category, evidence_hash, status):
        if category not in KINDS or not re.fullmatch(r"[a-f0-9]{64}", evidence_hash):
            raise ValueError("invalid incident provenance")
        if status not in {"observed", "proposed", "rejected", "failed", "verified"}:
            raise ValueError("invalid status")
        if (category == "verified_defence") != (status == "verified"):
            raise ValueError("verification status mismatch")
        # Fixed vocabulary, never raw logs, model suggestions or source text.
        statement = (f"Synthetic Arena authorization {category}: status {status}. "
                     "This record describes a lab observation, not a production incident.")
        result = self.agent.remember(statement, kind=KINDS[category], project=PROJECT,
                                     source=f"arena-report-sha256:{evidence_hash}")
        if result.startswith(("Not stored", "Nothing")):
            raise ValueError("INFINITO refused incident record")
        return result

    def recall(self):
        return self.agent.recall("Arena authorization incident", project=PROJECT, limit=4)[:2500]

    def close(self):
        self.agent.close()
