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

    def record_reviewed_lesson(self, lesson):
        """Store a fixed, independently gated lab lesson, not raw/model-authored text."""
        from .lessons import LEVELS_LESSON
        if not isinstance(lesson, dict) or lesson.get("id") != "AUTHZ-LEVELS-001":
            raise ValueError("unreviewed security lesson")
        if lesson.get("status") != "reviewed_synthetic_recording" or lesson.get("lesson") != LEVELS_LESSON:
            raise ValueError("lesson provenance or content mismatch")
        source = lesson.get("source_sha256")
        if not isinstance(source, str) or not re.fullmatch(r"[a-f0-9]{64}", source):
            raise ValueError("invalid source hash")
        result = self.agent.remember(lesson["lesson"], kind="fact", project=PROJECT,
                                     source=f"arena-review-sha256:{source}")
        if result.startswith(("Not stored", "Nothing")):
            raise ValueError("INFINITO refused reviewed lesson")
        return result

    def recall_lessons(self, query):
        """Read project-scoped prior synthetic lessons only; treat all text as untrusted data."""
        if not isinstance(query, str) or not 1 <= len(query) <= 120:
            raise ValueError("invalid recall query")
        return self.agent.recall(query, project=PROJECT, limit=3)[:3000]

    def close(self):
        self.agent.close()
