import hashlib
import json
import math
import re
import sqlite3
import threading
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .interfaces import EmbeddingProvider
from .types import MaintenanceReport, MemoryKind, MemoryRecord, MemoryStatus


_TOKEN_RE = re.compile(r"[\wáéíóúüñ]+", re.I)


def _normalize_text(text: str) -> str:
    return " ".join(_TOKEN_RE.findall(text.lower()))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class HashEmbeddingProvider:
    """Deterministic local embedding baseline with no external dependency.

    It is intentionally simple and primarily useful for tests/offline operation.
    A semantic provider such as OpenAI can be injected through the same protocol.
    """

    def __init__(self, dimensions: int = 128):
        self.dimensions = dimensions

    def embed(self, text: str) -> List[float]:
        vector = [0.0] * self.dimensions
        for token in _TOKEN_RE.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimensions
            sign = 1.0 if ((value >> 8) & 1) else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]


class OpenAIEmbeddingProvider:
    """Thin adapter around an already-created OpenAI-compatible client."""

    def __init__(self, client, model: str = "text-embedding-3-small"):
        self.client = client
        self.model = model

    def embed(self, text: str) -> List[float]:
        response = self.client.embeddings.create(input=[text], model=self.model)
        return list(response.data[0].embedding)


@dataclass(frozen=True)
class ExtractedFact:
    subject: str
    predicate: str
    value: str
    exclusive: bool = False


class SimpleFactExtractor:
    """Conservative structured-fact extractor for user-model memories.

    Only patterns whose semantics are clear are structured. Unrecognized text
    remains a normal memory instead of being forced into a brittle schema.
    """

    _PATTERNS: Tuple[Tuple[re.Pattern, str, bool], ...] = (
        (re.compile(r"\bme llamo\s+([^,.!?]+)", re.I), "name", True),
        (re.compile(r"\bmi nombre es\s+([^,.!?]+)", re.I), "name", True),
        (re.compile(r"\bvivo en\s+([^,.!?]+)", re.I), "location", True),
        (re.compile(r"\btengo\s+(\d{1,3})\s+años\b", re.I), "age", True),
        (re.compile(r"\bmi color favorito es\s+([^,.!?]+)", re.I), "favorite_color", True),
        (re.compile(r"\bmi bici es\s+([^,.!?]+)", re.I), "bike", True),
        (re.compile(r"\bme gusta\s+([^,.!?]+)", re.I), "likes", False),
        (re.compile(r"\bme encanta\s+([^,.!?]+)", re.I), "likes", False),
        (re.compile(r"\bprefiero\s+([^,.!?]+)", re.I), "prefers", False),
    )

    def extract(self, text: str, kind: MemoryKind) -> Optional[ExtractedFact]:
        if kind != MemoryKind.USER_MODEL:
            return None
        for pattern, predicate, exclusive in self._PATTERNS:
            match = pattern.search(text)
            if match:
                value = _normalize_text(match.group(1)).strip()
                if value:
                    return ExtractedFact("user", predicate, value, exclusive)
        return None


class SQLiteCognitiveMemoryStore:
    """Persistent cognitive memory with hybrid retrieval and memory lineage.

    Core properties:
    - SQLite persistence
    - optional semantic embeddings
    - lexical + vector + cognitive ranking
    - reinforcement of repeated facts
    - contradiction/supersession for exclusive facts
    - reversible forgetting
    - consolidation for imported/legacy duplicates
    """

    def __init__(
        self,
        path: str = "data/infinito3_memory.db",
        embedding_provider: Optional[EmbeddingProvider] = None,
        fact_extractor: Optional[SimpleFactExtractor] = None,
    ):
        self.path = path
        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.embedding_provider = embedding_provider or HashEmbeddingProvider()
        self.fact_extractor = fact_extractor or SimpleFactExtractor()
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    normalized_content TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_accessed_at TEXT,
                    access_count INTEGER NOT NULL DEFAULT 0,
                    embedding_json TEXT,
                    fact_subject TEXT,
                    fact_predicate TEXT,
                    fact_value TEXT,
                    status TEXT NOT NULL,
                    supersedes_id TEXT
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_status ON memories(status)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_fact ON memories(fact_subject, fact_predicate, fact_value)"
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memories_normalized ON memories(normalized_content)"
            )

    def _safe_embed(self, text: str) -> Optional[List[float]]:
        try:
            return list(self.embedding_provider.embed(text))
        except Exception:
            return None

    def add(self, record: MemoryRecord) -> MemoryRecord:
        normalized = _normalize_text(record.content)
        if not normalized:
            return record

        fact = self.fact_extractor.extract(record.content, record.kind)
        if fact:
            record.fact_subject = fact.subject
            record.fact_predicate = fact.predicate
            record.fact_value = fact.value

        now = datetime.utcnow()
        record.updated_at = now
        embedding = self._safe_embed(record.content)

        with self._lock, self._conn:
            exact = self._conn.execute(
                "SELECT * FROM memories WHERE normalized_content = ? AND status = ? LIMIT 1",
                (normalized, MemoryStatus.ACTIVE.value),
            ).fetchone()
            if exact is not None:
                return self._reinforce(exact, record)

            if fact is not None:
                equivalent = self._conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE fact_subject = ? AND fact_predicate = ? AND fact_value = ? AND status = ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (fact.subject, fact.predicate, fact.value, MemoryStatus.ACTIVE.value),
                ).fetchone()
                if equivalent is not None:
                    return self._reinforce(equivalent, record)

                if fact.exclusive:
                    previous = self._conn.execute(
                        """
                        SELECT * FROM memories
                        WHERE fact_subject = ? AND fact_predicate = ? AND status = ? AND fact_value != ?
                        ORDER BY updated_at DESC LIMIT 1
                        """,
                        (fact.subject, fact.predicate, MemoryStatus.ACTIVE.value, fact.value),
                    ).fetchone()
                    if previous is not None:
                        record.supersedes_id = previous["id"]
                        self._conn.execute(
                            "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                            (MemoryStatus.SUPERSEDED.value, now.isoformat(), previous["id"]),
                        )

            self._conn.execute(
                """
                INSERT INTO memories (
                    id, content, normalized_content, kind, importance, confidence,
                    metadata_json, created_at, updated_at, last_accessed_at,
                    access_count, embedding_json, fact_subject, fact_predicate,
                    fact_value, status, supersedes_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.id,
                    record.content,
                    normalized,
                    record.kind.value,
                    float(record.importance),
                    float(record.confidence),
                    json.dumps(record.metadata, ensure_ascii=False, default=str),
                    record.created_at.isoformat(),
                    record.updated_at.isoformat(),
                    record.last_accessed_at.isoformat() if record.last_accessed_at else None,
                    int(record.access_count),
                    json.dumps(embedding) if embedding is not None else None,
                    record.fact_subject,
                    record.fact_predicate,
                    record.fact_value,
                    record.status.value,
                    record.supersedes_id,
                ),
            )
        return record

    def _reinforce(self, row: sqlite3.Row, incoming: MemoryRecord) -> MemoryRecord:
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata["reinforcement_count"] = int(metadata.get("reinforcement_count", 0)) + 1
        importance = max(float(row["importance"]), float(incoming.importance))
        confidence = min(1.0, float(row["confidence"]) + 0.05)
        now = datetime.utcnow().isoformat()
        self._conn.execute(
            """
            UPDATE memories
            SET importance = ?, confidence = ?, metadata_json = ?, updated_at = ?
            WHERE id = ?
            """,
            (importance, confidence, json.dumps(metadata, ensure_ascii=False), now, row["id"]),
        )
        refreshed = self._conn.execute("SELECT * FROM memories WHERE id = ?", (row["id"],)).fetchone()
        return self._row_to_record(refreshed)

    def get(self, memory_id: str) -> Optional[MemoryRecord]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM memories WHERE id = ?", (memory_id,)).fetchone()
        return self._row_to_record(row) if row else None

    def all(self, include_inactive: bool = False) -> List[MemoryRecord]:
        with self._lock:
            if include_inactive:
                rows = self._conn.execute("SELECT * FROM memories ORDER BY created_at").fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM memories WHERE status = ? ORDER BY created_at",
                    (MemoryStatus.ACTIVE.value,),
                ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def search(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        query_tokens = Counter(_TOKEN_RE.findall(query.lower()))
        query_embedding = self._safe_embed(query)
        now = datetime.utcnow()

        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM memories WHERE status = ?",
                (MemoryStatus.ACTIVE.value,),
            ).fetchall()

        scored: List[Tuple[float, sqlite3.Row]] = []
        for row in rows:
            record_tokens = Counter(_TOKEN_RE.findall(row["content"].lower()))
            overlap = sum((query_tokens & record_tokens).values())
            union = sum((query_tokens | record_tokens).values()) or 1
            lexical = overlap / union

            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            vector = max(0.0, _cosine(query_embedding, embedding)) if query_embedding and embedding else 0.0

            created_at = datetime.fromisoformat(row["created_at"])
            age_days = max(0.0, (now - created_at).total_seconds() / 86400.0)
            recency = 0.5 ** (age_days / 90.0)
            access = min(1.0, math.log1p(int(row["access_count"])) / 4.0)

            score = (
                0.42 * lexical
                + 0.38 * vector
                + 0.08 * float(row["importance"])
                + 0.05 * float(row["confidence"])
                + 0.04 * recency
                + 0.03 * access
            )
            if lexical > 0.0 or vector >= 0.10:
                scored.append((score, row))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[:top_k]

        if selected:
            accessed_at = now.isoformat()
            with self._lock, self._conn:
                self._conn.executemany(
                    """
                    UPDATE memories
                    SET access_count = access_count + 1, last_accessed_at = ?
                    WHERE id = ?
                    """,
                    [(accessed_at, row["id"]) for _, row in selected],
                )

        return [self.get(row["id"]) for _, row in selected if self.get(row["id"]) is not None]

    def history_for_fact(self, subject: str, predicate: str) -> List[MemoryRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE fact_subject = ? AND fact_predicate = ?
                ORDER BY created_at
                """,
                (subject, predicate),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def consolidate(self) -> MaintenanceReport:
        """Merge duplicate/equivalent active records without destroying lineage."""
        active = self.all()
        groups: Dict[Tuple[str, ...], List[MemoryRecord]] = defaultdict(list)
        for record in active:
            if record.fact_subject and record.fact_predicate and record.fact_value:
                key = ("fact", record.fact_subject, record.fact_predicate, record.fact_value)
            else:
                key = ("text", _normalize_text(record.content))
            groups[key].append(record)

        consolidated = 0
        with self._lock, self._conn:
            for records in groups.values():
                if len(records) < 2:
                    continue
                records.sort(key=lambda r: (r.importance, r.confidence, r.updated_at), reverse=True)
                winner = records[0]
                for duplicate in records[1:]:
                    self._conn.execute(
                        "UPDATE memories SET status = ?, supersedes_id = ?, updated_at = ? WHERE id = ?",
                        (
                            MemoryStatus.SUPERSEDED.value,
                            winner.id,
                            datetime.utcnow().isoformat(),
                            duplicate.id,
                        ),
                    )
                    consolidated += 1
                self._conn.execute(
                    "UPDATE memories SET confidence = ? WHERE id = ?",
                    (min(1.0, winner.confidence + 0.05 * (len(records) - 1)), winner.id),
                )
        return MaintenanceReport(consolidated=consolidated)

    def forget(
        self,
        min_retention: float = 0.22,
        max_active: Optional[int] = None,
        half_life_days: float = 90.0,
        protect_importance: float = 0.85,
    ) -> MaintenanceReport:
        """Soft-forget weak memories using importance, confidence, age and use.

        Forgotten rows stay in SQLite for audit/restore and are excluded from
        normal retrieval. Highly important memories are protected by default.
        """
        now = datetime.utcnow()
        records = self.all()
        ranked: List[Tuple[float, MemoryRecord]] = []

        durability = {
            MemoryKind.WORKING: 0.05,
            MemoryKind.EPISODIC: 0.25,
            MemoryKind.SEMANTIC: 0.70,
            MemoryKind.USER_MODEL: 0.85,
        }

        for record in records:
            age_days = max(0.0, (now - record.updated_at).total_seconds() / 86400.0)
            recency = 0.5 ** (age_days / half_life_days)
            access = min(1.0, math.log1p(record.access_count) / 4.0)
            retention = (
                0.35 * record.importance
                + 0.25 * record.confidence
                + 0.18 * recency
                + 0.10 * access
                + 0.12 * durability[record.kind]
            )
            ranked.append((retention, record))

        to_forget = {
            record.id
            for retention, record in ranked
            if retention < min_retention and record.importance < protect_importance
        }

        if max_active is not None and len(records) - len(to_forget) > max_active:
            candidates = [
                (retention, record)
                for retention, record in ranked
                if record.id not in to_forget and record.importance < protect_importance
            ]
            candidates.sort(key=lambda item: item[0])
            extra = len(records) - len(to_forget) - max_active
            to_forget.update(record.id for _, record in candidates[:extra])

        if to_forget:
            with self._lock, self._conn:
                self._conn.executemany(
                    "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                    [
                        (MemoryStatus.FORGOTTEN.value, now.isoformat(), memory_id)
                        for memory_id in to_forget
                    ],
                )
        return MaintenanceReport(forgotten=len(to_forget))

    def restore(self, memory_id: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE id = ? AND status = ?",
                (
                    MemoryStatus.ACTIVE.value,
                    datetime.utcnow().isoformat(),
                    memory_id,
                    MemoryStatus.FORGOTTEN.value,
                ),
            )
        return cursor.rowcount > 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) AS n FROM memories GROUP BY status"
            ).fetchall()
        result = {status.value: 0 for status in MemoryStatus}
        for row in rows:
            result[row["status"]] = int(row["n"])
        return result

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            id=row["id"],
            content=row["content"],
            kind=MemoryKind(row["kind"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            metadata=json.loads(row["metadata_json"] or "{}"),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
            last_accessed_at=datetime.fromisoformat(row["last_accessed_at"]) if row["last_accessed_at"] else None,
            access_count=int(row["access_count"]),
            status=MemoryStatus(row["status"]),
            supersedes_id=row["supersedes_id"],
            fact_subject=row["fact_subject"],
            fact_predicate=row["fact_predicate"],
            fact_value=row["fact_value"],
        )
