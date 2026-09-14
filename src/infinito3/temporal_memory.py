import json
from collections import Counter
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

from .persistent_memory import (
    SQLiteCognitiveMemoryStore,
    _TOKEN_RE,
    _cosine,
    _normalize_text,
)
from .types import MemoryRecord, MemoryStatus


class TemporalAwareSQLiteMemoryStore(SQLiteCognitiveMemoryStore):
    """SQLite projection that accepts already-structured temporal facts.

    Legacy callers still use SQLiteCognitiveMemoryStore behavior. Cognitive-event
    projections may pre-populate fact_subject/predicate/value and mark a fact as
    exclusive, allowing bilingual or paraphrased events to share the same
    lineage even when SimpleFactExtractor cannot parse the surface text.
    """

    def add(self, record: MemoryRecord) -> MemoryRecord:
        structured = bool(
            record.fact_subject and record.fact_predicate and record.fact_value
        )
        if not structured:
            return super().add(record)

        normalized = _normalize_text(record.content)
        if not normalized:
            return record
        record.fact_value = _normalize_text(record.fact_value)
        exclusive = bool(record.metadata.get("fact_exclusive"))
        now = datetime.utcnow()
        record.updated_at = now
        embedding = self._safe_embed(record.content)

        with self._lock, self._conn:
            equivalent = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE fact_subject = ? AND fact_predicate = ? AND fact_value = ? AND status = ?
                ORDER BY updated_at DESC LIMIT 1
                """,
                (
                    record.fact_subject,
                    record.fact_predicate,
                    record.fact_value,
                    MemoryStatus.ACTIVE.value,
                ),
            ).fetchone()
            if equivalent is not None:
                return self._reinforce(equivalent, record)

            if exclusive:
                previous = self._conn.execute(
                    """
                    SELECT * FROM memories
                    WHERE fact_subject = ? AND fact_predicate = ? AND status = ? AND fact_value != ?
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (
                        record.fact_subject,
                        record.fact_predicate,
                        MemoryStatus.ACTIVE.value,
                        record.fact_value,
                    ),
                ).fetchone()
                if previous is not None:
                    record.supersedes_id = previous["id"]
                    previous_metadata = json.loads(previous["metadata_json"] or "{}")
                    previous_metadata["temporal_valid_to"] = now.isoformat()
                    previous_metadata["superseded_by"] = record.id
                    self._conn.execute(
                        """
                        UPDATE memories
                        SET status = ?, updated_at = ?, metadata_json = ?
                        WHERE id = ?
                        """,
                        (
                            MemoryStatus.SUPERSEDED.value,
                            now.isoformat(),
                            json.dumps(previous_metadata, ensure_ascii=False, default=str),
                            previous["id"],
                        ),
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

    def retract_fact(
        self,
        subject: str,
        predicate: str,
        value: str,
        *,
        reason: str = "retracted",
        source_text: str = "",
        at: Optional[datetime] = None,
    ) -> List[str]:
        """Close active matching fact versions without deleting their history."""
        normalized_value = _normalize_text(value)
        changed_at = at or datetime.utcnow()
        affected: List[str] = []
        with self._lock, self._conn:
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE fact_subject = ? AND fact_predicate = ? AND status = ?
                """,
                (subject, predicate, MemoryStatus.ACTIVE.value),
            ).fetchall()
            for row in rows:
                stored_value = _normalize_text(row["fact_value"] or "")
                if not self._fact_values_match(stored_value, normalized_value):
                    continue
                metadata = json.loads(row["metadata_json"] or "{}")
                metadata.update(
                    {
                        "temporal_valid_to": changed_at.isoformat(),
                        "retraction_reason": reason,
                        "retraction_source_text": source_text,
                    }
                )
                self._conn.execute(
                    """
                    UPDATE memories
                    SET status = ?, updated_at = ?, metadata_json = ?
                    WHERE id = ?
                    """,
                    (
                        MemoryStatus.SUPERSEDED.value,
                        changed_at.isoformat(),
                        json.dumps(metadata, ensure_ascii=False, default=str),
                        row["id"],
                    ),
                )
                affected.append(str(row["id"]))
        return affected

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        include_inactive: bool = False,
    ) -> List[MemoryRecord]:
        if not include_inactive:
            return super().search(query, top_k=top_k)

        query_tokens = Counter(_TOKEN_RE.findall(query.lower()))
        query_embedding = self._safe_embed(query)
        now = datetime.utcnow()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE status IN (?, ?)
                """,
                (MemoryStatus.ACTIVE.value, MemoryStatus.SUPERSEDED.value),
            ).fetchall()

        scored: List[Tuple[float, object]] = []
        for row in rows:
            record_tokens = Counter(_TOKEN_RE.findall(row["content"].lower()))
            overlap = sum((query_tokens & record_tokens).values())
            union = sum((query_tokens | record_tokens).values()) or 1
            lexical = overlap / union
            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            vector = max(0.0, _cosine(query_embedding, embedding)) if query_embedding and embedding else 0.0
            recency = 0.0
            try:
                created = datetime.fromisoformat(row["created_at"])
                age_days = max(0.0, (now - created).total_seconds() / 86400.0)
                recency = 0.5 ** (age_days / 90.0)
            except Exception:
                pass
            historical_bonus = 0.07 if row["status"] == MemoryStatus.SUPERSEDED.value else 0.0
            score = (
                0.42 * lexical
                + 0.40 * vector
                + 0.08 * float(row["importance"])
                + 0.04 * float(row["confidence"])
                + 0.03 * recency
                + historical_bonus
            )
            if lexical > 0.0 or vector >= 0.10:
                scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [self._row_to_record(row) for _, row in scored[:top_k]]

    def semantic_scores(self, query: str, memory_ids: Sequence[str]) -> Dict[str, float]:
        ids = [str(memory_id) for memory_id in memory_ids if memory_id]
        if not ids:
            return {}
        query_embedding = self._safe_embed(query)
        if not query_embedding:
            return {}
        placeholders = ",".join("?" for _ in ids)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT id, embedding_json FROM memories WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
        result: Dict[str, float] = {}
        for row in rows:
            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            if embedding:
                result[str(row["id"])] = max(0.0, _cosine(query_embedding, embedding))
        return result

    @staticmethod
    def _fact_values_match(left: str, right: str) -> bool:
        if left == right:
            return True
        if left and right and (left in right or right in left):
            return min(len(left), len(right)) >= 4
        left_terms = set(left.split())
        right_terms = set(right.split())
        if not left_terms or not right_terms:
            return False
        return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms))) >= 0.6
