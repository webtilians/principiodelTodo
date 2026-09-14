import json
from datetime import datetime
from typing import List, Optional

from .persistent_memory import _cosine, _normalize_text
from .temporal_memory import TemporalAwareSQLiteMemoryStore
from .types import MemoryStatus


class SemanticTemporalMemoryStore(TemporalAwareSQLiteMemoryStore):
    """Temporal store with conservative semantic target resolution.

    Resolution order:
    1. exact/lexical identity inside the requested predicate;
    2. unique exact value identity across active predicates for the same subject;
    3. semantic similarity inside the requested predicate.

    Step 2 handles ontology drift such as `drinks=kombucha` versus an existing
    `likes=kombucha` without encoding a domain dictionary. It only fires when a
    single active record owns that exact normalized value.
    """

    semantic_retraction_threshold = 0.48
    semantic_retraction_margin = 0.05

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
        lexical = super().retract_fact(
            subject,
            predicate,
            value,
            reason=reason,
            source_text=source_text,
            at=at,
        )
        if lexical:
            return lexical

        normalized_value = _normalize_text(value)
        cross_predicate = self._unique_active_value_match(subject, normalized_value)
        if cross_predicate is not None:
            return self._close_row(
                cross_predicate,
                reason=reason,
                source_text=source_text,
                at=at,
                metadata_extra={"cross_predicate_value_retraction": True},
            )

        query_embedding = self._safe_embed(value)
        if not query_embedding:
            return []

        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE fact_subject = ? AND fact_predicate = ? AND status = ?
                """,
                (subject, predicate, MemoryStatus.ACTIVE.value),
            ).fetchall()

        scored = []
        for row in rows:
            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            if not embedding:
                continue
            score = max(0.0, _cosine(query_embedding, embedding))
            scored.append((score, row))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        if not scored:
            return []

        best_score, best = scored[0]
        second = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < self.semantic_retraction_threshold:
            return []
        if len(scored) > 1 and best_score - second < self.semantic_retraction_margin:
            return []

        return self._close_row(
            best,
            reason=reason,
            source_text=source_text,
            at=at,
            metadata_extra={
                "semantic_retraction": True,
                "semantic_retraction_score": best_score,
            },
        )

    def _unique_active_value_match(self, subject: str, normalized_value: str):
        if not normalized_value:
            return None
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT * FROM memories
                WHERE fact_subject = ? AND status = ?
                """,
                (subject, MemoryStatus.ACTIVE.value),
            ).fetchall()
        matches = [
            row for row in rows
            if _normalize_text(str(row["fact_value"] or "")) == normalized_value
        ]
        return matches[0] if len(matches) == 1 else None

    def _close_row(self, row, *, reason: str, source_text: str, at: Optional[datetime], metadata_extra=None):
        changed_at = at or datetime.utcnow()
        metadata = json.loads(row["metadata_json"] or "{}")
        metadata.update(
            {
                "temporal_valid_to": changed_at.isoformat(),
                "retraction_reason": reason,
                "retraction_source_text": source_text,
                **(metadata_extra or {}),
            }
        )
        with self._lock, self._conn:
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
        return [str(row["id"])]
