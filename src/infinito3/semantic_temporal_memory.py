import json
from datetime import datetime
from typing import List, Optional

from .persistent_memory import _cosine, _normalize_text
from .temporal_memory import TemporalAwareSQLiteMemoryStore
from .types import MemoryStatus


class SemanticTemporalMemoryStore(TemporalAwareSQLiteMemoryStore):
    """Temporal store with conservative semantic target resolution.

    Retraction first uses exact/lexical fact identity from the base store. Only
    when that finds nothing does it compare one query embedding with embeddings
    already stored for active values of the same structured predicate. No
    bilingual dictionary or domain vocabulary is encoded here.
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

        changed_at = at or datetime.utcnow()
        metadata = json.loads(best["metadata_json"] or "{}")
        metadata.update(
            {
                "temporal_valid_to": changed_at.isoformat(),
                "retraction_reason": reason,
                "retraction_source_text": source_text,
                "semantic_retraction": True,
                "semantic_retraction_score": best_score,
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
                    best["id"],
                ),
            )
        return [str(best["id"])]
