import json
from datetime import datetime
from typing import List, Optional

from .semantic_temporal_memory import SemanticTemporalMemoryStore
from .types import MemoryStatus


class ResolvedSemanticTemporalMemoryStore(SemanticTemporalMemoryStore):
    """Semantic temporal store with auditable ID-based retraction closure.

    Normal lexical/embedding retraction remains the first path.  A higher-level
    resolver may select one already-stored memory id when language changes make
    value matching ambiguous (for example, a cross-language retraction).  The
    selected record is closed rather than deleted, preserving temporal history.
    """

    def retract_fact(self, *args, **kwargs) -> List[str]:
        affected = list(super().retract_fact(*args, **kwargs) or [])
        if affected:
            self._mark_recorded_at(affected)
        return affected

    def retract_memory_id(
        self,
        memory_id: str,
        *,
        reason: str = "retracted",
        source_text: str = "",
        at: Optional[datetime] = None,
        resolver: str = "semantic_membership",
    ) -> List[str]:
        changed_at = at or datetime.utcnow()
        recorded_at = datetime.utcnow()
        with self._lock, self._conn:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ? AND status = ?",
                (str(memory_id), MemoryStatus.ACTIVE.value),
            ).fetchone()
            if row is None:
                return []
            metadata = json.loads(row["metadata_json"] or "{}")
            metadata.update(
                {
                    "temporal_valid_to": changed_at.isoformat(),
                    "retraction_reason": reason,
                    "retraction_source_text": source_text,
                    "retraction_recorded_at": recorded_at.isoformat(),
                    "retraction_target_resolver": resolver,
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
                    str(memory_id),
                ),
            )
        return [str(memory_id)]

    def _mark_recorded_at(self, memory_ids: List[str]) -> None:
        recorded_at = datetime.utcnow().isoformat()
        with self._lock, self._conn:
            for memory_id in memory_ids:
                row = self._conn.execute(
                    "SELECT metadata_json FROM memories WHERE id = ?",
                    (str(memory_id),),
                ).fetchone()
                if row is None:
                    continue
                metadata = json.loads(row["metadata_json"] or "{}")
                if metadata.get("retraction_recorded_at"):
                    continue
                metadata["retraction_recorded_at"] = recorded_at
                self._conn.execute(
                    "UPDATE memories SET metadata_json = ? WHERE id = ?",
                    (json.dumps(metadata, ensure_ascii=False, default=str), str(memory_id)),
                )
