import json
from typing import Dict, List, Sequence

from .context_builder import _TOKEN_RE
from .generalized_context_builder import GeneralizedContextBuilder
from .persistent_memory import SQLiteCognitiveMemoryStore, _cosine
from .types import ContextItem


class SemanticScoringSQLiteMemoryStore(SQLiteCognitiveMemoryStore):
    """SQLite memory store that can score an existing candidate set semantically.

    Normal retrieval remains unchanged. The extra method is deliberately narrow:
    it embeds one focused query and compares it with embeddings already stored for
    the supplied memory ids. This avoids another per-memory embedding call and
    keeps semantic cohort selection outside the persistence schema.
    """

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

        scores: Dict[str, float] = {}
        for row in rows:
            embedding = json.loads(row["embedding_json"]) if row["embedding_json"] else None
            if embedding:
                scores[str(row["id"])] = max(0.0, _cosine(query_embedding, embedding))
        return scores


class SemanticCohortContextBuilder(GeneralizedContextBuilder):
    """Generalized builder with an optional semantic second-stage for multi-value facts.

    A query such as "all outdoor activities I like" should not pull every
    `likes` fact merely because they share a predicate. The builder first uses
    normal retrieval, then re-scores only the already retrieved same-predicate
    candidates against a compact semantic focus extracted from the request.

    No domain vocabulary is encoded here: music, food, sport and future domains
    use the same mechanism.
    """

    _FOCUS_STOP_WORDS = {
        "a", "al", "de", "del", "el", "la", "los", "las", "un", "una",
        "y", "o", "que", "qué", "cual", "cuál", "cuales", "cuáles",
        "dime", "cuentame", "cuéntame", "incluye", "incluir", "todos", "todas",
        "todo", "toda", "recuerda", "recuerdas", "recuerdes", "me", "mi", "mis",
        "te", "he", "dicho", "contado", "gustan", "gusta", "gustado",
        "the", "a", "an", "and", "or", "of", "to", "what", "which", "tell",
        "include", "all", "every", "everything", "remember", "remembered", "my",
        "me", "i", "you", "told", "do", "does", "like", "liked",
    }

    @classmethod
    def _semantic_focus_query(cls, query: str) -> str:
        tokens = _TOKEN_RE.findall(query.lower())
        focused = [token for token in tokens if token not in cls._FOCUS_STOP_WORDS and not token.isdigit()]
        return " ".join(focused) or query

    def _prune_memory_pool(self, query: str, items: Sequence[ContextItem]) -> List[ContextItem]:
        baseline = super()._prune_memory_pool(query, items)
        if len(items) <= 1 or not self._query_allows_multiple(query):
            return baseline

        ranked = sorted(items, key=lambda item: item.score, reverse=True)
        top_predicate = ranked[0].metadata.get("fact_predicate")
        if not top_predicate:
            return baseline

        same_predicate = [
            item
            for item in ranked
            if item.metadata.get("fact_predicate") == top_predicate and item.memory_id
        ]
        if len(same_predicate) <= 1:
            return baseline

        scorer = getattr(self.memory_store, "semantic_scores", None)
        if scorer is None:
            return baseline

        focus = self._semantic_focus_query(query)
        semantic = scorer(focus, [item.memory_id for item in same_predicate])
        if len(semantic) < 2:
            return baseline

        ordered = sorted(
            same_predicate,
            key=lambda item: semantic.get(str(item.memory_id), -1.0),
            reverse=True,
        )
        top_score = semantic.get(str(ordered[0].memory_id), 0.0)
        if top_score <= 0.0:
            return baseline

        # Broad requests (for example "all my preferences") tend to give every
        # value similar low-to-mid similarity. Narrow facets produce a sharper
        # semantic cluster. Use a relative floor and then an elbow only when the
        # gap is substantial, so the policy adapts without domain labels.
        floor = max(0.16, top_score * 0.62, top_score - 0.22)
        eligible = [
            item
            for item in ordered
            if semantic.get(str(item.memory_id), 0.0) >= floor
        ]
        if not eligible:
            eligible = [ordered[0]]

        if len(eligible) >= 3:
            values = [semantic.get(str(item.memory_id), 0.0) for item in eligible]
            gaps = [(values[index] - values[index + 1], index) for index in range(len(values) - 1)]
            largest_gap, gap_index = max(gaps, key=lambda pair: pair[0])
            if largest_gap >= 0.075 and gap_index >= 1:
                eligible = eligible[: gap_index + 1]

        selected_ids = {item.memory_id for item in eligible}
        selected = [item for item in ranked if item.memory_id in selected_ids]
        for item in selected:
            item.metadata["semantic_focus"] = focus
            item.metadata["semantic_focus_score"] = semantic.get(str(item.memory_id), 0.0)
        return selected
