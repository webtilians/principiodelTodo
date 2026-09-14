import json
from typing import Dict, List, Optional, Sequence

from .context_builder import _TOKEN_RE
from .generalized_context_builder import GeneralizedContextBuilder
from .persistent_memory import SQLiteCognitiveMemoryStore, _cosine
from .semantic_reranker import SemanticMembershipReranker
from .types import ContextItem, ContextPacket


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
    """Generalized builder with semantic retrieval and optional membership reranking.

    Embeddings remain the first semantic stage. When a multi-value query leaves
    several same-predicate candidates and an injectable reranker is configured,
    the reranker classifies only that already-retrieved candidate set. Its usage
    is attached to ContextPacket diagnostics so evaluation can account for the
    extra model cost explicitly.
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

    def __init__(self, *args, reranker: Optional[SemanticMembershipReranker] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.reranker = reranker
        self._reranker_events: List[Dict[str, object]] = []

    def build(self, *args, **kwargs) -> ContextPacket:
        self._reranker_events = []
        packet = super().build(*args, **kwargs)
        if self._reranker_events:
            packet.diagnostics["semantic_reranker"] = self._summarize_reranker_events()
        return packet

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

        # A generative/classification reranker is deliberately optional. When
        # present it sees the full small same-predicate candidate set so it can
        # recover a true member that raw embedding rank placed below a distractor.
        if self.reranker is not None and len(same_predicate) >= 3:
            reranked = self.reranker.rerank(query, same_predicate)
            event = {
                "success": bool(reranked.success),
                "candidate_count": len(same_predicate),
                "selected_count": len(reranked.selected_ids),
                "provider": reranked.provider,
                "model": reranked.model,
                "input_tokens": int(reranked.usage.get("input_tokens") or 0),
                "output_tokens": int(reranked.usage.get("output_tokens") or 0),
                "total_tokens": int(reranked.usage.get("total_tokens") or 0),
                "error": reranked.error,
            }
            self._reranker_events.append(event)
            if reranked.success:
                selected_ids = set(reranked.selected_ids)
                selected = [item for item in ranked if str(item.memory_id) in selected_ids]
                for item in selected:
                    item.metadata["semantic_reranker_selected"] = True
                return selected

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

        # Embedding-only fallback. It stays available for offline/local modes
        # and whenever the injected reranker fails.
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

    def _summarize_reranker_events(self) -> Dict[str, object]:
        providers = sorted({str(event.get("provider") or "unknown") for event in self._reranker_events})
        models = sorted({str(event.get("model") or "") for event in self._reranker_events if event.get("model")})
        errors = [str(event["error"]) for event in self._reranker_events if event.get("error")]
        return {
            "calls": len(self._reranker_events),
            "successful_calls": sum(bool(event.get("success")) for event in self._reranker_events),
            "candidate_count": sum(int(event.get("candidate_count") or 0) for event in self._reranker_events),
            "selected_count": sum(int(event.get("selected_count") or 0) for event in self._reranker_events),
            "input_tokens": sum(int(event.get("input_tokens") or 0) for event in self._reranker_events),
            "output_tokens": sum(int(event.get("output_tokens") or 0) for event in self._reranker_events),
            "total_tokens": sum(int(event.get("total_tokens") or 0) for event in self._reranker_events),
            "providers": providers,
            "models": models,
            "errors": errors,
        }
