import re
from collections import Counter
from typing import List

from .types import GateDecision, MemoryKind, MemoryRecord


_TOKEN_RE = re.compile(r"[\wáéíóúüñ]+", re.I)


class RuleBasedMemoryGate:
    """Transparent baseline gate for INFINITO 3.0.

    This is intentionally simple: every future learned gate must beat this
    baseline on a held-out evaluation set before replacing it.
    """

    _TRIVIAL = {
        "hola", "buenas", "gracias", "vale", "ok", "okay", "adios", "adiós",
        "perfecto", "genial", "entendido", "si", "sí", "no",
    }
    _IDENTITY_MARKERS = ("me llamo ", "mi nombre es ", "soy ", "vivo en ")
    _STRUCTURED_USER_MARKERS = ("mi bici es ", "mi color favorito es ")
    _PREFERENCE_MARKERS = ("me gusta ", "prefiero ", "odio ", "me encanta ")
    _EVENT_MARKERS = ("mañana ", "pasado mañana", "tengo cita", "tengo reunión", "tengo que ")
    _AGE_RE = re.compile(r"\btengo\s+\d{1,3}\s+años\b", re.I)

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def evaluate(self, text: str) -> GateDecision:
        normalized = " ".join(text.lower().strip().split())
        tokens = _TOKEN_RE.findall(normalized)
        reasons: List[str] = []

        if not tokens:
            return GateDecision(False, 0.0, MemoryKind.WORKING, ["empty"])

        if normalized in self._TRIVIAL or (len(tokens) <= 2 and all(t in self._TRIVIAL for t in tokens)):
            return GateDecision(False, 0.05, MemoryKind.WORKING, ["trivial"])

        score = 0.25
        kind = MemoryKind.EPISODIC

        if any(marker in normalized for marker in self._IDENTITY_MARKERS):
            score += 0.55
            kind = MemoryKind.USER_MODEL
            reasons.append("user_identity")

        if any(marker in normalized for marker in self._STRUCTURED_USER_MARKERS) or self._AGE_RE.search(normalized):
            score += 0.55
            kind = MemoryKind.USER_MODEL
            reasons.append("structured_user_fact")

        if any(marker in normalized for marker in self._PREFERENCE_MARKERS):
            score += 0.45
            kind = MemoryKind.USER_MODEL
            reasons.append("preference")

        if any(marker in normalized for marker in self._EVENT_MARKERS):
            score += 0.35
            kind = MemoryKind.EPISODIC
            reasons.append("future_event")

        if len(tokens) >= 8:
            score += 0.1
            reasons.append("information_density")

        if "?" in text or "¿" in text:
            score -= 0.20
            reasons.append("question_penalty")

        score = max(0.0, min(score, 1.0))
        return GateDecision(score >= self.threshold, score, kind, reasons or ["baseline"])


class InMemoryMemoryStore:
    """Deterministic store used for tests and as the simplest retrieval baseline."""

    def __init__(self):
        self._records: List[MemoryRecord] = []

    def add(self, record: MemoryRecord) -> MemoryRecord:
        self._records.append(record)
        return record

    def all(self) -> List[MemoryRecord]:
        return list(self._records)

    def search(self, query: str, top_k: int = 5) -> List[MemoryRecord]:
        query_tokens = Counter(_TOKEN_RE.findall(query.lower()))
        if not query_tokens:
            return []

        scored = []
        for record in self._records:
            record_tokens = Counter(_TOKEN_RE.findall(record.content.lower()))
            overlap = sum((query_tokens & record_tokens).values())
            union = sum((query_tokens | record_tokens).values()) or 1
            lexical = overlap / union
            score = lexical + (record.importance * 0.05)
            if overlap:
                scored.append((score, record))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:top_k]]
