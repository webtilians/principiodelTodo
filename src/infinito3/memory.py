import re
from collections import Counter
from typing import List

from .types import GateDecision, MemoryKind, MemoryRecord


_TOKEN_RE = re.compile(r"[\wáéíóúüñ]+", re.I)

# A sentence ends at . ! ? or … followed by a space and a capital, a digit or
# an opening quote; "Dr. Pardo" and "1.150" stay whole.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?…])[\"'”’»)\]]*\s+(?=[\"'“‘«(¿¡]*[A-ZÁÉÍÓÚÑ0-9])")
_ABBREVIATIONS = {
    "dr", "dra", "sr", "sra", "srta", "mr", "mrs", "ms", "st", "prof", "vs", "etc", "approx",
    "no", "nº", "av", "avda", "ud", "uds", "jr", "inc", "ltd", "co", "mt", "ft",
}


# "I support a food bank each winter, so could you suggest...?" states a fact.
_STATEMENT_THEN_QUESTION = re.compile(
    r"^(?:i|i'm|i've|i'd|my|we|we're|we've|our)\b[^?]{8,}?,\s*(?:so|but|and|because)\b", re.I
)


_LEAD_WORDS = {"y", "pero", "oye", "entonces", "and", "so", "hey"}
_PREPOSITIONS = {"con", "a", "de", "en", "para", "por", "sobre", "desde", "hasta",
                 "with", "for", "about", "to", "from"}
_ASKING = {"qué", "quién", "quiénes", "dónde", "adónde", "cuándo", "cómo", "cuál", "cuáles",
           "cuánto", "cuánta", "cuántos", "cuántas"}
# "¡Qué bien lo pasé!" exclaims, it does not ask.
_EXCLAIMED = {"bien", "mal", "pena", "suerte", "bonito", "bonita", "rico", "rica", "guay", "gracia",
              "horror", "bueno", "buena", "maravilla"}
_ASKING_UNACCENTED = {"que", "quien", "quienes", "donde", "adonde", "cuando", "como", "cual", "cuales",
                      "cuanto", "cuanta", "cuantos", "cuantas"}
# Without an accent the same words start statements ("cuando llegue te llamo", "como te
# dije, vivo en Madrid"), so they ask only before a verb that usually asks about the user.
_ASKING_VERBS = {"sabes", "recuerdas", "conoces", "crees", "tienes", "puedes", "es", "son", "era",
                 "fue", "voy", "vivo", "trabajo", "estudio", "estoy", "soy", "tengo", "hago", "hice",
                 "dije", "llamo", "hay"}
_WH = {"what", "who", "whom", "whose", "where", "when", "why", "which", "how"}
_WH_CONTRACTED = {"what's", "who's", "where's", "when's", "how's", "whats", "wheres", "hows"}
_AUXILIARIES = {"is", "are", "was", "were", "am", "do", "does", "did", "can", "could", "should",
                "would", "will", "have", "has", "had", "may", "might", "shall"}
_HOW_MUCH = {"much", "many", "long", "old", "far", "often"}
# Requests for information addressed to the assistant ask too: "dime dónde vivo",
# "haz una lista de todo lo que sabes de mí", "tell me my dentist's name".
_TELL_ME = {"dime", "dígame", "cuéntame", "cuentame", "muéstrame", "muestrame", "enséñame", "enseñame",
            "explícame", "explicame", "enumera", "list"}
_TELL_ME_EN = {"tell", "show", "give"}
_SUBJECTS = {"i", "you", "my", "your", "we", "our", "he", "she", "it", "they", "there", "this", "that"}


def asks_without_marks(sentence: str) -> bool:
    """True for a question typed without ? or ¿, as most people chat: "con quien voy el jueves".

    It errs towards statements: a question taken for a statement is stored, a statement
    taken for a question is lost.
    """
    words = re.findall(r"[\wáéíóúüñ']+", sentence.casefold().replace("’", "'"))
    while words and words[0] in _LEAD_WORDS:
        words = words[1:]
    if words and words[0] in _PREPOSITIONS:
        words = words[1:]
    if len(words) < 2:
        return False
    first, second = words[0], words[1]
    short = len(words) <= 12 and "," not in sentence
    if first in _TELL_ME or (first in _TELL_ME_EN and second == "me") or words[:3] in (
            ["haz", "una", "lista"], ["make", "a", "list"]):
        return short
    if first in _ASKING:
        return second not in _EXCLAIMED
    if first in _ASKING_UNACCENTED:
        return second in _ASKING_VERBS and len(words) <= 10 and "," not in sentence
    if first in _WH_CONTRACTED:
        return True
    if first in _WH:
        return second in _AUXILIARIES or (first == "how" and second in _HOW_MUCH)
    return first in _AUXILIARIES and second in _SUBJECTS


def is_question(sentence: str) -> bool:
    """True for a sentence that only asks; one that states something first is not."""
    s = sentence.strip().rstrip("\"'”’»)").strip()
    if not (s.endswith("?") or s.startswith("¿")):
        return "¿" not in s and "?" not in s and asks_without_marks(s)
    if s.find("¿") > 3:  # "Tengo asma, ¿debería llevar el inhalador?"
        return False
    return not _STATEMENT_THEN_QUESTION.search(s.replace("’", "'"))


def split_sentences(text: str) -> List[str]:
    """Split a message into sentences."""
    sentences: List[str] = []
    for line in re.split(r"\n+", text):
        current: List[str] = []
        for piece in _SENTENCE_BREAK.split(line.strip()):
            piece = piece.strip()
            if not piece:
                continue
            words = current[-1][:-1].split() if current and current[-1].endswith(".") else []
            if words and words[-1].lower() in _ABBREVIATIONS:
                current[-1] = f"{current[-1]} {piece}"
            else:
                current.append(piece)
        sentences.extend(current)
    return sentences


class RuleBasedMemoryGate:
    """Transparent baseline gate for INFINITO 3.0.

    Every safe statement is stored: in E10 the facts that mattered later were
    decisions in the user's own words, and a marker list missed all of them.
    Markers only raise importance and choose the memory kind. Questions and
    acknowledgements are not stored; secrets never reach the gate and the
    engine never persists PII.

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

    def __init__(self, threshold: float = 0.55, min_statement_tokens: int = 3):
        self.threshold = threshold
        self.min_statement_tokens = min_statement_tokens

    def evaluate(self, text: str) -> GateDecision:
        normalized = " ".join(text.lower().strip().split())
        tokens = _TOKEN_RE.findall(normalized)
        reasons: List[str] = []

        if not tokens:
            return GateDecision(False, 0.0, MemoryKind.WORKING, ["empty"])

        if normalized in self._TRIVIAL or all(t in self._TRIVIAL for t in tokens):
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

        sentences = split_sentences(text)
        is_question = "?" in text or "¿" in text or bool(sentences) and all(
            asks_without_marks(s) for s in sentences)
        if is_question:
            score -= 0.20
            reasons.append("question_penalty")

        score = max(0.0, min(score, 1.0))
        statement = not is_question and len(tokens) >= self.min_statement_tokens
        if statement and score < self.threshold:
            reasons.append("statement")
        return GateDecision(score >= self.threshold or statement, score, kind, reasons or ["baseline"])


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

    def forget_candidates(self, query: str, top_k=None) -> List[MemoryRecord]:
        """Every memory, those sharing words with the request first."""
        found = self.search(query, top_k=len(self._records))
        return found + [record for record in self._records if record not in found][: None if top_k is None else max(0, top_k - len(found))]

    def purge(self, memory_ids) -> int:
        """Delete memories for good."""
        doomed = set(memory_ids)
        before = len(self._records)
        self._records = [record for record in self._records if record.id not in doomed]
        return before - len(self._records)
