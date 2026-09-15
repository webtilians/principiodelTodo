"""Deterministic, entity-independent read intent for experimental context routing.

ContextIntent v2 centralizes bounded language routing for retrieval, historical
state, goals, preferences and unambiguous standalone requests. Unknown requests
still retain retrieval; the parser deliberately prefers false negatives over
silently hiding potentially relevant personal state.
"""
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import FrozenSet, Optional, Tuple

from .temporal import parse_explicit_date, parse_weekday_date, parse_weekday_range


_NOTE_PREDICATE_ALIASES = frozenset((
    "test_phrase", "verification_phrase", "literal_test_phrase",
    "literal_verification_phrase",
))


def normalize(text):
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", text.lower())
                            if not unicodedata.combining(c)).split())


def detect_history_cue(text: str) -> Optional[str]:
    """Return a semantic operator for historical wording, without entity names."""
    q = normalize(text)
    cues = (
        ("lost_interest", r"\b(?:lost|lose|lost my) interest\b|\bperdi interes\b"),
        ("lost_appeal", r"\b(?:lost|lose|losing) (?:its |the )?appeal\b|\bdejo de atraerme\b"),
        ("no_longer_interest", r"\b(?:doesn'?t|does not|no longer) interest\b|\bya no me interesa\b"),
        ("no_longer_enjoy", r"\b(?:no longer|don'?t|do not) enjoy\b|\bya no disfruto\b"),
        ("stopped", r"\b(?:stop|stopped|stopping)\b|\b(?:deje|pare)\b"),
        ("dropped", r"\b(?:drop|dropped|abandon(?:ed)?)\b|\babandone\b"),
        ("predecessor", r"\b(?:precede[ds]?|preceding|predecessor|previous|previously|prior|former|formerly|earlier|came before|before)\b|\b(?:anterior|antes|precedio|previo)\b"),
    )
    for cue, pattern in cues:
        if re.search(pattern, q):
            return cue
    return None


@dataclass(frozen=True)
class ContextIntent:
    version: str = "context_intent_v2"
    mode: str = "unknown"
    historical: bool = False
    recent: bool = False
    predicates: FrozenSet[str] = frozenset()
    window: Optional[Tuple[datetime, datetime]] = None
    future_only: bool = False
    history_cue: Optional[str] = None
    goal_status: Optional[str] = None
    goal_status_check: bool = False

    @property
    def retrieve(self):
        return self.mode != "standalone"


def resolve_context_intent(text: str, now: Optional[datetime] = None) -> ContextIntent:
    now = now or datetime.now()
    q = normalize(text)
    question = bool("?" in q or re.match(
        r"^(what|which|where|when|how|is|are|do i|did i|tell me|recall|remind me|show|list|"
        r"name|return|give|explain|define|calculate|dime|que|cual|donde|cuando|explica|"
        r"calcula|nombra|recuerda|muestra|lista)\b", q))
    history_cue = detect_history_cue(q)
    historical = bool(history_cue or re.search(
        r"\b(histor\w*|used to|no longer|retracted|ya no|historico|historica)\b", q))
    recent = bool(re.search(r"\b(newer|recent\w*|new|nuev\w*|reciente\w*)\b", q))
    personal = bool(re.search(r"\b(my|mine|me|i|our|mi|mis|mio|mios|nuestro|nuestra)\b", q))
    mutation = bool(re.search(r"\b(remember|store|save|mark|cancel|move|reschedule|"
                              r"recuerda|guarda|marca|cancela|mueve|reprograma)\b", q))

    if question and not personal and not mutation:
        arithmetic = bool(re.search(
            r"\d+\s*(?:[+*/×÷−-]|plus|times|divided by|minus|por|mas|entre)\s*\d+|"
            r"\d+\s+(?:squared|al cuadrado)", q))
        general = bool(re.match(
            r"^(?:what is (?:a |an |the capital)|why does|name (?:one fact|the capital)|"
            r"give (?:me )?one fact|explain|define|explica|define|dime la capital)\b", q))
        if arithmetic or general:
            return ContextIntent(mode="standalone")

    if not question:
        return ContextIntent(historical=historical, history_cue=history_cue)

    explicit_goals = bool(re.search(
        r"\b(calendar|agenda|commitment\w*|appointment\w*|goals?|tasks?|scheduled|pending|"
        r"compromiso\w*|cita\w*|pendiente\w*|tarea\w*|objetivo\w*)\b", q))
    closed_status = bool(re.search(
        r"\b(completed|complete|closed|done|finished|cancelled|canceled|resolved|"
        r"completad\w*|cerrad\w*|terminad\w*|cancelad\w*)\b", q))
    open_status = bool(re.search(
        r"\b(still open|open|pending|outstanding|remaining|unfinished|"
        r"sigue abierto|sigue pendiente|abiert\w*|pendiente\w*)\b", q))
    lifecycle_check = bool(
        re.match(r"^(?:is|are|was|were|esta|estan|sigue|siguen)\b", q)
        and (closed_status or open_status)
    )
    lifecycle_contrast = bool(closed_status and open_status)
    goals = explicit_goals or lifecycle_check or lifecycle_contrast
    goal_status = "any" if closed_status and open_status else "closed" if closed_status else "open" if open_status else None

    preferences = bool(re.search(
        r"\b(hobb\w*|interest\w*|preference\w*|activit\w*|pastime\w*|leisure|enjoy\w*|"
        r"aficion\w*|actividad\w*|pasatiempo\w*|preferencia\w*)\b|me gusta|gustan", q))

    predicates = set()
    if personal or historical or "current" in q or "profile" in q or "perfil" in q:
        for predicate, pattern in (
            ("pet_name", r"\b(pet|parrot|dog|cat|mascota|perro|gato|loro)\b"),
            ("location", r"\b(city|home|location|residence|residency|live|lived|based|ciudad|vivo|vivia|residencia|ubicacion)\b"),
            ("bike", r"\b(bike|bicycle|ride|bici|bicicleta|montura)\b"),
            ("studying_language", r"\b(language|idioma|lengua)\b"),
            ("occupation", r"\b(occupation|job|profession|work|employment|trabajo|profesion|empleo)\b"),
            ("favorite_color", r"\b(colou?r|color)\b"),
            ("age", r"\b(age|old|edad|anos)\b"),
        ):
            if re.search(pattern, q):
                predicates.add(predicate)
        name_scope = re.sub(
            r"\b(?:pet|parrot|dog|cat|mascota|perro|gato|loro)\s+(?:name|nombre)\b",
            "", q)
        if re.search(r"\b(name|nombre|llamo|llamabas)\b|call me", name_scope):
            predicates.add("name")

    if re.search(
        r"(?:literal\s+)?(?:verification|test) phrase|frase (?:literal )?de (?:prueba|verificacion)", q):
        predicates.update(_NOTE_PREDICATE_ALIASES)

    window = None
    future = bool(re.search(r"\b(future|upcoming|futur\w*|proxim\w*)\b", q))
    if goals:
        focus = text.rsplit("¿", 1)[-1]
        span = parse_weekday_range(focus, now)
        target = parse_explicit_date(focus, now) or parse_weekday_date(focus, now)
        if span:
            window = (datetime.combine(span[0], datetime.min.time()),
                      datetime.combine(span[1] + timedelta(days=1), datetime.min.time()))
        elif target and re.search(r"\b(before|antes de)\b", q):
            window = (datetime.min, datetime.combine(target, datetime.min.time()))
        elif target or re.search(r"\b(today|hoy|this morning|esta manana)\b", q):
            day = target or now.date()
            start = datetime.combine(day, datetime.min.time())
            end = start + timedelta(days=1)
            if "this morning" in q or "esta manana" in q:
                end = start + timedelta(hours=12)
            window = (start, end)

    mode = "goals" if goals else "preferences" if preferences else "facts" if predicates else "unknown"
    return ContextIntent(
        mode=mode, historical=historical, recent=recent, predicates=frozenset(predicates),
        window=window, future_only=future, history_cue=history_cue, goal_status=goal_status,
        goal_status_check=lifecycle_check,
    )
