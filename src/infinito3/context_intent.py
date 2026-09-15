"""Deterministic, entity-independent read intent for experimental context routing.

This is a bounded language parser, not a claim of universal intent recognition.
Unknown requests retain retrieval; only unambiguous standalone requests bypass it.
"""
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import FrozenSet, Optional, Tuple

from .temporal import parse_explicit_date, parse_weekday_date, parse_weekday_range


def normalize(text):
    return " ".join("".join(c for c in unicodedata.normalize("NFKD", text.lower())
                            if not unicodedata.combining(c)).split())


@dataclass(frozen=True)
class ContextIntent:
    version: str = "context_intent_v1"
    mode: str = "unknown"
    historical: bool = False
    recent: bool = False
    predicates: FrozenSet[str] = frozenset()
    window: Optional[Tuple[datetime, datetime]] = None
    future_only: bool = False

    @property
    def retrieve(self):
        return self.mode != "standalone"


def resolve_context_intent(text: str, now: Optional[datetime] = None) -> ContextIntent:
    now = now or datetime.now()
    q = normalize(text)
    question = bool("?" in q or re.match(
        r"^(what|which|where|when|how|is|are|do i|did i|tell me|name|return|give|explain|"
        r"define|calculate|dime|que|cual|donde|cuando|explica|calcula|nombra)\b", q))
    historical = bool(re.search(
        r"\b(before|previous\w*|histor\w*|used to|no longer|stop(?:ped)?|dropped|retracted|"
        r"lost (?:interest|its appeal)|does(?:n't| not) interest|antes|anterior\w*|"
        r"ya no|deje|abandone)\b", q))
    recent = bool(re.search(r"\b(newer|recent\w*|new|nuev\w*|reciente\w*)\b", q))
    personal = bool(re.search(r"\b(my|mine|me|i|mi|mis|mio|nuestro)\b", q))
    mutation = bool(re.search(r"\b(remember this|mark|cancel|move|reschedule|"
                              r"recuerda esto|marca|cancela|guarda)\b", q))
    if question and not personal and not mutation:
        arithmetic = bool(re.search(r"\d+\s*(?:[+*/×÷−-]|plus|times|divided by|minus|por|mas|entre)\s*\d+|\d+\s+(?:squared|al cuadrado)", q))
        general = bool(re.match(r"^(?:what is (?:a |an |the capital)|why does|"
                                r"name (?:one fact|the capital)|give (?:me )?one fact|"
                                r"explain|define|explica|define|dime la capital)\b", q))
        if arithmetic or general:
            return ContextIntent(mode="standalone")
    if not question:
        return ContextIntent(historical=historical)
    goals = bool(re.search(r"\b(calendar|agenda|commitment\w*|appointment\w*|"
                           r"goals?|tasks?|scheduled|pending|compromiso\w*|cita\w*|"
                           r"pendiente\w*|tarea\w*|objetivo\w*)\b", q))
    preferences = bool(re.search(r"\b(hobb\w*|interest\w*|preference\w*|activit\w*|"
                                 r"enjoy\w*|aficion\w*|actividad\w*|preferencia\w*)\b|me gusta|gustan", q))
    predicates = set()
    if personal or historical or "current" in q:
        for predicate, pattern in (
            ("pet_name", r"\b(pet|parrot|dog|cat|mascota|perro|gato|loro)\b"),
            ("location", r"\b(city|home|location|live|lived|ciudad|vivo|vivia|ubicacion)\b"),
            ("bike", r"\b(bike|bicycle|bici|bicicleta)\b"),
            ("studying_language", r"\b(language|idioma|lengua)\b"),
            ("occupation", r"\b(occupation|job|profession|work|trabajo|profesion)\b"),
            ("favorite_color", r"\b(colou?r|color)\b"),
            ("age", r"\b(age|old|edad|anos)\b"),
        ):
            if re.search(pattern, q): predicates.add(predicate)
        name_scope = re.sub(r"\b(?:pet|parrot|dog|cat|mascota)\s+name\b", "", q)
        if re.search(r"\b(name|nombre|llamo|llamabas)\b|call me", name_scope):
            predicates.add("name")
    if re.search(r"(?:verification|test) phrase|frase de (?:prueba|verificacion)", q):
        predicates.update(("test_phrase", "verification_phrase"))
    window = None
    future = bool(re.search(r"\b(future|upcoming|futur\w*)\b", q))
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
            if "this morning" in q or "esta manana" in q: end = start + timedelta(hours=12)
            window = (start, end)
    return ContextIntent(mode="goals" if goals else "preferences" if preferences else "facts" if predicates else "unknown",
                         historical=historical, recent=recent, predicates=frozenset(predicates),
                         window=window, future_only=future)
