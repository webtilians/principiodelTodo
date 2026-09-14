import re
import unicodedata
from datetime import date, datetime, timedelta
from typing import Optional, Tuple


_MONTHS = {
    "enero": 1,
    "january": 1,
    "febrero": 2,
    "february": 2,
    "marzo": 3,
    "march": 3,
    "abril": 4,
    "april": 4,
    "mayo": 5,
    "may": 5,
    "junio": 6,
    "june": 6,
    "julio": 7,
    "july": 7,
    "agosto": 8,
    "august": 8,
    "septiembre": 9,
    "setiembre": 9,
    "september": 9,
    "octubre": 10,
    "october": 10,
    "noviembre": 11,
    "november": 11,
    "diciembre": 12,
    "december": 12,
}

_WEEKDAYS = {
    "lunes": 0,
    "monday": 0,
    "martes": 1,
    "tuesday": 1,
    "miercoles": 2,
    "wednesday": 2,
    "jueves": 3,
    "thursday": 3,
    "viernes": 4,
    "friday": 4,
    "sabado": 5,
    "saturday": 5,
    "domingo": 6,
    "sunday": 6,
}


def normalize_calendar_text(text: str) -> str:
    return " ".join(
        "".join(
            char
            for char in unicodedata.normalize("NFKD", text.lower())
            if not unicodedata.combining(char)
        ).split()
    )


def parse_explicit_date(text: str, now: datetime) -> Optional[date]:
    """Parse a human day/month date without silently changing its year."""
    normalized = normalize_calendar_text(text)
    match = re.search(
        r"\b(\d{1,2})\s+(?:de\s+)?([a-z]+)(?:\s+(?:de\s+)?(\d{4}))?\b",
        normalized,
    )
    if not match:
        return None

    day = int(match.group(1))
    month = _MONTHS.get(match.group(2))
    if month is None:
        return None
    year = int(match.group(3)) if match.group(3) else now.year
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_weekday_date(text: str, now: datetime) -> Optional[date]:
    """Resolve the next occurrence of a weekday, including today itself."""
    normalized = normalize_calendar_text(text)
    matched_weekday = None
    for token, weekday in _WEEKDAYS.items():
        if re.search(rf"\b{re.escape(token)}\b", normalized):
            matched_weekday = weekday
            break
    if matched_weekday is None:
        return None

    days_ahead = (matched_weekday - now.weekday()) % 7
    return now.date() + timedelta(days=days_ahead)


def parse_weekday_range(text: str, now: datetime) -> Optional[Tuple[date, date]]:
    """Resolve an explicit inclusive weekday range in Spanish or English.

    Examples: ``from Wednesday through Sunday``, ``from Tuesday to Friday``,
    ``de miércoles a domingo`` and ``entre jueves y sábado``.  The first day is
    bound to its next occurrence (today is allowed); the second day is resolved
    forward from that start so ranges can cross a week boundary without domain
    assumptions.
    """
    normalized = normalize_calendar_text(text)
    weekday_pattern = "|".join(sorted(_WEEKDAYS, key=len, reverse=True))
    patterns = (
        rf"\bfrom\s+({weekday_pattern})\s+(?:to|through|thru|until)\s+({weekday_pattern})\b",
        rf"\bde\s+({weekday_pattern})\s+(?:a|hasta)\s+({weekday_pattern})\b",
        rf"\bentre\s+({weekday_pattern})\s+y\s+({weekday_pattern})\b",
    )
    match = next((candidate for pattern in patterns if (candidate := re.search(pattern, normalized))), None)
    if match is None:
        return None

    start_weekday = _WEEKDAYS[match.group(1)]
    end_weekday = _WEEKDAYS[match.group(2)]
    start = now.date() + timedelta(days=(start_weekday - now.weekday()) % 7)
    end = start + timedelta(days=(end_weekday - start_weekday) % 7)
    return start, end


def parse_weekend_window(text: str, now: datetime) -> Optional[Tuple[date, date]]:
    """Resolve conversational weekend planning to Friday-through-Sunday."""
    normalized = normalize_calendar_text(text)
    if "fin de semana" not in normalized and "weekend" not in normalized:
        return None

    week_start = now.date() - timedelta(days=now.weekday())
    shift = 7 if any(
        marker in normalized
        for marker in ("proximo fin de semana", "siguiente fin de semana", "next weekend")
    ) else 0
    friday = week_start + timedelta(days=4 + shift)
    sunday = friday + timedelta(days=2)
    return friday, sunday


def calendar_reference_present(text: str) -> bool:
    normalized = normalize_calendar_text(text)
    if any(re.search(rf"\b{re.escape(token)}\b", normalized) for token in _WEEKDAYS):
        return True
    if any(re.search(rf"\b{re.escape(token)}\b", normalized) for token in _MONTHS):
        return True
    return any(
        marker in normalized
        for marker in (
            "hoy",
            "today",
            "manana",
            "tomorrow",
            "pasado manana",
            "day after tomorrow",
            "semana que viene",
            "proxima semana",
            "next week",
            "fin de semana",
            "weekend",
        )
    )
