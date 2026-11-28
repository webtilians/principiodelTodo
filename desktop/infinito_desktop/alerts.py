"""When to tell the person: only when a check gets worse, into a warning or a failure, and at most once a day for each.

A check that stays bad is not news, and «no lo sé» alone is not worth a notification: the icon's colour says it.
What it last saw is kept in the app's state file, so a restart does not tell the same thing again.
"""
from typing import Dict, Iterable, List

from .checks import FAIL, RANK, WARN, Check


def due(checks: Iterable[Check], seen: Dict[str, str], told: Dict[str, str], today: str) -> List[Check]:
    """The checks to notify now. seen is each check's state at the last look, told the day each was last notified."""
    news = []
    for check in checks:
        before = seen.get(check.key)
        worse = before is None or RANK[check.state] > RANK.get(before, 0)
        if check.state in (WARN, FAIL) and worse and told.get(check.key) != today:
            news.append(check)
    return news


def remember(checks: Iterable[Check], seen: Dict[str, str], told: Dict[str, str], notified: Iterable[Check],
             today: str) -> None:
    for check in checks:
        seen[check.key] = check.state
    for check in notified:
        told[check.key] = today
