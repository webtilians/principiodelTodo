"""Pure helpers for deterministic preference ordering."""

_ORDER_MARKERS = (
    "most recently", "most recent", "latest", "newest", "recently",
)
_ADD_MARKERS = (
    "did i add", "have i added", "did i start", "have i started",
    "did i take up", "have i taken up",
)


def membership_query(query: str) -> str:
    focused = " ".join(str(query).split())
    lowered = focused.lower()
    for marker in _ORDER_MARKERS + _ADD_MARKERS:
        lowered = lowered.replace(marker, " ")
    return " ".join(lowered.split()) or focused


def choose_latest(items, key):
    if not items:
        return []
    return [max(items, key=key)]
