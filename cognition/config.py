"""Where the engine keeps its things, which model plays each role, and how much it may spend each day.

The defaults below are a starting point; ~/.infinito/cognition/config.json overrides any of them. Prices are an
upper bound per million tokens in US dollars, set on the high side on purpose: the budget then stops a run too early,
never too late. A model without a price is never called live.
"""
import copy
import json
import os
from pathlib import Path

ROLES = ("observer", "investigator", "supervisor", "baseline")

DEFAULTS = {
    "server": "https://infinitomemory.com",
    "app_name": "INFINITO Motor cognitivo",
    # The day's review is due once the local clock passes this hour; a day the computer was off is caught up later.
    "review_hour": 21,
    # Between reviews, this many new memories wake the observer for a pass of its own.
    "observer_threshold": 10,
    # Night hours [from, until), local clock, or None. When set, nothing calls a model outside them: the day's review
    # runs at the first tick inside, and until the hours end the supervisor studies older memories, a batch at a time
    # (study_batch_chars each), then closes the night with a reflection (study_close_minutes are kept for it).
    # The night's findings grow with every pass; the closing call reads at most study_close_chars of them, so a long
    # night still fits a model's context and the time kept for it.
    "night_hours": None,
    "study_batch_chars": 45000,
    "study_pass_minutes": 20,
    "study_close_minutes": 25,
    "study_close_chars": 60000,
    # How many times over all memories one night may read, or None for as long as the hours last: a fast paid or
    # free-quota API would otherwise spend a whole quota in one night (1 = each memory at most once a night).
    "study_rounds_per_night": None,
    # Euros per day: the first seven days as the pilot planned them, then budget_eur_after for every day after.
    # There is no last day: the engine runs until the person pauses it.
    "budget_eur": [0.5, 0.5, 1.0, 1.0, 2.0, 2.0, 1.0],
    "budget_eur_after": 1.0,
    # Dollars to euros, rounded up so a cost is never underestimated (an approximation, labelled as such).
    "usd_to_eur": 1.0,
    # Characters per token used to estimate a prompt before sending it: low, so the estimate is high.
    "chars_per_token": 3.0,
    "max_checks": 5,
    "missions": [],
    # No model is chosen until the person chooses one (python -m cognition models ...): provider is "anthropic" or
    # "openai" (OpenAI, or any API that speaks its chat completions, with base_url and key_name).
    "roles": {
        "observer": {"provider": None, "model": None, "max_output": 6000, "max_input_chars": 200000},
        "investigator": {"provider": None, "model": None, "max_output": 8000, "max_input_chars": 200000},
        "supervisor": {"provider": None, "model": None, "max_output": 24000, "max_input_chars": 450000},
        "baseline": {"provider": None, "model": None, "max_output": 4000, "max_input_chars": 200000},
    },
    # Input and output (thinking included) in USD per million tokens, for every model the engine may call live:
    # the person writes them from the provider's price page, rounded up. A model missing here is never called.
    "prices_usd_per_million": {},
}


def home() -> Path:
    return Path(os.environ.get("INFINITO_COGNITION_HOME") or Path.home() / ".infinito" / "cognition")


def _merge(base: dict, extra: dict) -> dict:
    merged = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load(folder: Path) -> dict:
    """The defaults with the person's config.json over them."""
    path = folder / "config.json"
    try:
        extra = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        extra = {}
    return _merge(DEFAULTS, extra)


def save_override(folder: Path, key: str, value) -> None:
    """Change one setting in config.json, keeping the rest of what the person wrote there."""
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "config.json"
    try:
        current = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        current = {}
    current[key] = value
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(current, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
