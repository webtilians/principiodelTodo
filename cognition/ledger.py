"""What the engine may spend each day, and what it has spent.

Before every call the engine reserves a high estimate of its cost (the prompt at a low characters-per-token rate, plus
the whole output allowance, thinking included). If the day's budget cannot hold it, the call is not made. After the
call, the real usage is charged. A model with no known price is never called live, since its cost cannot be bounded.
The provider's own account limit is the last line of defence and stays the person's to set.
"""
import json
from datetime import date
from pathlib import Path
from typing import Optional


class UnknownPrice(Exception):
    pass


def prices_of(config: dict, model: str):
    try:
        price_in, price_out = config["prices_usd_per_million"][model]
        return float(price_in), float(price_out)
    except (KeyError, TypeError, ValueError):
        raise UnknownPrice(model) from None


def estimate_eur(config: dict, model: str, prompt_chars: int, max_output: int) -> float:
    price_in, price_out = prices_of(config, model)
    tokens_in = prompt_chars / float(config["chars_per_token"])
    return (tokens_in * price_in + max_output * price_out) / 1e6 * float(config["usd_to_eur"])


def cost_eur(config: dict, model: str, usage: dict) -> float:
    price_in, price_out = prices_of(config, model)
    out = usage.get("output", 0) + usage.get("thinking", 0)
    return (usage.get("input", 0) * price_in + out * price_out) / 1e6 * float(config["usd_to_eur"])


class Ledger:
    def __init__(self, folder: Path, config: dict):
        self.path, self.config = folder / "ledger.json", config
        try:
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.data = {"start": None, "days": {}}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def cap(self, day: date) -> float:
        """The day's budget: the pilot's first seven days as planned, then budget_eur_after, with no last day."""
        start = date.fromisoformat(self.data["start"]) if self.data.get("start") else day
        index = (day - start).days
        schedule = self.config["budget_eur"]
        if 0 <= index < len(schedule):
            return float(schedule[index])
        return float(self.config["budget_eur_after"])

    def spent(self, day: date) -> float:
        return float(self.data["days"].get(day.isoformat(), {}).get("spent_eur", 0.0))

    def remaining(self, day: date) -> float:
        return max(0.0, self.cap(day) - self.spent(day))

    def charge(self, day: date, eur: float, *, overrun: Optional[float] = None) -> None:
        if not self.data.get("start"):
            self.data["start"] = day.isoformat()
        entry = self.data["days"].setdefault(day.isoformat(), {"spent_eur": 0.0, "calls": 0})
        entry["spent_eur"] = round(entry["spent_eur"] + eur, 6)
        entry["calls"] += 1
        if overrun:
            entry["overrun_eur"] = round(entry.get("overrun_eur", 0.0) + overrun, 6)
        self._save()
