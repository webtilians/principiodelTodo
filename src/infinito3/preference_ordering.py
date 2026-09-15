"""Pure helpers for deterministic ordering."""


def choose_latest(items, key):
    if not items:
        return []
    return [max(items, key=key)]
