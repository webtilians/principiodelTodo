"""The engine's own referee, with no model in it: what a role says must rest on memories it was actually shown.

A model can cite a memory and say something it does not say (INFINITO saw it in F12-E). Whether the words match is
for the supervisor and the person to judge; what a program can check for certain is that every id cited exists in
what that role was given. A claim citing an id outside it, or none, is set aside as NOT_IN_EVIDENCE and never reaches
the person's memory. The same check applies to the supervisor: its reflection keeps only sources that exist, and is
not published if none is left.
"""
import re
from typing import Dict, Iterable, List, Sequence, Set, Tuple

MEMORY_ID = re.compile(r"\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\]")
REFLECTION_ID = re.compile(r"^\[([0-9a-f]{8})\] \(", re.MULTILINE)
OPEN_MEMORY = re.compile(r"\[([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\] "
                         r"\(\d{4}-\d{2}-\d{2}, (?:pending|to_verify)\b")
RELATIONS = ("expands", "reconsiders", "questions")
SCOPES = ("self", "user", "relationship", "project", "decision")


def memory_ids(text: str) -> List[str]:
    return list(dict.fromkeys(MEMORY_ID.findall(text or "")))


def reflection_ids(text: str) -> List[str]:
    return list(dict.fromkeys(REFLECTION_ID.findall(text or "")))


def _cited(item: dict, key: str) -> List[str]:
    value = item.get(key)
    return [str(v) for v in value] if isinstance(value, list) else []


def grounded(found: Iterable[dict], key: str, known: Set[str]) -> Tuple[List[dict], List[dict]]:
    """Split claims into those whose every cited id is known (and cite at least one) and those set aside."""
    kept, aside = [], []
    for item in found:
        cited = _cited(item, key)
        if cited and all(c in known for c in cited):
            kept.append(item)
        else:
            aside.append({**item, "rejected": "NOT_IN_EVIDENCE",
                          "unknown_ids": [c for c in cited if c not in known] or ["(none cited)"]})
    return kept, aside


def open_ids(text: str) -> Set[str]:
    """The memories still open (pending or to_verify), as INFINITO's lines say: when one is closed it is forgotten,
    and INFINITO deletes every reflection resting on it."""
    return set(OPEN_MEMORY.findall(text or ""))


def reflection(found, known_memories: Set[str], known_reflections: Set[str],
               avoid: Set[str] = frozenset()) -> Tuple[dict, List[str]]:
    """The supervisor's reflection as INFINITO will take it, and what had to be dropped from it. Empty dict: none.
    Sources in avoid (open memories) are dropped when others remain, so closing a pending item does not take the
    reflection with it; if it rests on nothing else, it keeps them and says so."""
    if not isinstance(found, dict) or not str(found.get("text") or "").strip():
        return {}, ["no reflection"]
    dropped = []
    sources = [s for s in _cited(found, "sources") if s in known_memories]
    dropped += [f"source {s} not in evidence" for s in _cited(found, "sources") if s not in known_memories]
    if not sources:
        return {}, dropped + ["no source left: not published"]
    lasting = [s for s in sources if s not in avoid]
    if lasting and len(lasting) < len(sources):
        dropped.append(f"{len(sources) - len(lasting)} pending source(s) left out, so closing them does not delete it")
        sources = lasting
    elif not lasting:
        dropped.append("it rests only on pending memories: it will go when they are closed")
    relates = []
    for link in found.get("relates_to") or []:
        if isinstance(link, dict) and link.get("id") in known_reflections and link.get("how") in RELATIONS:
            relates.append({"id": link["id"], "how": link["how"]})
        else:
            dropped.append(f"relation {link!r} not to a reflection it was shown")
    scope = found.get("scope") if found.get("scope") in SCOPES else "project"
    questions = [str(q)[:300] for q in (found.get("questions") or []) if str(q).strip()][:5]
    kept = {"reflection": str(found["text"]).strip()[:2000], "sources": sources, "scope": scope,
            "advice": (str(found.get("advice") or "").strip()[:600] or None), "questions": questions or None,
            "relates_to": relates or None, "project": (str(found.get("project") or "").strip() or None)}
    return kept, dropped


def coverage(inventory: Sequence[str], cited_by: Dict[str, Iterable[str]], *, reported_new: int,
             shown_new: int, truncated: Dict[str, bool]) -> dict:
    """Which of the day's memories each role cited, which none did, and what could not be seen at all."""
    whole = list(dict.fromkeys(inventory))
    seen = {role: sorted(set(ids) & set(whole)) for role, ids in cited_by.items()}
    anyone = set().union(*seen.values()) if seen else set()
    limits = []
    if reported_new > shown_new:
        limits.append(f"INFINITO reported {reported_new} new memories and the engine could read {shown_new}")
    limits += [f"the {role}'s input was cut to fit its limit" for role, cut in truncated.items() if cut]
    return {"inventory": len(whole), "cited_by": {role: len(ids) for role, ids in seen.items()},
            "cited_by_nobody": [i for i in whole if i not in anyone], "limits": limits,
            "complete": not limits}
