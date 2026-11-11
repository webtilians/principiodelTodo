"""The cognitive engine's daily reviews, for the person to read in full and decide on, on their memory page.

The engine (cognition/ in this repository) is one more app connected to the person's memory. Once a day its supervisor
reviews the day: what was new, what its children (observer, investigator) made of it, how each hypothesis held up,
its own reflection, and the guidelines it proposes for each role. cognition_report keeps that review here, sealed with
the memory's key (cognition.sealed), as the engine wrote it.

The memory page shows a notice on every page while a review is unread or a guideline waits, and the review in full under
«Motor cognitivo». There, and only there, signed in with their password, the person accepts, rejects or withdraws each
guideline. No tool can: cognition_decisions only tells the engine what the person decided, and the engine applies an
accepted guideline from its next call on, and drops a withdrawn one.

Forgetting reaches here too: when memories are forgotten, every part of a review that cites one of them goes (an audit
note, a hypothesis, the copy of a reflection), and their ids leave every list, so nothing forgotten can be rebuilt
from what the engine said about it.
"""
import html
import json
import os
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, TypeVar

FILE = "cognition.sealed"
MAGIC = b"CGN1"
ROLES = ("observer", "investigator", "supervisor")
VERDICTS = ("supported", "refuted", "untested")
DECISIONS = {"approve": "approved", "reject": "rejected", "revoke": "revoked"}
MAX_REVIEWS = 400
MAX_ITEMS = 40
MAX_TEXT = 2500
MAX_SHORT = 600
MAX_IDS = 60

TOOL_DESCRIPTIONS = {
    "cognition_report": "Only for INFINITO's cognitive engine: keep the day's review so the user can read it in full on "
                        "their memory page and decide on the guidelines it proposes. review is the engine's JSON report "
                        "(run, summary, reflection, coverage, audit, hypotheses, costs, proposals). No tool decides on "
                        "a guideline: only the user, on their page.",
    "cognition_decisions": "Only for INFINITO's cognitive engine: what the user decided, on their memory page, about each "
                           "guideline the engine proposed (pending, approved, rejected or revoked), as JSON.",
}
TOOL_HINTS = {
    "cognition_report": {"title": "Keep the engine's review", "readOnlyHint": False, "destructiveHint": False,
                         "idempotentHint": True, "openWorldHint": False},
    "cognition_decisions": {"title": "The user's decisions on the engine's guidelines", "readOnlyHint": True,
                            "openWorldHint": False},
}
NEEDS_ACCOUNT = "The cognitive engine needs an INFINITO account with sign-in; this memory uses a private address."

TEXT = {
    "es": {
        "nav_cognition": "Motor cognitivo", "cognition_title": "Motor cognitivo",
        "cognition_note": "Lo que tu motor cognitivo revisa cada día: lo que leyó, lo que pensaron el observador y el "
                          "investigador, cómo juzgó el supervisor cada hipótesis, su reflexión y lo que costó. Las pautas "
                          "que propone para cada rol no se aplican hasta que tú las aceptas aquí; una aceptada se puede "
                          "retirar.",
        "cognition_empty": "Tu motor cognitivo aún no ha dejado ninguna revisión.",
        "cognition_notice": "Tu motor cognitivo dejó una revisión nueva{when}{waiting}.",
        "cognition_notice_waiting": " con {n} pauta(s) esperando tu decisión",
        "cognition_notice_only_waiting": "Hay {n} pauta(s) del motor cognitivo esperando tu decisión.",
        "cognition_notice_go": "Léela y decide",
        "cognition_new": "Nueva", "cognition_by": "por",
        "cognition_summary": "Lo que te dice el supervisor", "cognition_reflection": "Su reflexión",
        "cognition_reflection_link": "verla en Reflexiones", "cognition_advice": "Consejo que se da",
        "cognition_questions": "Preguntas abiertas", "cognition_coverage": "Cobertura",
        "cognition_coverage_line": "Leyó {read} recuerdos nuevos (INFINITO informó de {reported}); sin citar por "
                                   "nadie: {nobody}.",
        "cognition_complete": "El supervisor tuvo el día entero.", "cognition_limits": "Límites",
        "cognition_audit": "Auditoría de los hijos", "cognition_hypotheses": "Hipótesis y veredictos",
        "cognition_refuted_by": "La refutaría", "cognition_rests_on": "Se apoya en",
        "cognition_verdicts": {"supported": "apoyada", "refuted": "refutada", "untested": "sin probar"},
        "cognition_hypothesis_note": "Una hipótesis apoyada sigue siendo una hipótesis.",
        "cognition_costs": "Coste", "cognition_costs_line": "{spent} € de un tope de {cap} € ese día.",
        "cognition_proposals": "Pautas que propone", "cognition_for": "para el",
        "cognition_roles": {"observer": "observador", "investigator": "investigador", "supervisor": "supervisor"},
        "cognition_why": "Por qué", "cognition_test": "Cómo se comprobaría",
        "cognition_status": {"pending": "esperando tu decisión", "approved": "aceptada", "rejected": "rechazada",
                             "revoked": "retirada"},
        "cognition_note_label": "Nota para el motor (opcional)",
        "cognition_approve": "Aceptar", "cognition_reject": "Rechazar", "cognition_revoke": "Retirar",
        "cognition_done": {"approved": "Pauta aceptada: el motor la aplicará desde su próxima llamada.",
                           "rejected": "Pauta rechazada: no se aplicará.",
                           "revoked": "Pauta retirada: el motor dejará de aplicarla."},
        "cognition_gone": "Esa pauta ya no está o ya se decidió.",
        "cognition_refused": "La nota parecía contener un secreto y no se guardó. No la repitas.",
    },
    "en": {
        "nav_cognition": "Cognitive engine", "cognition_title": "Cognitive engine",
        "cognition_note": "What your cognitive engine reviews each day: what it read, what the observer and the "
                          "investigator made of it, how the supervisor judged each hypothesis, its reflection and what "
                          "it cost. The guidelines it proposes for each role apply only once you accept them here; an "
                          "accepted one can be withdrawn.",
        "cognition_empty": "Your cognitive engine has not left any review yet.",
        "cognition_notice": "Your cognitive engine left a new review{when}{waiting}.",
        "cognition_notice_waiting": " with {n} guideline(s) waiting for your decision",
        "cognition_notice_only_waiting": "{n} guideline(s) from the cognitive engine wait for your decision.",
        "cognition_notice_go": "Read it and decide",
        "cognition_new": "New", "cognition_by": "by",
        "cognition_summary": "What the supervisor tells you", "cognition_reflection": "Its reflection",
        "cognition_reflection_link": "see it in Reflections", "cognition_advice": "Advice it gives itself",
        "cognition_questions": "Open questions", "cognition_coverage": "Coverage",
        "cognition_coverage_line": "It read {read} new memories (INFINITO reported {reported}); cited by no one: "
                                   "{nobody}.",
        "cognition_complete": "The supervisor had the whole day.", "cognition_limits": "Limits",
        "cognition_audit": "Audit of its children", "cognition_hypotheses": "Hypotheses and verdicts",
        "cognition_refuted_by": "Would be refuted by", "cognition_rests_on": "Rests on",
        "cognition_verdicts": {"supported": "supported", "refuted": "refuted", "untested": "untested"},
        "cognition_hypothesis_note": "A supported hypothesis is still a hypothesis.",
        "cognition_costs": "Cost", "cognition_costs_line": "€{spent} of a €{cap} cap that day.",
        "cognition_proposals": "Guidelines it proposes", "cognition_for": "for the",
        "cognition_roles": {"observer": "observer", "investigator": "investigator", "supervisor": "supervisor"},
        "cognition_why": "Why", "cognition_test": "How it would be checked",
        "cognition_status": {"pending": "waiting for your decision", "approved": "accepted", "rejected": "rejected",
                             "revoked": "withdrawn"},
        "cognition_note_label": "Note for the engine (optional)",
        "cognition_approve": "Accept", "cognition_reject": "Reject", "cognition_revoke": "Withdraw",
        "cognition_done": {"approved": "Guideline accepted: the engine applies it from its next call.",
                           "rejected": "Guideline rejected: it will not apply.",
                           "revoked": "Guideline withdrawn: the engine stops applying it."},
        "cognition_gone": "That guideline is gone or already decided.",
        "cognition_refused": "The note looked like it held a secret and was not kept. Do not repeat it.",
    },
}

T = TypeVar("T")
_locks: Dict[str, threading.Lock] = {}
_guard = threading.Lock()


class Refused(ValueError):
    """What a call asked cannot be done; its message is what the agent is told."""


def _lock_of(folder: Path) -> threading.Lock:
    with _guard:
        return _locks.setdefault(str(Path(folder).resolve()), threading.Lock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _empty() -> dict:
    return {"reviews": []}


def _read(folder: Path, key: bytes, account: str) -> dict:
    from .remote_accounts import _unseal

    path = Path(folder) / FILE
    if not path.exists():
        return _empty()
    blob = path.read_bytes()
    plain = _unseal(key, blob[len(MAGIC):], f"cognition:{account}") if blob.startswith(MAGIC) else None
    if plain is None:
        raise ValueError(f"{path} is not this memory's cognitive engine.")
    return {**_empty(), **json.loads(plain.decode("utf-8"))}


def _write(folder: Path, key: bytes, account: str, data: dict) -> None:
    from .remote_accounts import _seal

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    data["reviews"] = data["reviews"][-MAX_REVIEWS:]
    temporary = folder / (FILE + ".tmp")
    temporary.write_bytes(MAGIC + _seal(key, json.dumps(data, ensure_ascii=False).encode("utf-8"), f"cognition:{account}"))
    os.chmod(temporary, 0o600)
    temporary.replace(folder / FILE)


def read(folder: Path, key: bytes, account: str) -> dict:
    with _lock_of(folder):
        return _read(folder, key, account)


def change(folder: Path, key: bytes, account: str, action: Callable[[dict], T]) -> T:
    """Runs action on the reviews and keeps what it changed, one change at a time; if it raises, nothing is written."""
    with _lock_of(folder):
        data = _read(folder, key, account)
        result = action(data)
        _write(folder, key, account, data)
        return result


def _clean(value, limit: int, what: str) -> str:
    from .safety import SensitiveInformationFilter
    from .types import SafetyLevel

    text = str(value or "").strip()[:limit]
    if text and SensitiveInformationFilter().inspect(text).level == SafetyLevel.FORBIDDEN:
        raise Refused(f"Not kept: the {what} looks like it holds a secret, and secrets never go in a review. Do not "
                      "repeat it.")
    return text


def _ids(value) -> List[str]:
    return [str(v).strip()[:64] for v in (value if isinstance(value, list) else []) if str(v).strip()][:MAX_IDS]


def _number(value) -> float:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return 0.0


# What the engine does

def report(data: dict, app: Optional[str], review: dict) -> dict:
    """Keep one review as the engine sent it, cleaned and bounded. A run already kept is not kept twice."""
    if not isinstance(review, dict):
        raise Refused("Not kept: review is the engine's JSON report.")
    run = _clean(review.get("run"), 40, "run")
    if not run:
        raise Refused("Not kept: the review needs its run.")
    if any(r["run"] == run for r in data["reviews"]):
        raise Refused(f"Already kept: the review of run {run} is on the user's page.")
    taken = {p["id"] for r in data["reviews"] for p in r["proposals"]}
    reflection = review.get("reflection") if isinstance(review.get("reflection"), dict) else {}
    coverage = review.get("coverage") if isinstance(review.get("coverage"), dict) else {}
    costs = review.get("costs") if isinstance(review.get("costs"), dict) else {}
    kept = {
        "id": secrets.token_hex(4), "at": _now(), "run": run, "by": app or "an app",
        "summary": _clean(review.get("summary"), MAX_TEXT, "summary"),
        "reflection": {"id": _clean(reflection.get("id"), 16, "reflection id"),
                       "text": _clean(reflection.get("text"), MAX_TEXT, "reflection"),
                       "advice": _clean(reflection.get("advice"), MAX_SHORT, "advice"),
                       "questions": [_clean(q, 300, "question") for q in (reflection.get("questions") or [])[:5]],
                       "sources": _ids(reflection.get("sources")), "model": _clean(reflection.get("model"), 60, "model")},
        "coverage": {"read": int(_number(coverage.get("read"))), "reported": int(_number(coverage.get("reported"))),
                     "nobody": int(_number(coverage.get("nobody"))),
                     "limits": [_clean(x, 300, "limit") for x in (coverage.get("limits") or [])[:10]]},
        "audit": [{"role": _clean(a.get("role"), 20, "role"), "issue": _clean(a.get("issue"), 20, "issue"),
                   "note": _clean(a.get("note"), MAX_SHORT * 2, "note"), "ids": _ids(a.get("ids"))}
                  for a in (review.get("audit") or [])[:MAX_ITEMS] if isinstance(a, dict)],
        "hypotheses": [{"id": _clean(h.get("id"), 12, "id"), "statement": _clean(h.get("statement"), MAX_SHORT, "statement"),
                        "falsifier": _clean(h.get("falsifier"), MAX_SHORT, "falsifier"),
                        "verdict": h.get("verdict") if h.get("verdict") in VERDICTS else "untested",
                        "why": _clean(h.get("why"), MAX_SHORT * 2, "verdict"), "ids": _ids(h.get("ids"))}
                       for h in (review.get("hypotheses") or [])[:MAX_ITEMS] if isinstance(h, dict)],
        "costs": {"spent_eur": _number(costs.get("spent_eur")), "cap_eur": _number(costs.get("cap_eur"))},
        "proposals": [],
        "seen": False,
    }
    for proposal in (review.get("proposals") or [])[:MAX_ITEMS]:
        if not isinstance(proposal, dict) or proposal.get("target_role") not in ROLES:
            continue
        pid = _clean(proposal.get("id"), 40, "id")
        guideline = _clean(proposal.get("guideline"), MAX_SHORT, "guideline")
        if not pid or not guideline or pid in taken:
            continue
        taken.add(pid)
        kept["proposals"].append({"id": pid, "target_role": proposal["target_role"], "guideline": guideline,
                                  "rationale": _clean(proposal.get("rationale"), MAX_SHORT, "rationale"),
                                  "test": _clean(proposal.get("test"), MAX_SHORT, "test"),
                                  "status": "pending", "decided_at": "", "note": ""})
    data["reviews"].append(kept)
    return kept


def decisions(data: dict) -> str:
    """Every guideline proposed and the person's word on it, for the engine."""
    return json.dumps([{"id": p["id"], "target_role": p["target_role"], "status": p["status"],
                        "decided_at": p["decided_at"], "note": p["note"]}
                       for r in data["reviews"] for p in r["proposals"]], ensure_ascii=False)


def drop_citing(data: dict, forgotten: Sequence[str]) -> int:
    """Forgetting reaches the reviews: every part citing a forgotten memory goes, and its id from every list."""
    gone, changed = set(forgotten), 0
    for review in data["reviews"]:
        for part in ("audit", "hypotheses"):
            kept = [item for item in review[part] if not gone & set(item["ids"])]
            changed += len(review[part]) - len(kept)
            review[part] = kept
        if gone & set(review["reflection"]["sources"]):
            review["reflection"].update(text="", advice="", questions=[], sources=[])
            changed += 1
    return changed


# What the person does, on their page

def decide(data: dict, proposal_id: str, action: str, note: str = "") -> str:
    status = DECISIONS.get(action)
    proposal = next((p for r in data["reviews"] for p in r["proposals"] if p["id"] == proposal_id), None)
    allowed = {"approved": ("pending",), "rejected": ("pending",), "revoked": ("approved",)}
    if status is None or proposal is None or proposal["status"] not in allowed[status]:
        raise Refused("gone")
    proposal.update(status=status, decided_at=_now(), note=_clean(note, MAX_SHORT, "note"))
    return status


def mark_seen(data: dict) -> None:
    for review in data["reviews"]:
        review["seen"] = True


def unseen(data: dict) -> int:
    return sum(not review["seen"] for review in data["reviews"])


def waiting(data: dict) -> int:
    return sum(p["status"] == "pending" for r in data["reviews"] for p in r["proposals"])


def notice(t: dict, data: dict) -> str:
    """The line every page shows while a review is unread or a guideline waits."""
    new, pending = [r for r in data["reviews"] if not r["seen"]], waiting(data)
    if not new and not pending:
        return ""
    if new:
        when = f" ({_date(new[-1]['at'])})"
        text = t["cognition_notice"].format(when=when, waiting=t["cognition_notice_waiting"].format(n=pending)
                                            if pending else "")
    else:
        text = t["cognition_notice_only_waiting"].format(n=pending)
    return (f'<p class="ok" role="status" style="margin:0 0 1rem">{html.escape(text)} '
            f'<a href="/memory?view=cognition">{t["cognition_notice_go"]} →</a></p>')


def _date(at: str) -> str:
    try:
        return datetime.fromisoformat(at).strftime("%d/%m/%Y %H:%M")
    except ValueError:
        return at[:16]


def body(t: dict, data: dict, csrf: str, hidden_fields, *, message: str = "", error: str = "") -> str:
    e = html.escape
    parts = [f'<p class="ok" role="status">{e(message)}</p>' if message else "",
             f'<p class="error" role="alert">{e(error)}</p>' if error else "",
             f'<p class="note">{t["cognition_note"]}</p>']
    if not data["reviews"]:
        return "".join(parts) + f'<p>{t["cognition_empty"]}</p>'
    for review in reversed(data["reviews"]):
        head = [_date(review["at"]), f'{t["cognition_by"]} {e(review["by"])}']
        if not review["seen"]:
            head.insert(0, f'<b>{t["cognition_new"]}</b>')
        card = [f'<p class="meta">{" · ".join(head)}</p>']
        if review["summary"]:
            card.append(f'<h3>{t["cognition_summary"]}</h3><p style="white-space:pre-wrap">{e(review["summary"])}</p>')
        reflection = review["reflection"]
        if reflection["text"]:
            link = (f' · <a href="/memory?view=reflections#r-{e(reflection["id"])}">{t["cognition_reflection_link"]}</a>'
                    if reflection["id"] else "")
            model = f' ({e(reflection["model"])})' if reflection["model"] else ""
            card.append(f'<h3>{t["cognition_reflection"]}{model}</h3>'
                        f'<p style="white-space:pre-wrap">{e(reflection["text"])}</p>')
            if reflection["advice"]:
                card.append(f'<p class="meta"><b>{t["cognition_advice"]}:</b> {e(reflection["advice"])}</p>')
            if reflection["questions"]:
                card.append(f'<p class="meta"><b>{t["cognition_questions"]}:</b></p><ul class="meta">'
                            + "".join(f"<li>{e(q)}</li>" for q in reflection["questions"]) + "</ul>")
            card.append(f'<p class="meta">{t["cognition_rests_on"]}: {len(reflection["sources"])}{link}</p>')
        coverage = review["coverage"]
        card.append(f'<h3>{t["cognition_coverage"]}</h3><p class="meta">'
                    + e(t["cognition_coverage_line"].format(read=coverage["read"], reported=coverage["reported"],
                                                            nobody=coverage["nobody"]))
                    + " " + (e(t["cognition_complete"]) if not coverage["limits"] else
                             f'<b>{t["cognition_limits"]}:</b> ' + "; ".join(e(x) for x in coverage["limits"]))
                    + "</p>")
        if review["audit"]:
            card.append(f'<h3>{t["cognition_audit"]}</h3><ul class="meta">' + "".join(
                f'<li><b>{e(t["cognition_roles"].get(a["role"], a["role"]))}</b> · {e(a["issue"])}: {e(a["note"])}</li>'
                for a in review["audit"]) + "</ul>")
        if review["hypotheses"]:
            items = "".join(
                f'<li><b>{e(h["id"])}</b> {e(h["statement"])}<br><span class="meta">{t["cognition_refuted_by"]}: '
                f'{e(h["falsifier"]) or "—"} · <b>{t["cognition_verdicts"][h["verdict"]]}</b>'
                f'{": " + e(h["why"]) if h["why"] else ""}</span></li>' for h in review["hypotheses"])
            card.append(f'<h3>{t["cognition_hypotheses"]}</h3><ul>{items}</ul>'
                        f'<p class="meta">{t["cognition_hypothesis_note"]}</p>')
        card.append(f'<h3>{t["cognition_costs"]}</h3><p class="meta">'
                    + e(t["cognition_costs_line"].format(spent=f'{review["costs"]["spent_eur"]:.3f}',
                                                         cap=f'{review["costs"]["cap_eur"]:.2f}')) + "</p>")
        if review["proposals"]:
            card.append(f'<h3>{t["cognition_proposals"]}</h3>')
            for p in review["proposals"]:
                pid = e(p["id"])
                state = t["cognition_status"][p["status"]]
                card.append(
                    f'<div class="memo"><div><p><b>{pid}</b> {t["cognition_for"]} '
                    f'<b>{e(t["cognition_roles"][p["target_role"]])}</b> · <i>{state}</i></p>'
                    f'<p style="white-space:pre-wrap">{e(p["guideline"])}</p>'
                    + (f'<p class="meta"><b>{t["cognition_why"]}:</b> {e(p["rationale"])}</p>' if p["rationale"] else "")
                    + (f'<p class="meta"><b>{t["cognition_test"]}:</b> {e(p["test"])}</p>' if p["test"] else "")
                    + (f'<p class="meta">{e(p["note"])}</p>' if p["note"] else ""))
                if p["status"] in ("pending", "approved"):
                    buttons = (f'<button class="quiet" type="submit" name="action" value="cognition_approve">'
                               f'{t["cognition_approve"]}</button><button class="bad-quiet" type="submit" name="action" '
                               f'value="cognition_reject">{t["cognition_reject"]}</button>'
                               if p["status"] == "pending" else
                               f'<button class="bad-quiet" type="submit" name="action" value="cognition_revoke">'
                               f'{t["cognition_revoke"]}</button>')
                    card.append(f'<form method="post" action="/memory">{hidden_fields(csrf=csrf, id=p["id"])}'
                                f'<label for="n-{pid}" class="meta">{t["cognition_note_label"]}</label>'
                                f'<textarea id="n-{pid}" name="text" style="min-height:2.5rem"></textarea>'
                                f'<div class="actions">{buttons}</div></form>')
                card.append("</div></div>")
        parts.append(f'<article class="memo" id="c-{e(review["id"])}"><div>{"".join(card)}</div></article>')
    return "".join(parts)
