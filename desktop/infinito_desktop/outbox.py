"""Sending what waits: each note goes once, in order, and what happens to it depends only on the server's answer.

- 200: kept in the memory, or already there («duplicate»): it leaves the queue.
- 400, 403 or 413: the server will never take it (a refused secret, a switched-off capability...): it leaves the queue,
  and the panel says why.
- 401: the session is over; 429: too fast; no network: it stays, and is tried again later.
"""
from typing import Callable, Dict, List, Optional, Tuple

from .waiting import Queue

FINAL = (400, 403, 413)
# Why a note waits when it could not even be sent (session.Store.send says it in its OSError).
WAITS = {"none": "La app aún no ha entrado en tu cuenta: la nota espera aquí, cifrada, hasta que entres.",
         "old": "La app tiene que volver a entrar para poder guardar notas: la nota espera aquí, cifrada, hasta "
                "entonces."}
OFFLINE = "Sin conexión con tu memoria: se enviará sola en cuanto vuelva."


def outcome(status: int, answer: dict) -> Tuple[str, str]:
    """(kept | refused | later, a sentence for the person)."""
    if status == 200 and answer.get("saved"):
        return "kept", ("Ya estaba en tu memoria." if answer.get("duplicate") else "Guardado en tu memoria.")
    if status in FINAL:
        detail = str(answer.get("detail") or answer.get("error") or f"HTTP {status}")
        if detail.startswith("Not stored:"):
            detail = "tu memoria no lo guarda (" + detail.removeprefix("Not stored:").strip().split(".")[0] + ")"
        return "refused", f"No se ha guardado: {detail}."
    if status == 401:
        return "later", "La sesión de la app ha caducado: vuelve a entrar y se enviará sola."
    if status == 429:
        return "later", "Demasiadas seguidas: se enviará en un momento."
    return "later", f"El servidor respondió HTTP {status}: se volverá a intentar."


def deliver(queue: Queue, send: Callable[[dict], Tuple[int, dict]], now: Optional[float] = None) -> List[Dict]:
    """Tries every note that is due, oldest first; stops at the first network failure (the rest would fail too).
    Returns what happened to each: event_id, outcome and sentence."""
    happened = []
    for item in queue.due(now):
        event = item["event"]
        try:
            status, answer = send(event)
        except OSError as error:
            queue.later(event["event_id"], now)
            happened.append({"event_id": event["event_id"], "outcome": "later",
                             "said": WAITS.get(str(error), OFFLINE)})
            break
        result, said = outcome(status, answer)
        if result == "later":
            wait = None
            try:
                wait = float(answer.get("retry_after") or 0) or None
            except (TypeError, ValueError):
                pass
            queue.later(event["event_id"], now, wait)
        else:
            queue.done(event["event_id"])
        happened.append({"event_id": event["event_id"], "outcome": result, "said": said,
                         "memory_id": answer.get("memory_id")})
        if result == "later" and status in (401, 429):
            break
    return happened
