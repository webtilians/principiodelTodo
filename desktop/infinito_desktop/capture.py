"""The quick capture: what the person types or dictates becomes a note, a Device API event (origin declared).

The text goes as the person wrote it, but first through the same secret filter as the memory
(src/infinito3/safety.py), here on the PC: what it would refuse is never queued nor sent, and the window says why.
"""
import uuid
from datetime import datetime
from typing import Optional, Tuple

MAX_TEXT = 2000  # as remember (src/infinito3/mcp_server.py, MAX_STATEMENT)
MAX_PROJECT = 80
# The kinds remember takes, as the window shows them.
KINDS = {"": None, "Dato": "fact", "Decisión": "decision", "Pendiente": "pending", "Propuesta": "proposal",
         "Acuerdo": "agreement", "Suposición": "assumption", "Por verificar": "to_verify"}
# What the filter found, in words for the window; the found text itself is never shown back.
SECRETS = {"password": "una contraseña", "pin": "un PIN", "api_key": "una clave de API", "private_key": "una clave privada",
           "national_id": "un número de documento de identidad", "bank_account": "un número de cuenta",
           "card_number": "un número de tarjeta"}


def refusal(text: str) -> Optional[str]:
    """Why this text cannot be kept, in a sentence for the person, or None if it can."""
    from infinito3.safety import SafetyLevel, SensitiveInformationFilter

    text = (text or "").strip()
    if not text:
        return "No hay nada que guardar."
    if len(text) > MAX_TEXT:
        return f"Es demasiado largo: una nota tiene como mucho {MAX_TEXT} caracteres."
    found = SensitiveInformationFilter().inspect(text)
    if found.level == SafetyLevel.FORBIDDEN:
        what = SECRETS.get(found.reason, "un secreto")
        return (f"No se guarda: parece {what}, y los secretos nunca van a la memoria. Guárdalo en tu bóveda o en tu "
                "gestor de contraseñas.")
    return None


def note(text: str, kind: Optional[str], project: Optional[str], device_id: str,
         now: Optional[datetime] = None) -> Tuple[Optional[dict], Optional[str]]:
    """(the event, None) to queue and send, or (None, why not)."""
    why = refusal(text)
    if why:
        return None, why
    project = (project or "").strip()[:MAX_PROJECT] or None
    moment = (now or datetime.now().astimezone()).replace(microsecond=0)
    payload = {"text": text.strip(), **({"kind": kind} if kind else {}), **({"project": project} if project else {})}
    return {"v": 1, "event_id": str(uuid.uuid4()), "device_id": device_id, "capability": "note", "type": "text",
            "origin": "declared", "observed_from": moment.isoformat(), "observed_to": moment.isoformat(),
            "payload": payload}, None
