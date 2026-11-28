"""Whether INFINITO works on this PC: one check per piece, each a pure function of what it reads.

Each says bien (ok), aviso (warn), fallo (fail) or «no lo sé» (unknown), with a sentence for the person and a link to
where it is fixed. What a check cannot read is «no lo sé», never «bien»: the backups task said 0 while it refused every
copy from 7/11/2026, and the engine's notice said «ha terminado la revisión» while its supervisor had no quota.
"""
import csv
import io
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Iterable, List, Optional

OK, WARN, FAIL, UNKNOWN = "ok", "warn", "fail", "unknown"
# How bad each is, for the icon's colour and for telling the person only when something gets worse.
RANK = {OK: 0, UNKNOWN: 1, WARN: 2, FAIL: 3}
WORDS = {OK: "bien", WARN: "aviso", FAIL: "fallo", UNKNOWN: "no lo sé"}

BACKUP_FILE = re.compile(r"^infinito-(\d{8}-\d{6})\.tar\.gz$")
BACKUP_LINE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}) (ok|FAILED): ?(.*)$")
BACKUP_DAYS = 3
REVIEW_DAYS = 2
LOG_TAIL = 64 * 1024
# What Task Scheduler says of a task that is running, or has not run yet: neither is a failure.
TASK_RUNNING, TASK_NOT_YET = 267009, 267011


@dataclass
class Check:
    key: str
    state: str
    title: str
    sentence: str
    link: str = ""

    def as_dict(self) -> dict:
        return {**asdict(self), "word": WORDS[self.state]}


def worst(checks: Iterable[Check]) -> str:
    """The colour of the icon: red with any failure, amber with any warning or doubt, green only when all is well."""
    states = [check.state for check in checks]
    if not states:
        return UNKNOWN
    top = max(states, key=RANK.get)
    return WARN if top == UNKNOWN else top


def _day(moment: datetime) -> str:
    return f"{moment:%d/%m/%Y}"


def _uri(path: Path) -> str:
    return Path(path).resolve().as_uri() if Path(path).exists() else ""


# Backups: the task «INFINITO copias de seguridad» (remote/deploy/windows/infinito-backup-pull.ps1).

def newest_backup(folder: Path) -> Optional[datetime]:
    """When the newest copy on this PC was made (its name says it, in UTC), or None if there is none."""
    made = []
    for path in Path(folder).glob("infinito-*.tar.gz"):
        found = BACKUP_FILE.match(path.name)
        if found:
            made.append(datetime.strptime(found.group(1), "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc))
    return max(made) if made else None


def backups(folder: Path, now: datetime) -> Check:
    folder = Path(folder)
    log, title = folder / "pull.log", "Copias de seguridad"
    try:
        with log.open("rb") as handle:
            handle.seek(max(0, log.stat().st_size - LOG_TAIL))
            lines = handle.read().decode("utf-8-sig", errors="replace").splitlines()
    except OSError:
        return Check("backups", UNKNOWN, title, f"No encuentro el registro de las copias ({log}).", _uri(folder))
    runs = [found for found in map(BACKUP_LINE.match, (line.strip() for line in lines)) if found]
    newest = newest_backup(folder)
    good = f"La última copia buena es del {_day(newest)}" if newest else "No hay ninguna copia buena en este PC"
    if not runs:
        return Check("backups", UNKNOWN, title, f"El registro de las copias no dice nada que entienda. {good}.",
                     _uri(folder))
    when, outcome, detail = runs[-1].groups()
    if outcome == "FAILED":
        reason = detail.strip().rstrip(".")[:160] or "sin motivo"
        return Check("backups", FAIL, title, f"La última copia ({when}) falló: {reason}. {good}.", _uri(folder))
    age = (now.astimezone(timezone.utc) - newest).days if newest else None
    if age is None or age > BACKUP_DAYS:
        return Check("backups", FAIL, title, f"{good}, hace {age} días: no llegan copias nuevas." if newest
                     else f"{good}.", _uri(folder))
    return Check("backups", OK, title, f"{good}.", _uri(folder))


# The cognitive engine (cognition/), from its own summary (cognition/status.py).

def _when(text: Optional[str]) -> Optional[datetime]:
    try:
        moment = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.astimezone()


def engine(summary: Optional[dict], now: datetime, page: str) -> Check:
    title = "Motor cognitivo"
    if summary is None:
        return Check("engine", UNKNOWN, title, "El motor cognitivo no tiene carpeta en este PC.")
    if summary.get("diary_problems"):
        return Check("engine", FAIL, title, "Su diario no está íntegro: " + "; ".join(summary["diary_problems"])[:200]
                     + ". Comprueba con python -m cognition verify.", page)
    if summary.get("paused"):
        since = _when(summary.get("paused_since"))
        return Check("engine", WARN, title, f"Está en pausa desde el {_day(since)} a las {since:%H:%M}." if since
                     else "Está en pausa.", page)
    if not summary.get("signed_in"):
        return Check("engine", WARN, title, "No tiene sesión en INFINITO: python -m cognition login.", page)
    review = summary.get("last_review")
    if not review:
        return Check("engine", UNKNOWN, title, "Aún no ha hecho ninguna revisión.", page)
    at = _when(review.get("at"))
    if not review.get("judged"):
        problem = review.get("problem") or "no dio una respuesta utilizable"
        day = f" del {_day(at)}" if at else ""
        return Check("engine", WARN, title, f"La última revisión{day} quedó sin juzgar: {problem}.", page)
    if at is None or now - at > timedelta(days=REVIEW_DAYS):
        return Check("engine", WARN, title, f"La última revisión es del {_day(at)}: hace más de {REVIEW_DAYS} días."
                     if at else "No sé cuándo fue la última revisión.", page)
    spent = f" Hoy lleva {summary.get('spent_today_eur', 0):.2f} € de {summary.get('cap_today_eur', 0):.2f} €."
    waiting = summary.get("proposals_waiting") or 0
    asks = f" {waiting} pauta{'s' if waiting != 1 else ''} espera{'n' if waiting != 1 else ''} tu decisión." if waiting else ""
    return Check("engine", OK, title, f"Última revisión: {_day(at)} a las {at:%H:%M}.{spent}{asks}", page)


# Scheduled tasks: schtasks /query /fo csv /v, read by position, because its headings come in the PC's language.

def scheduled_tasks(text: str) -> List[dict]:
    """The INFINITO tasks in schtasks' CSV: name, next run, status, last run, last result and whether it is on."""
    found, seen = [], set()
    for row in csv.reader(io.StringIO(text)):
        if len(row) < 12 or row[0] == "" or "\\INFINITO" not in row[1] or row[1] in seen:
            continue
        seen.add(row[1])
        try:
            result: Optional[int] = int(row[6])
        except ValueError:
            result = None
        state = row[11].strip().lower()
        found.append({"name": row[1].lstrip("\\"), "next": row[2], "status": row[3], "last": row[5],
                      "result": result, "enabled": not state.startswith(("deshabilit", "disabled", "désactiv"))})
    return found


def tasks(rows: Optional[List[dict]]) -> Check:
    title = "Tareas programadas"
    if rows is None:
        return Check("tasks", UNKNOWN, title, "No he podido leer las tareas programadas de Windows.")
    if not rows:
        return Check("tasks", UNKNOWN, title, "No hay tareas «INFINITO» programadas en este PC.")
    problems = []
    for row in rows:
        if not row["enabled"]:
            problems.append(f"«{row['name']}» está deshabilitada")
        elif row["result"] is None:
            problems.append(f"de «{row['name']}» no entiendo el último resultado")
        elif row["result"] not in (0, TASK_RUNNING, TASK_NOT_YET):
            problems.append(f"«{row['name']}» terminó con el código {row['result'] & 0xFFFFFFFF:#x} ({row['last']})")
    if problems:
        said = "; ".join(problems)
        return Check("tasks", WARN, title, said[:1].upper() + said[1:] + ".")
    return Check("tasks", OK, title, f"{len(rows)} tarea{'s' if len(rows) != 1 else ''}, todas bien.")


# Claude Code's hooks (infinito-remote context and checkpoint) and their session.

def _commands(settings: dict) -> List[str]:
    lines = []
    for groups in (settings.get("hooks") or {}).values():
        for group in groups if isinstance(groups, list) else []:
            for hook in (group.get("hooks") or []) if isinstance(group, dict) else []:
                if isinstance(hook, dict):
                    args = hook.get("args") or []
                    lines.append(" ".join([str(hook.get("command") or "")] + [str(a) for a in args]))
    return lines


def hooks(claude_settings: Path, session: Path) -> Check:
    title = "Hooks de Claude Code"
    try:
        settings = json.loads(Path(claude_settings).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Check("hooks", WARN, title, "Claude Code no tiene ajustes en este PC, así que tampoco los hooks de "
                                           "INFINITO.")
    except (OSError, ValueError):
        return Check("hooks", UNKNOWN, title, f"No he podido leer {claude_settings}.")
    commands = _commands(settings) if isinstance(settings, dict) else []
    missing = [word for word in ("context", "checkpoint")
               if not any(re.search(rf"infinito-remote(\.exe)?\b.*\b{word}\b", line) for line in commands)]
    if missing:
        return Check("hooks", WARN, title, "Falta el hook «infinito-remote " + "» y «infinito-remote ".join(missing)
                     + "» en ~/.claude/settings.json.")
    if not Path(session).exists():
        return Check("hooks", WARN, title, "Los hooks están, pero sin sesión: infinito-remote login "
                                           "https://infinitomemory.com.")
    return Check("hooks", OK, title, "Los hooks están y tienen sesión.")


# The server, and this app's own session.

def server(answer: Callable[[], Optional[int]], address: str) -> Check:
    title = "Servidor"
    try:
        status = answer()
    except Exception:  # noqa: BLE001  # whatever happened on the way, it did not answer
        status = None
    if status == 200:
        return Check("server", OK, title, f"{address} responde.", address)
    return Check("server", FAIL, title, f"{address} no responde" + (f" (HTTP {status})." if status else "."), address)


def session(outcome: str, address: str) -> Check:
    """outcome is what renewing this app's session gave: ok, none (never signed in), expired or offline."""
    title = "Sesión de la app"
    if outcome == "ok":
        return Check("session", OK, title, f"La app tiene sesión en {address}.", f"{address}/memory?view=apps")
    if outcome == "none":
        return Check("session", WARN, title, "La app no ha entrado en tu cuenta: menú «Entrar en mi cuenta».")
    if outcome == "expired":
        return Check("session", FAIL, title, "Su sesión caducó o se desconectó desde el portal: vuelve a entrar.",
                     f"{address}/memory?view=apps")
    return Check("session", UNKNOWN, title, "No he podido renovar la sesión: ¿sin conexión?")
