"""python -m cognition <command>

    login                 sign the engine in to INFINITO (opens the sign-in page; you type your password there)
    models ROLE PROVIDER MODEL --price-in USD --price-out USD [--base-url URL --key-name ENV]
                          choose a role's model and its price per million tokens (rounded up)
    authorize             let it spend, within the daily budget; until then every review is a dry run
    deauthorize           back to dry runs
    tick                  what the scheduled task runs every hour: a review if one is due, else an observer check
    review                a review now, whatever the hour
    status                where it stands: last review, today's spending, proposals waiting, paused or not
    proposals             the guidelines the supervisor proposed, with their status
    approve ID | reject ID | revoke ID     your word on a proposal
    rate LABEL util|no    your rating of a blind hypothesis from a report
    mission add TEXT | mission list | mission remove N
    pause | resume        stop or restart it; nothing else changes
    purge MEMORY_ID...    delete what the diary and reports say about memories you forgot
    verify                check the diary's chain
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from . import config as settings
from .cycle import Busy, Engine
from .diary import Diary, check, proposals
from .infinito import Infinito, InfinitoError
from .ledger import Ledger
from .providers import make

ROLES = ("observer", "investigator", "supervisor", "baseline")


def _engine(folder: Path, conf: dict) -> Engine:
    live = bool(conf.get("live_authorized"))
    return Engine(folder, conf, Infinito(folder / "session.json"), lambda role, chosen: make(chosen), live=live)


def _say(text: str) -> None:
    if sys.stdout is None:  # pythonw, as the scheduled task runs it: no console
        return
    sys.stdout.buffer.write((text + "\n").encode("utf-8"))
    sys.stdout.flush()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m cognition", description="INFINITO's cognitive engine.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("login")
    commands.add_parser("logout")
    chosen = commands.add_parser("models")
    chosen.add_argument("role", choices=ROLES)
    chosen.add_argument("provider", choices=("anthropic", "openai"))
    chosen.add_argument("model")
    chosen.add_argument("--price-in", type=float, required=True)
    chosen.add_argument("--price-out", type=float, required=True)
    chosen.add_argument("--base-url")
    chosen.add_argument("--key-name")
    chosen.add_argument("--effort", choices=("none", "low", "medium", "high", "xhigh", "max"),
                        help="reasoning effort, for the models that take it (OpenAI's)")
    chosen.add_argument("--max-output", type=int, help="output tokens allowed per call, reasoning included")
    commands.add_parser("authorize")
    commands.add_parser("deauthorize")
    commands.add_parser("tick")
    commands.add_parser("review")
    commands.add_parser("status")
    commands.add_parser("proposals")
    for word in ("approve", "reject", "revoke"):
        decide = commands.add_parser(word)
        decide.add_argument("id")
        decide.add_argument("--why", default="")
    rate = commands.add_parser("rate")
    rate.add_argument("label")
    rate.add_argument("value", choices=("util", "no"))
    mission = commands.add_parser("mission")
    mission.add_argument("action", choices=("add", "list", "remove"))
    mission.add_argument("value", nargs="?")
    commands.add_parser("pause")
    commands.add_parser("resume")
    purge = commands.add_parser("purge")
    purge.add_argument("ids", nargs="+")
    commands.add_parser("verify")
    args = parser.parse_args(argv)

    folder = settings.home()
    folder.mkdir(parents=True, exist_ok=True)
    conf = settings.load(folder)
    diary = Diary(folder)

    if args.command == "login":
        _say(Infinito(folder / "session.json").login(conf["server"], conf["app_name"]))
        _say("En tu página de memoria, en Apps, dale el nivel «Leer y añadir»: es todo lo que necesita.")
    elif args.command == "logout":
        _say("Sesión borrada." if Infinito(folder / "session.json").logout() else "No había sesión.")
    elif args.command == "models":
        roles = conf["roles"]
        roles[args.role] = {**roles[args.role], "provider": args.provider, "model": args.model,
                            **({"base_url": args.base_url} if args.base_url else {}),
                            **({"key_name": args.key_name} if args.key_name else {}),
                            **({"reasoning_effort": args.effort} if args.effort else {}),
                            **({"max_output": args.max_output} if args.max_output else {})}
        prices = {**conf["prices_usd_per_million"], args.model: [args.price_in, args.price_out]}
        settings.save_override(folder, "roles", roles)
        settings.save_override(folder, "prices_usd_per_million", prices)
        diary.append("model_choice", role=args.role, provider=args.provider, model=args.model,
                     price_usd_per_million=[args.price_in, args.price_out])
        chosen_now = roles[args.role]
        _say(f"{args.role}: {args.provider} {args.model} a {args.price_in}/{args.price_out} $ por millón, "
             f"esfuerzo {chosen_now.get('reasoning_effort') or 'el del modelo'}, "
             f"hasta {chosen_now['max_output']} tokens de salida.")
    elif args.command == "authorize":
        missing = [r for r in ROLES if not conf["roles"][r].get("model")
                   or conf["roles"][r]["model"] not in conf["prices_usd_per_million"]]
        if missing:
            _say("Falta elegir modelo y precio para: " + ", ".join(missing) + " (python -m cognition models ...).")
            return 2
        settings.save_override(folder, "live_authorized", True)
        diary.append("spend_authorization", granted=True, budget_eur=conf["budget_eur"],
                     budget_eur_after=conf["budget_eur_after"])
        _say("Autorizado: gastará como mucho el tope de cada día "
             f"({', '.join(str(b) for b in conf['budget_eur'])} € los siete primeros, luego {conf['budget_eur_after']} €).")
    elif args.command == "deauthorize":
        settings.save_override(folder, "live_authorized", False)
        diary.append("spend_authorization", granted=False)
        _say("Desautorizado: desde ahora, solo pruebas sin gasto.")
    elif args.command in ("tick", "review"):
        engine = _engine(folder, conf)
        if args.command == "tick":
            try:
                said = engine.tick()
            except InfinitoError as error:
                said = f"INFINITO: {error}"
            line = f"{datetime.now():%Y-%m-%d %H:%M} {said}"
            with (folder / "tick.log").open("a", encoding="utf-8") as log:
                log.write(line + "\n")
            _say(line)
        else:
            try:
                with engine.lock():
                    _say(engine.review())
            except Busy:
                _say("Ya hay una ejecución en marcha.")
                return 1
    elif args.command == "status":
        ledger, today = Ledger(folder, conf), datetime.now().date()
        last = diary.last("review_end")
        waiting = [p for p in proposals(diary) if p["status"] == "pending"]
        signed = Infinito(folder / "session.json").session()
        _say("\n".join([
            f"Conectado a INFINITO: {'sí, ' + signed['server'] if signed else 'no (python -m cognition login)'}",
            f"Modo: {'real, autorizado a gastar' if conf.get('live_authorized') else 'prueba, sin gasto'}"
            + (" · EN PAUSA" if (folder / 'PAUSED').exists() else ""),
            f"Última revisión: {last['at'] + ' · ' + last['outcome'] if last else 'ninguna'}",
            f"Hoy: {ledger.spent(today):.3f} € de {ledger.cap(today):.2f} €",
            f"Pautas esperando tu aprobación: {len(waiting)}",
            "Modelos: " + "; ".join(f"{r} {conf['roles'][r].get('model') or '(sin elegir)'}" for r in ROLES),
            f"Diario: {'íntegro' if not check(diary) else '; '.join(check(diary))}",
            f"Informes: {folder / 'reports'}"]))
    elif args.command == "proposals":
        found = proposals(diary)
        _say("\n".join(f"{p['id']} [{p['status']}] {p['target_role']}: {p['guideline']}" for p in found)
             or "El supervisor aún no ha propuesto ninguna pauta.")
    elif args.command in ("approve", "reject", "revoke"):
        found = {p["id"]: p for p in proposals(diary)}
        if args.id not in found:
            _say(f"No hay ninguna propuesta {args.id}.")
            return 2
        decision = {"approve": "approved", "reject": "rejected", "revoke": "revoked"}[args.command]
        if decision == "revoked" and found[args.id]["status"] != "approved":
            _say("Solo se retira una pauta aprobada.")
            return 2
        diary.append("policy_decision", proposal=args.id, decision=decision, actor="owner",
                     body={"why": args.why})
        _say(f"{args.id}: {decision}. " + ("Se aplicará al rol desde la próxima llamada." if decision == "approved"
                                           else "No se aplica."))
    elif args.command == "rate":
        if not diary.last("blind_item", label=args.label):
            _say(f"No hay ninguna hipótesis {args.label}.")
            return 2
        diary.append("rating", label=args.label, value=args.value, actor="owner")
        _say(f"{args.label}: {args.value}.")
    elif args.command == "mission":
        missions = list(conf.get("missions") or [])
        if args.action == "add" and args.value:
            missions.append(args.value)
        elif args.action == "remove" and args.value:
            missions.pop(int(args.value) - 1)
        if args.action != "list":
            settings.save_override(folder, "missions", missions)
            diary.append("missions", count=len(missions), body={"missions": missions})
        _say("\n".join(f"{n}. {m}" for n, m in enumerate(missions, 1)) or "Sin misiones: exploración general.")
    elif args.command == "pause":
        (folder / "PAUSED").write_text(datetime.now().isoformat(), encoding="utf-8")
        diary.append("paused", actor="owner")
        _say("En pausa. Nada se ejecuta hasta `python -m cognition resume`.")
    elif args.command == "resume":
        (folder / "PAUSED").unlink(missing_ok=True)
        diary.append("resumed", actor="owner")
        _say("En marcha otra vez.")
    elif args.command == "purge":
        count = diary.purge(args.ids)
        removed = 0
        for report in (folder / "reports").glob("*.md"):
            if any(i in report.read_text(encoding="utf-8") for i in args.ids):
                report.unlink()
                removed += 1
        _say(f"Borrados {count} cuerpos del diario y {removed} informes que los citaban.")
    elif args.command == "verify":
        problems = check(diary)
        _say("El diario está íntegro." if not problems else "\n".join(problems))
        return 1 if problems else 0
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except InfinitoError as error:
        _say(f"INFINITO: {error}")
        raise SystemExit(1)
