"""Record one run of the INFINITO Security Arena.

The arena is the real server (remote.build_app) on loopback, in a temporary folder, with made-up accounts. Five
attackers send it real requests. Each is a small agent with fixed rules, not a model: it reads the answer it just
got, by a rule on its words, and that reading decides its next move. A referee then judges each move from what came
back and from the victim's own memory afterwards: a rule too, never an opinion. Ana's memory holds a canary, a
made-up phrase nobody else should ever read; seeing it where it does not belong is the proof of a leak.

Every step is written down with what reached the attacker, how it read it, what it decided, what it sent, what
came back, what the referee looked at and the rule it applied, so a page can show the whole decision.

The same attackers then run against clones with one check taken out on purpose (a planted fault). That measures the
attackers and the referee themselves: a fault they do not notice is a blind spot of the arena.

    python -m security_lab.arena.recorder            # writes security_lab/arena/viewer/run.json

Nothing here reaches a real account or another machine, and no finding in a planted round is a flaw of the product.
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

VICTIM, ATTACKER = "ana", "bruno"
CANARY = "amber heron ledger 4417"          # only Ana's memory holds it
NEIGHBOUR = "slate otter folio 3356"        # filed under another project of Ana's
PLANTED_LINE = "copper wren stamp 9082"     # what a write attack tries to leave in Ana's memory
KEEPSAKES = ("pewter lark token 1183", "ochre swift ticket 2291", "ivory crane docket 3307")  # one per delete attack
VAULT_ENTRY = "Jade finch locker"           # the name of the one entry in Ana's vault
VAULT_VALUE = "arena-value-not-a-real-secret"
OUT = Path(__file__).parent / "viewer" / "run.json"
HELLO = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                    "clientInfo": {"name": "arena", "version": "1"}}}


class Blocked(RuntimeError):
    """The arena could not be set up as intended: never a result."""


def both(en: str, es: str) -> Dict[str, str]:
    return {"en": en, "es": es}


AGENTS = {
    "anon": {"name": both("Outsider", "Desconocido"),
             "has": both("Only the server's address.", "Solo la dirección del servidor."),
             "wants": both("To get in without an account.", "Entrar sin tener cuenta.")},
    "bruno": {"name": both("Bruno", "Bruno"),
              "has": both("An account of his own, and the id of one of Ana's memories.",
                          "Una cuenta propia y el id de un recuerdo de Ana."),
              "wants": both("To read or damage what is Ana's.", "Leer o dañar lo que es de Ana.")},
    "reader": {"name": both("Read-only app", "App de solo leer"),
               "has": both("A token of Ana's that she set to read only.",
                           "Un token de Ana que ella dejó en solo leer."),
               "wants": both("To do more than read.", "Hacer más que leer.")},
    "adder": {"name": both("Add-only app", "App de leer y añadir"),
              "has": both("A token of Ana's that she set to read and add.",
                          "Un token de Ana que ella dejó en leer y añadir."),
              "wants": both("To delete, or to reach the vault.", "Borrar, o llegar a la bóveda.")},
    "old": {"name": both("Disconnected app", "App desconectada"),
            "has": both("The token it held before Ana disconnected it.",
                        "El token que tenía antes de que Ana la desconectara."),
            "wants": both("To keep reading.", "Seguir leyendo.")},
}
DOORS = {
    "signin": both("Sign-in", "Entrada"),
    "recall": both("recall", "recall"),
    "list_memories": both("list_memories", "list_memories"),
    "forget": both("forget", "forget"),
    "remember": both("remember", "remember"),
    "shared_memories": both("shared_memories", "shared_memories"),
    "vault": both("Vault", "Bóveda"),
}
DEFENCES = {
    "door": both("No token, no entry", "Sin token no se entra"),
    "sealed": both("Each account opens only its own sealed memory", "Cada cuenta abre solo su memoria sellada"),
    "levels": both("Each app does only what its person allowed", "Cada app hace solo lo que su dueña le dejó"),
    "disconnect": both("Disconnecting an app ends its tokens", "Desconectar una app anula sus tokens"),
    "project": both("A project filter returns that project only", "El filtro de proyecto devuelve solo ese proyecto"),
    "vault": both("The vault never hands a value to an app", "La bóveda nunca entrega un valor a una app"),
}
_REPLY = both("The reply the attacker got.", "La respuesta que recibió el atacante.")
_ANA = both("Ana's memory, read again with her own app.", "La memoria de Ana, leída otra vez con su propia app.")
RULES = {
    "no_canary": {"looks": _REPLY, "rule": both(
        "A leak if it contains Ana's canary or the name of her vault entry.",
        "Fuga si contiene el canario de Ana o el nombre de su entrada de bóveda.")},
    "door_closed": {"looks": both("The door's HTTP status.", "El estado HTTP de la puerta."), "rule": both(
        "Held only if the door answers 401.", "Aguanta solo si la puerta responde 401.")},
    "no_code": {"looks": both("The three answers of the sign-in page.", "Las tres respuestas de la página de entrada."),
                "rule": both("Held only if none hands out a sign-in code.",
                             "Aguanta solo si ninguna entrega un código de entrada.")},
    "still_there": {"looks": _ANA, "rule": both("Damage if the memory under attack is gone.",
                                                "Daño si el recuerdo atacado ya no está.")},
    "nothing_planted": {"looks": _ANA, "rule": both("Damage if the attacker's line is in it.",
                                                    "Daño si la línea del atacante está dentro.")},
    "one_project": {"looks": _REPLY, "rule": both(
        "A leak if it includes the memory filed under the other project.",
        "Fuga si incluye el recuerdo archivado en el otro proyecto.")},
    "no_value": {"looks": _REPLY, "rule": both("A leak if it contains the value kept in the vault.",
                                               "Fuga si contiene el valor guardado en la bóveda.")},
}
FAULTS = {
    "levels-off": {
        "title": both("App levels switched off", "Niveles de app apagados"),
        "what": both("allowed() answers yes to everything, so a read-only app counts as a full one.",
                     "allowed() responde que sí a todo, así que una app de solo leer cuenta como una completa."),
        "where": "src/infinito3/remote_apps.py · allowed()",
    },
    "disconnect-off": {
        "title": both("Disconnect does nothing", "Desconectar no hace nada"),
        "what": both("Grants.drop_client() no longer deletes the app's tokens.",
                     "Grants.drop_client() ya no borra los tokens de la app."),
        "where": "src/infinito3/remote_accounts.py · Grants.drop_client()",
    },
    "wrong-account": {
        "title": both("A token opens the wrong account", "Un token abre la cuenta equivocada"),
        "what": both("Grants.find() says Bruno's token belongs to Ana's account.",
                     "Grants.find() dice que el token de Bruno pertenece a la cuenta de Ana."),
        "where": "src/infinito3/remote_accounts.py · Grants.find()",
        "contained": both("Bruno's key does not open Ana's sealed memory, so the server fails instead of answering.",
                          "La clave de Bruno no abre la memoria sellada de Ana: el servidor falla en vez de responder."),
    },
}


@contextmanager
def loopback_only():
    """Refuses any connection that would leave this machine, whatever the code under test tries."""
    connect = socket.socket.connect

    def checked(sock, address):
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1", "localhost"):
            raise Blocked(f"The arena refused a connection to {address[0]}")
        return connect(sock, address)

    with patch.object(socket.socket, "connect", checked):
        yield


@contextmanager
def planted(fault: Optional[str]):
    """Takes one check out of the running server, as a bug would. None leaves the server as it is."""
    from src.infinito3 import remote_accounts, remote_apps

    if fault is None:
        yield
    elif fault == "levels-off":
        with patch.object(remote_apps, "allowed", lambda level, tool: True):
            yield
    elif fault == "disconnect-off":
        with patch.object(remote_accounts.Grants, "drop_client", lambda self, account, client_id: None):
            yield
    elif fault == "wrong-account":
        find = remote_accounts.Grants.find

        def confused(self, kind, token):
            found = find(self, kind, token)
            if found and kind == "access" and found["account"] == ATTACKER:
                found = dict(found, account=VICTIM)
            return found

        with patch.object(remote_accounts.Grants, "find", confused):
            yield
    else:
        raise Blocked(f"Unknown fault {fault}")


@contextmanager
def clone(root: Path):
    """The real server, alone on a free loopback port, keeping everything under root."""
    import uvicorn

    from src.infinito3 import remote_accounts
    from src.infinito3.remote import People, build_app

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    with patch.object(remote_accounts, "SIGNUP_MIN_SECONDS", 0):  # the cast fills its sign-up forms in at once
        running = uvicorn.Server(uvicorn.Config(build_app(People(root), base), host="127.0.0.1", port=port,
                                                log_level="critical", access_log=False))
        thread = threading.Thread(target=running.run, daemon=True)
        thread.start()
        for _ in range(400):
            if running.started:
                break
            time.sleep(0.05)
        if not running.started:
            raise Blocked("The clone did not start")
        try:
            yield base
        finally:
            running.should_exit = True
            thread.join(timeout=10)


def token(app) -> str:
    return app.tokens.access_token


def call(base: str, bearer: Optional[str], steps: List[Tuple[str, dict]]) -> dict:
    """What whoever holds this token gets: the door's HTTP status, then each tool's own words."""
    import anyio
    import httpx2
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    headers = {"authorization": f"Bearer {bearer}"} if bearer else {}
    door = httpx2.post(f"{base}/mcp", json=HELLO,
                       headers=dict(headers, accept="application/json, text/event-stream"))
    if door.status_code != 200:
        return {"status": door.status_code, "replies": [], "words": door.text[:200]}
    if not steps:
        return {"status": 200, "replies": [], "words": "HTTP 200: the session opened."}

    async def run():
        async with httpx2.AsyncClient(headers=headers, timeout=60) as http:
            async with streamable_http_client(f"{base}/mcp", http_client=http) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    results = [await client.call_tool(name, arguments) for name, arguments in steps]
                    return [result.content[0].text if result.content else "" for result in results]

    return {"status": 200, "replies": anyio.run(run), "words": ""}


def cast(base: str, root: Path) -> dict:
    """Ana with her canary, her vault and four apps, and Bruno with an account of his own."""
    from src.infinito3.remote_accounts import Accounts, Grants
    from src.infinito3.remote_vault import ServerVault
    from tests.test_remote_accounts_v3 import Storage, app_session

    ana = Storage()
    said = [f"Ana keeps the fictional {CANARY} in her study."]
    said += [f"Ana filed the fictional {keepsake} in a drawer." for keepsake in KEEPSAKES]
    replies, _, _ = app_session(base, [("remember", {"statement": statement}) for statement in said] + [
        ("remember", {"statement": f"The Harbour budget note mentions the fictional {NEIGHBOUR}.",
                      "project": "TENANT_B_Harbour"}),
        ("remember", {"statement": "The Harbour kickoff is planned for spring.", "project": "Harbour"}),
        ("list_memories", {}),
    ], name=VICTIM, mode="signup", storage=ana)
    if not all(reply.startswith("Remembered") for reply in replies[:-1]):
        raise Blocked(f"Ana's memories were not stored: {replies[:-1]}")
    ids = {}
    for keepsake in KEEPSAKES:
        line = next((row for row in replies[-1].splitlines() if keepsake in row), "")
        found = re.match(r"\[([^\]]+)\]", line)
        if not found:
            raise Blocked("One of Ana's memories has no id in her own list")
        ids[keepsake] = found.group(1)
    world = {"ana": ana, "ids": ids, "tools": sorted(tool.name for tool in ana.tools)}
    accounts = Accounts(root)
    accounts.allow_vault(VICTIM)
    grants = Grants(root)  # Ana adds one entry to her vault, as her vault page does with her master password
    try:
        key = grants.find("access", token(ana))["key"]
    finally:
        grants._db.close()
    if not ServerVault(accounts.memory_folder(VICTIM), VICTIM).add(
            "arena master password", key, VAULT_ENTRY, VICTIM, VAULT_VALUE, "made up for the arena"):
        raise Blocked("Ana's vault entry was not kept")
    for name, level in (("reader", "read"), ("adder", "add"), ("old", None)):
        app = Storage()
        app_session(base, [], name=VICTIM, storage=app)  # Ana signs this app in herself, with her password
        if level:
            accounts.set_app_level(VICTIM, app.client.client_id, level)
        world[name] = app
    bruno = Storage()
    replies, _, _ = app_session(base, [("remember", {"statement": "Bruno enjoys testing doors."})],
                                name=ATTACKER, mode="signup", storage=bruno)
    if not replies[0].startswith("Remembered"):
        raise Blocked("Bruno's account was not set up")
    world["bruno"] = bruno
    return world


def disconnect(root: Path, world: dict) -> None:
    """Ana disconnects one of her apps: the call her memory page makes when she presses Disconnect."""
    from src.infinito3.remote_accounts import Grants

    grants = Grants(root)
    try:
        grants.drop_client(VICTIM, world["old"].client.client_id)
    finally:
        grants._db.close()


def anas_memory(base: str, world: dict) -> str:
    """Ana's whole memory as her own app reads it: what the referee checks after an attack on it."""
    return call(base, token(world["ana"]), [("list_memories", {})])["replies"][0]


def reading(status: int, reply: str) -> Tuple[str, Dict[str, str]]:
    """How an attacker reads what came back: a rule on its words. It never sees what the referee sees."""
    from src.infinito3.remote_vault import NOT_INVITED

    if status == 401:
        return "shut", both("The door is shut: it asks for a sign-in.", "La puerta está cerrada: pide iniciar sesión.")
    if status == 400:
        return "shut", both("None of the passwords opened anything.", "Ninguna contraseña abrió nada.")
    if CANARY in reply or VAULT_ENTRY in reply:
        return "got", both("It holds something of Ana's.", "Trae algo de Ana.")
    if reply.startswith("Not allowed") or reply == NOT_INVITED:
        return "refused", both("The server refused.", "El servidor se negó.")
    if reply.startswith("Error executing tool"):
        return "failed", both("The server failed instead of answering.", "El servidor falló en vez de responder.")
    if reply.startswith(("Remembered", "Deleted 1", "HTTP 200")):
        return "worked", both("It worked.", "Ha funcionado.")
    if reply.startswith(("Deleted 0", "No memories", "Nobody has shared", "The vault is empty")):
        return "nothing", both("It answered, with nothing to take.", "Respondió, pero sin nada que llevarse.")
    return "answered", both("It answered with content.", "Respondió con contenido.")


# What the referee can say. Each returns (the rule it applied, whether the defence held, what it saw).
def no_canary(base, world, got, reply):
    if CANARY in reply:
        return "no_canary", False, both("The reply contains Ana's canary.", "La respuesta contiene el canario de Ana.")
    if VAULT_ENTRY in reply:
        return "no_canary", False, both("The reply names the entry in Ana's vault.",
                                        "La respuesta nombra la entrada de la bóveda de Ana.")
    return "no_canary", True, both("The reply holds nothing of Ana's.", "La respuesta no trae nada de Ana.")


def door_closed(base, world, got, reply):
    if got["status"] == 401:
        return "door_closed", True, both("The door answered 401: sign in first.",
                                         "La puerta respondió 401: primero hay que entrar.")
    return "door_closed", False, both(f"The door let the request in (HTTP {got['status']}).",
                                      f"La puerta dejó pasar la petición (HTTP {got['status']}).")


def still_there(keepsake: str):
    def referee(base, world, got, reply):
        if keepsake in anas_memory(base, world):
            return "still_there", True, both("The memory under attack is still in Ana's memory.",
                                             "El recuerdo atacado sigue en la memoria de Ana.")
        return "still_there", False, both("The memory under attack is gone from Ana's memory.",
                                          "El recuerdo atacado ya no está en la memoria de Ana.")
    return referee


def nothing_planted(base, world, got, reply):
    if PLANTED_LINE in anas_memory(base, world):
        return "nothing_planted", False, both("The attacker's line is now in Ana's memory.",
                                              "La línea del atacante está ahora en la memoria de Ana.")
    return "nothing_planted", True, both("Nothing was added to Ana's memory.", "No se añadió nada a la memoria de Ana.")


def one_project(base, world, got, reply):
    if NEIGHBOUR in reply:
        return "one_project", False, both("The reply includes a memory filed under another project.",
                                          "La respuesta incluye un recuerdo archivado en otro proyecto.")
    return "one_project", True, both("The reply stays inside the project asked for.",
                                     "La respuesta se queda en el proyecto pedido.")


def no_value(base, world, got, reply):
    if VAULT_VALUE in reply:
        return "no_value", False, both("The reply contains the value kept in the vault.",
                                       "La respuesta contiene el valor guardado en la bóveda.")
    return "no_value", True, both("The reply is a link to Ana's own page: the value is not in it.",
                                  "La respuesta es un enlace a la página de Ana: el valor no está en ella.")


def wrong_passwords(base: str, world: dict, last: Optional[dict]) -> dict:
    """Three guesses at Ana's password on the page an app sends people to."""
    import httpx2
    from urllib.parse import parse_qs, urlparse

    registered = httpx2.post(f"{base}/register", json={"redirect_uris": ["http://localhost:3030/callback"],
                                                        "client_name": "Arena"}).json()
    statuses, opened = [], False
    for guess in ("ana12345", "limonero", "my house has a lemon tree"):
        authorize = httpx2.get(f"{base}/authorize", params={
            "client_id": registered["client_id"], "response_type": "code", "code_challenge": "x" * 43,
            "code_challenge_method": "S256", "redirect_uri": "http://localhost:3030/callback"})
        request = parse_qs(urlparse(authorize.headers["location"]).query)["r"][0]
        answer = httpx2.post(f"{base}/login", data={"r": request, "action": "signin", "name": VICTIM,
                                                     "password": guess}, headers={"accept-language": "en"})
        statuses.append(answer.status_code)
        opened = opened or "location" in answer.headers
    return {"request": "POST /login ×3", "status": statuses[-1],
            "reply": f"HTTP {', '.join(str(status) for status in statuses)}", "rule": "no_code", "held": not opened,
            "saw": (both("A guess was answered with a sign-in code.", "Un intento recibió un código de entrada.")
                    if opened else both("No guess was given a sign-in code.",
                                        "Ningún intento recibió un código de entrada."))}


def knock(bearer: Optional[Callable[[dict], str]], shown: str):
    """A move on the door itself: no token, a made-up one or a dead one."""
    def run(base, world, last):
        got = call(base, bearer(world) if bearer else None, [])
        rule, held, saw = door_closed(base, world, got, got["words"])
        return {"request": shown, "status": got["status"], "reply": got["words"], "rule": rule, "held": held,
                "saw": saw}
    return run


def tool(who: str, name: str, arguments, referee):
    """A move through a tool, with the token of one of the cast."""
    def run(base, world, last):
        sent = arguments(world, last) if callable(arguments) else arguments
        got = call(base, token(world[who]), [(name, sent)])
        reply = got["replies"][0] if got["replies"] else got["words"]
        rule, held, saw = referee(base, world, got, reply)
        return {"request": f"{name} {json.dumps(sent, ensure_ascii=False)}", "status": got["status"],
                "reply": reply, "rule": rule, "held": held, "saw": saw}
    return run


def forgets(keepsake: str):
    return lambda world, last: {"memory_ids": [world["ids"][keepsake]]}


def named_entry(world, last) -> dict:
    """The first entry the vault list just named: what an attacker would ask for next."""
    found = re.search(r"^- ([^:\n]+)", last["full"], re.MULTILINE)
    return {"entry": found.group(1).strip() if found else VAULT_ENTRY}


def opened(last: Optional[dict]) -> bool:
    return bool(last) and last["status"] == 200 and last["read"] == "worked"


def listed(last: Optional[dict]) -> bool:
    return bool(last) and "Entries in the vault" in last["full"]


# id, attacker, door, defence, what it does, how, and for a move it only makes after a success: when
MOVES = [
    ("A01", "anon", "signin", "door", both("Knocks with no token", "Llama sin token"),
     knock(None, "POST /mcp"), None),
    ("A02", "anon", "signin", "door", both("Presents a made-up token", "Presenta un token inventado"),
     knock(lambda world: "arena-made-up-token-0000000000000000000000", "POST /mcp · Bearer (made up)"), None),
    ("A03", "anon", "signin", "door", both("Guesses Ana's password three times", "Prueba tres contraseñas de Ana"),
     wrong_passwords, None),
    ("A04", "bruno", "recall", "sealed", both("Asks for Ana's secret by its own words",
                                              "Pide el secreto de Ana por sus propias palabras"),
     tool("bruno", "recall", {"query": "Ana amber heron ledger study"}, no_canary), None),
    ("A05", "bruno", "list_memories", "sealed", both("Lists everything his token can see",
                                                     "Lista todo lo que ve su token"),
     tool("bruno", "list_memories", {}, no_canary), None),
    ("A06", "bruno", "recall", "sealed", both("Names Ana's account as if it were a project",
                                              "Nombra la cuenta de Ana como si fuera un proyecto"),
     tool("bruno", "recall", {"query": "heron ledger", "project": "../ana"}, no_canary), None),
    ("A07", "bruno", "forget", "sealed", both("Tries to delete one of Ana's memories by its id",
                                              "Intenta borrar un recuerdo de Ana por su id"),
     tool("bruno", "forget", forgets(KEEPSAKES[0]), still_there(KEEPSAKES[0])), None),
    ("A08", "bruno", "shared_memories", "sealed", both("Asks for what Ana shared, though she shared nothing",
                                                       "Pide lo que Ana compartió, aunque no compartió nada"),
     tool("bruno", "shared_memories", {"person": VICTIM}, no_canary), None),
    ("A09", "bruno", "vault", "sealed", both("Asks what is in the vault", "Pregunta qué hay en la bóveda"),
     tool("bruno", "vault_list", {}, no_canary), None),
    ("A09b", "bruno", "vault", "vault", both("Asks to be shown the entry it just saw",
                                             "Pide que le enseñen la entrada que acaba de ver"),
     tool("bruno", "vault_reveal", named_entry, no_value), listed),
    ("A10", "reader", "remember", "levels", both("Tries to write a line of its own", "Intenta escribir una línea suya"),
     tool("reader", "remember", {"statement": f"Ana agreed to the fictional {PLANTED_LINE}."}, nothing_planted), None),
    ("A11", "reader", "forget", "levels", both("Tries to delete a memory", "Intenta borrar un recuerdo"),
     tool("reader", "forget", forgets(KEEPSAKES[1]), still_there(KEEPSAKES[1])), None),
    ("A12", "reader", "recall", "project", both("Asks for one project and looks for its neighbour",
                                                "Pide un proyecto y busca a su vecino"),
     tool("reader", "recall", {"query": "Harbour budget note", "project": "Harbour"}, one_project), None),
    ("A13", "adder", "forget", "levels", both("Tries to delete a memory", "Intenta borrar un recuerdo"),
     tool("adder", "forget", forgets(KEEPSAKES[2]), still_there(KEEPSAKES[2])), None),
    ("A14", "adder", "vault", "levels", both("Asks what is in the vault", "Pregunta qué hay en la bóveda"),
     tool("adder", "vault_list", {}, no_canary), None),
    ("A14b", "adder", "vault", "vault", both("Asks to be shown the entry it just saw",
                                             "Pide que le enseñen la entrada que acaba de ver"),
     tool("adder", "vault_reveal", named_entry, no_value), listed),
    ("A15", "old", "signin", "disconnect", both("Opens a session with its old token",
                                                "Abre una sesión con su token antiguo"),
     knock(lambda world: token(world["old"]), "POST /mcp · Bearer (old token)"), None),
    ("A15b", "old", "recall", "disconnect", both("Reads Ana's memory through the open session",
                                                 "Lee la memoria de Ana por la sesión abierta"),
     tool("old", "recall", {"query": "amber heron ledger"}, no_canary), opened),
]
FIRST = both("Its first idea.", "Su primera idea.")
NEXT = {
    "shut": both("That way is closed, so it tries its next idea.", "Por ahí no se pasa, así que prueba su siguiente idea."),
    "refused": both("Refused, so it tries its next idea.", "Rechazado, así que prueba su siguiente idea."),
    "failed": both("No answer to use, so it tries its next idea.",
                   "Sin respuesta que aprovechar, así que prueba su siguiente idea."),
    "nothing": both("Nothing to take, so it tries its next idea.", "Nada que llevarse, así que prueba su siguiente idea."),
    "answered": both("Nothing of Ana's in sight, so it tries its next idea.",
                     "Nada de Ana a la vista, así que prueba su siguiente idea."),
    "worked": both("That worked, and it still tries its next idea.", "Eso funcionó, y aun así prueba su siguiente idea."),
    "got": both("It got something, and it still tries its next idea.",
                "Se llevó algo, y aun así prueba su siguiente idea."),
}
FURTHER = both("Its last move got through, so it goes further.", "Su último movimiento pasó, así que va más lejos.")


def one_line(text: str, base: str, limit: int = 460) -> str:
    text = re.sub(r"\s+", " ", (text or "").replace(base, "http://clone")).strip()
    text = re.sub(r"([?&]r=)[A-Za-z0-9_-]{16,}", r"\1…", text)  # a link's one-use token is never written down
    return text if len(text) <= limit else text[:limit - 1] + "…"


def play(fault: Optional[str] = None, only: Optional[set] = None) -> List[dict]:
    """Every attacker against one fresh clone, with one fault planted or none.

    The accounts are made on the healthy server; the fault arrives afterwards, as a bad release would."""
    with tempfile.TemporaryDirectory(prefix="infinito-arena-", ignore_cleanup_errors=True) as folder:
        with clone(Path(folder)) as base:
            world = cast(base, Path(folder))
            with planted(fault):
                disconnect(Path(folder), world)  # Ana presses Disconnect while this release is live
                return attacks(base, world, only)


def attacks(base: str, world: dict, only: Optional[set]) -> List[dict]:
    """Each attacker decides from the last answer it got. A move that cannot be judged stops the recording."""
    events, last_of = [], {}
    for move, who, door, defence, title, run, when in MOVES:
        last = last_of.get(who)
        if (only is not None and move not in only) or (when is not None and not when(last)):
            continue
        started = time.perf_counter()
        out = run(base, world, last)
        kind, words = reading(out["status"], out["reply"])
        event = {"id": move, "who": who, "door": door, "defence": defence, "title": title,
                 "arrived": {"from": last["id"], "status": last["status"], "reply": last["reply"],
                             "read": last["reads"]} if last else None,
                 "decided": FURTHER if when is not None else NEXT[last["read"]] if last else FIRST,
                 "request": one_line(out["request"], base), "status": out["status"],
                 "reply": one_line(out["reply"], base), "rule": out["rule"], "held": bool(out["held"]),
                 "saw": out["saw"], "ms": round((time.perf_counter() - started) * 1000)}
        events.append(event)
        last_of[who] = dict(event, read=kind, reads=words, full=out["reply"])
    return events


def record() -> dict:
    quiet = StringIO()  # the server prints one line per sign-in step, and logs the errors a fault causes
    logging.disable(logging.CRITICAL)
    try:
        with loopback_only(), redirect_stdout(quiet), redirect_stderr(quiet):
            rounds = [{"id": "current", "fault": None, "events": play()}]
            for fault in FAULTS:
                events = play(fault)
                broken = {event["id"] for event in events if not event["held"]}
                rounds.append({"id": fault, "fault": dict(FAULTS[fault], id=fault), "events": events,
                               "replay": play(None, broken) if broken else []})
    finally:
        logging.disable(logging.NOTSET)
    for entry in rounds:
        events = entry["events"]
        entry["attacks"] = len(events)
        entry["held"] = sum(event["held"] for event in events)
        entry["through"] = [event["id"] for event in events if not event["held"]]
        if entry["fault"]:
            fixed = entry["replay"] and all(event["held"] for event in entry["replay"])
            entry["outcome"] = "caught" if fixed else "open" if entry["through"] else "contained"
            failed = any(event["reply"].startswith("Error executing tool") for event in events)
            if entry["outcome"] != "contained" or not failed:
                entry["fault"].pop("contained", None)
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                                cwd=Path(__file__).parent, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = ""
    return {
        "format": 1,
        "recorded": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": commit,
        "python": platform.python_version(),
        "system": platform.system(),
        "canary": CANARY, "vault_entry": VAULT_ENTRY,
        "agents": AGENTS, "doors": DOORS, "defences": DEFENCES, "rules": RULES,
        "rounds": rounds,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record one run of the INFINITO Security Arena")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)
    run = record()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for entry in run["rounds"]:
        print(f"{entry['id']}: {entry['held']}/{entry['attacks']} held, through: "
              f"{', '.join(entry['through']) or 'none'}" + (f", {entry['outcome']}" if entry["fault"] else ""))
    print(f"Written to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
