"""INFINITO Teams actions (remote_teams_actions.py, ADR P3): a company's own systems, asked for from its people's AIs.

The server runs an action, never the model; anything that is not "direct" waits for a person on /teams, and no tool
confirms; what was asked is sealed with its space; each action that ends well leaves a memory with who, when and its
reference. A made-up accounting firm (Acme) and a made-up adapter only."""
import json
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from src.infinito3 import remote_teams_actions as actions_module
from src.infinito3.remote_accounts import Accounts
from src.infinito3.remote_teams import REQUESTS, Refused, Teams
from src.infinito3.remote_teams_actions import Actions, Catalogue, call_adapter, check_inputs, problems
from tests.test_remote_accounts_v3 import PASSWORD, app_session, server  # noqa: F401  # server is a fixture

pytest.importorskip("cryptography")

SECRET = "un secreto de adaptador bastante largo"
INVOICE = {"cliente": "Panadería Sol", "proveedor": "Harinas del Sur", "fecha": "2026-11-20", "base": 1200}


def catalogue(org: str, space: str, url: str = "http://127.0.0.1:1") -> dict:
    return {"adapters": {"gestoria-demo": {"url": url, "secret": SECRET,
                                           "operations": ["vencimientos", "registrar_factura"]}},
            "orgs": {org: [
                {"id": "consultar_vencimientos", "space": space, "name": "Consultar vencimientos",
                 "description": "Obligaciones de un cliente.", "adapter": "gestoria-demo", "operation": "vencimientos",
                 "mode": "direct", "writes": False, "remember": False, "input": {"cliente": "text<=60"}},
                {"id": "registrar_factura", "space": space, "name": "Registrar factura recibida",
                 "adapter": "gestoria-demo", "operation": "registrar_factura", "mode": "self",
                 "input": {"cliente": "text<=60", "proveedor": "text<=80", "fecha": "date", "base": "number",
                           "nota": "text<=100?"}},
                {"id": "registrar_doble", "space": space, "name": "Registrar con doble control",
                 "adapter": "gestoria-demo", "operation": "registrar_factura", "mode": "second",
                 "input": {"cliente": "text<=60"}}]}}


class Firm:
    """Acme: ceo owns it; ana adds to Clientes, carlos only reads it, bea is outside it. Calls to the adapter are
    recorded instead of sent, unless a test gives another call."""

    def __init__(self, folder, call=None):
        self.accounts = Accounts(folder)
        self.keys = {name: self.accounts.create(name, PASSWORD)[0] for name in ("ceo", "ana", "carlos", "bea")}
        self.accounts.allow_teams("ceo")
        self.teams = Teams(self.accounts)
        self.org = self.teams.create_org("ceo", self.keys["ceo"], "Acme")
        for name in ("ana", "carlos", "bea"):
            offer = {}
            self.teams.invite(self.org, "ceo", self.keys["ceo"],
                              lambda key, data: offer.update(key=key, data=data, account="ceo") or "token")
            self.teams.accept(offer, name, self.keys[name])
        self.space = self.teams.create_space(self.org, "ceo", self.keys["ceo"], "Clientes")
        self.teams.add(self.org, self.space, "ceo", self.keys["ceo"], "ana")
        self.teams.add(self.org, self.space, "ceo", self.keys["ceo"], "carlos", write=False)
        self.path = folder / "actions.json"
        self.path.write_text(json.dumps(catalogue(self.org, self.space)), encoding="utf-8")
        self.calls = []
        self.actions = Actions(self.teams, Catalogue(self.path), call or self.record)

    def record(self, adapter, operation, payload):
        self.calls.append((adapter["url"], operation, payload))
        if operation == "registrar_factura":
            return {"ok": True, "result": {"factura": 1043}, "ref": "factura 1043", "error": None}
        return {"ok": True, "result": [{"modelo": "303", "vence": "2027-01-30"}], "ref": None, "error": None}

    def ask(self, who, action, inputs, **extra):
        return self.actions.ask(self.org, who, self.keys[who], "Claude", action, inputs, **extra)

    def confirm(self, who, request, yes=True):
        return self.actions.confirm(self.org, who, self.keys[who], request, yes)

    def memories(self, who="ceo"):
        return [m["text"] for m in self.teams.memories(self.org, self.space, who, self.keys[who])]

    def policy(self):
        return json.loads((self.accounts.folder / "orgs" / self.org / "policy.json").read_text(encoding="utf-8"))


@pytest.fixture
def firm(tmp_path):
    return Firm(tmp_path)


# The catalogue and the inputs

def test_a_catalogue_says_what_is_wrong_with_it(firm):
    good = catalogue(firm.org, firm.space)
    assert problems(good) == []
    bad = json.loads(json.dumps(good))
    bad["adapters"]["gestoria-demo"]["url"] = "http://10.0.0.5/api"  # http only on this machine
    bad["adapters"]["gestoria-demo"]["secret"] = "corto"
    bad["orgs"][firm.org][0]["operation"] = "borrar_todo"
    bad["orgs"][firm.org][1]["input"]["base"] = "dinero"
    bad["orgs"][firm.org][2]["id"] = "consultar_vencimientos"
    found = "\n".join(problems(bad))
    for part in ("url must be https", "secret must be", "operation: one of its adapter's", "input 'base'", "twice"):
        assert part in found, part
    assert problems([]) and problems({"adapters": {}, "orgs": {"no-es-un-id": []}})


def test_an_unusable_catalogue_counts_as_empty(firm):
    firm.path.write_text("{ esto no es json", encoding="utf-8")
    assert firm.actions.available(firm.org, "ceo") == []
    assert "cannot be read" in firm.actions.catalogue.problems[0]


def test_only_the_actions_fields_reach_the_adapter_and_as_their_types(firm):
    schema = catalogue(firm.org, firm.space)["orgs"][firm.org][1]["input"]
    cleaned = check_inputs(schema, {**INVOICE, "base": "1.200,5".replace(".", ""), "fecha": "2026-11-20"})
    assert cleaned["base"] == 1200.5 and cleaned["fecha"] == "2026-11-20" and "nota" not in cleaned
    for wrong, said in (({**INVOICE, "borrar": True}, "takes no borrar"), ({"cliente": "Sol"}, "is needed"),
                        ({**INVOICE, "fecha": "20/11/2026"}, "a date"), ({**INVOICE, "base": "mucho"}, "a number"),
                        ({**INVOICE, "cliente": "x" * 61}, "up to 60"), ("no es un objeto", "JSON object")):
        with pytest.raises(Refused, match=said):
            check_inputs(schema, wrong)


# Asking, confirming, and what is left behind

def test_a_direct_action_runs_at_once_and_its_result_is_data(firm):
    done = firm.ask("carlos", "consultar_vencimientos", {"cliente": "Panadería Sol"})  # carlos only reads: a lookup
    assert done["state"] == "done" and done["result"]["result"] == [{"modelo": "303", "vence": "2027-01-30"}]
    [(_, operation, payload)] = firm.calls
    assert operation == "vencimientos" and payload["inputs"] == {"cliente": "Panadería Sol"}
    assert payload["by"] == "carlos" and payload["app"] == "Claude"
    assert firm.memories() == []  # this one is set not to leave a memory


def test_an_action_that_writes_waits_for_the_person_and_then_leaves_a_memory(firm):
    asked = firm.ask("ana", "registrar_factura", INVOICE)
    assert asked["state"] == "waiting" and firm.calls == []  # nothing runs before a person confirms
    with pytest.raises(Refused, match="not yours to confirm"):
        firm.confirm("carlos", asked["id"])  # in the space, but neither the one who asked nor someone who runs it
    assert any(e["action"].startswith("act_confirm") for e in firm.policy()["refused"])
    done = firm.confirm("ana", asked["id"])
    assert done["state"] == "done" and len(firm.calls) == 1
    [memory] = firm.memories()
    assert memory == "ana hizo «Registrar factura recibida» con Claude: factura 1043"  # the reference, not the data
    with pytest.raises(Refused, match="no longer waiting"):
        firm.confirm("ana", asked["id"])  # it ran once


def test_with_two_people_the_one_who_asked_cannot_confirm(firm):
    asked = firm.ask("ana", "registrar_doble", {"cliente": "Panadería Sol"})
    with pytest.raises(Refused, match="not yours to confirm"):
        firm.confirm("ana", asked["id"])
    firm.confirm("ceo", asked["id"])
    assert firm.memories() == ["ana hizo «Registrar con doble control» con Claude, confirmada por ceo: factura 1043"]


def test_a_person_may_turn_a_request_down(firm):
    asked = firm.ask("ana", "registrar_factura", INVOICE)
    turned = firm.confirm("ana", asked["id"], yes=False)
    assert turned["state"] == "rejected" and firm.calls == [] and firm.memories() == []
    assert "args" not in firm.policy()[REQUESTS][asked["id"]]  # what was asked does not stay


def test_only_who_may_add_to_the_space_asks_for_what_changes_data(firm):
    with pytest.raises(Refused, match="not ask for this action"):
        firm.ask("carlos", "registrar_factura", INVOICE)
    with pytest.raises(Refused, match="Not found"):
        firm.ask("bea", "consultar_vencimientos", {"cliente": "Sol"})  # outside the space: it is not there
    assert [a["id"] for a in firm.actions.available(firm.org, "bea")] == []
    usable = {a["id"]: a["usable"] for a in firm.actions.available(firm.org, "carlos")}
    assert usable == {"consultar_vencimientos": True, "registrar_factura": False, "registrar_doble": False}


def test_what_was_asked_is_sealed_and_never_in_the_record(firm):
    firm.ask("ana", "registrar_factura", INVOICE)
    on_disk = (firm.accounts.folder / "orgs" / firm.org / "policy.json").read_text(encoding="utf-8")
    assert "Harinas" not in on_disk and "Panadería" not in on_disk and SECRET not in on_disk
    assert [e["action"] for e in firm.policy()["audit"]][-1] == "act_ask:registrar_factura"


def test_the_same_idempotency_key_asks_once(firm):
    first = firm.ask("ana", "registrar_factura", INVOICE, idempotency_key="k-1")
    again = firm.ask("ana", "registrar_factura", INVOICE, idempotency_key="k-1")
    assert first["id"] == again["id"] and len(firm.policy()[REQUESTS]) == 1


def test_leaving_calls_off_what_was_waiting(firm):
    asked = firm.ask("ana", "registrar_factura", INVOICE)
    firm.teams.remove(firm.org, firm.space, "ceo", "ana")
    entry = firm.policy()[REQUESTS][asked["id"]]
    assert entry["state"] == "cancelled" and "args" not in entry
    with pytest.raises(Refused):
        firm.confirm("ceo", asked["id"])


def test_a_request_still_opens_after_the_space_changes_key(firm):
    asked = firm.ask("ana", "registrar_factura", INVOICE)
    firm.teams.remove(firm.org, firm.space, "ceo", "carlos")  # the space gets a new key the next time it is opened
    [waiting] = firm.actions.waiting_for(firm.org, "ceo", firm.keys["ceo"])
    assert waiting["args"]["proveedor"] == "Harinas del Sur"
    firm.confirm("ceo", asked["id"])
    assert firm.policy()["spaces"][firm.space]["version"] == 2 and firm.calls[0][2]["inputs"]["base"] == 1200.0


def test_when_the_system_fails_nothing_is_retried_and_nothing_is_remembered(tmp_path):
    firm = Firm(tmp_path, call=lambda *_: {"ok": False, "error": "el programa está caído"})
    asked = firm.ask("ana", "registrar_factura", INVOICE)
    failed = firm.confirm("ana", asked["id"])
    assert failed["state"] == "failed" and failed["result"]["error"] == "el programa está caído"
    assert firm.memories() == [] and len(firm.policy()[REQUESTS]) == 1
    assert "failed" in firm.actions.describe(firm.org, "ana", firm.keys["ana"], "/t").splitlines()[-1]


# The adapter, over HTTP

class _Adapter(BaseHTTPRequestHandler):
    answers = {}

    def do_POST(self):  # noqa: N802
        self.server.seen.append((self.path, self.headers.get("Authorization"),
                                 json.loads(self.rfile.read(int(self.headers["Content-Length"])))))
        status, body = self.answers.get(self.path, (200, b'{"result": {"ok": 1}, "ref": "exp-7"}'))
        self.send_response(status)
        if status == 302:
            self.send_header("Location", "http://127.0.0.1:9/elsewhere")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def adapter():
    running = ThreadingHTTPServer(("127.0.0.1", 0), _Adapter)
    running.seen = []
    threading.Thread(target=running.serve_forever, daemon=True).start()
    yield running, {"url": f"http://127.0.0.1:{running.server_address[1]}/api", "secret": SECRET}
    running.shutdown()


def test_the_adapter_gets_the_secret_and_the_inputs_and_answers_data(adapter):
    running, config = adapter
    answer = call_adapter(config, "registrar_factura", {"inputs": {"cliente": "Sol"}})
    assert answer == {"ok": True, "result": {"ok": 1}, "ref": "exp-7", "error": None}
    [(path, authorization, body)] = running.seen
    assert path == "/api/registrar_factura" and authorization == f"Bearer {SECRET}" and body["inputs"]["cliente"] == "Sol"


def test_an_adapter_that_redirects_fails_or_says_too_much_is_not_followed(adapter, monkeypatch):
    _, config = adapter
    monkeypatch.setattr(_Adapter, "answers", {"/api/redirige": (302, b""), "/api/roto": (500, b"{}"),
                                              "/api/texto": (200, b"no es json"),
                                              "/api/enorme": (200, b'{"result": "' + b"x" * 70000 + b'"}'),
                                              "/api/niega": (200, b'{"ok": false, "error": "sin permiso"}')})
    said = {op: call_adapter(config, op, {}) for op in ("redirige", "roto", "texto", "enorme", "niega")}
    assert all(not answer["ok"] for answer in said.values())
    assert "302" in said["redirige"]["error"] and "500" in said["roto"]["error"]
    assert "JSON" in said["texto"]["error"] and "too much" in said["enorme"]["error"]
    assert said["niega"]["error"] == "sin permiso"
    assert "could not be reached" in call_adapter({"url": "http://127.0.0.1:1", "secret": SECRET}, "x", {})["error"]


# Through MCP and the Teams page, with both flags on

@pytest.fixture
def on(monkeypatch, request, tmp_path):
    monkeypatch.setenv("INFINITO_ENTERPRISE_ENABLED", "1")
    monkeypatch.setenv("INFINITO_TEAMS_ACTIONS_ENABLED", "1")
    return request.getfixturevalue("server")


def test_without_the_actions_flag_there_is_nothing_new(server, monkeypatch, tmp_path):
    monkeypatch.setenv("INFINITO_ENTERPRISE_ENABLED", "1")
    monkeypatch.delenv("INFINITO_TEAMS_ACTIONS_ENABLED", raising=False)
    base, _, folder = server
    _, _, app = app_session(base, [("company_list_orgs", {})], mode="signup", name="ceo")
    assert not {"company_actions", "company_act"} & {tool.name for tool in app.tools}
    accounts = Accounts(folder)
    accounts.allow_teams("ceo")
    teams = Teams(accounts)
    org = teams.create_org("ceo", accounts.open("ceo", PASSWORD), "Acme")
    with pytest.raises(Refused):
        teams.set_app_level(org, "ceo", app.client.client_id, "act")


def test_an_app_asks_for_actions_only_with_act_and_never_confirms(on, adapter):
    base, _, folder = on
    _, config = adapter
    _, _, app = app_session(base, [("company_list_orgs", {})], mode="signup", name="ceo")
    accounts = Accounts(folder)
    accounts.allow_teams("ceo")
    ceo = accounts.open("ceo", PASSWORD)
    teams = Teams(accounts)
    org = teams.create_org("ceo", ceo, "Acme")
    space = teams.create_space(org, "ceo", ceo, "Clientes")
    data = catalogue(org, space, config["url"])
    (folder / "actions.json").write_text(json.dumps(data), encoding="utf-8")
    names = {tool.name for tool in app.tools}
    assert {"company_actions", "company_act"} <= names
    assert not [n for n in names if "confirm" in n or "approve" in n]  # no tool confirms anything

    teams.set_app_level(org, "ceo", app.client.client_id, "add")
    [refused] = app_session(base, [("company_act", {"org_id": org, "action_id": "consultar_vencimientos",
                                                    "inputs": {"cliente": "Sol"}})], storage=app)[0]
    assert refused.startswith("Not allowed") and "needs 'act'" in refused

    teams.set_app_level(org, "ceo", app.client.client_id, "act")
    listed, looked, asked = app_session(base, [
        ("company_actions", {"org_id": org}),
        ("company_act", {"org_id": org, "action_id": "consultar_vencimientos", "inputs": {"cliente": "Sol"}}),
        ("company_act", {"org_id": org, "action_id": "registrar_factura", "inputs": INVOICE})], storage=app)[0]
    assert "registrar_factura: Registrar factura recibida" in listed and "fecha (a date as YYYY-MM-DD)" in listed
    assert looked.startswith("Done: Consultar vencimientos") and "reference exp-7" in looked
    assert "waits for the user to confirm it on" in asked and "You cannot confirm it" in asked
    assert SECRET not in listed + looked + asked
    request_id = re.search(r"request ([0-9a-f]{16})", asked).group(1)

    # The person confirms on the Teams page, the only place it can be done.
    import httpx2

    page = httpx2.Client(base_url=base, headers={"accept-language": "es-ES"})
    assert page.post("/teams", data={"action": "signin", "name": "ceo", "password": PASSWORD}).status_code == 303
    shown = page.get(f"/teams?org={org}").text
    assert "Herramientas conectadas" in shown and "Esperan tu confirmación" in shown and "Harinas del Sur" in shown
    assert "ceo pidió «Registrar factura recibida» con Prueba" in shown and "la confirma quien la pide" in shown
    csrf = re.search(r'name="csrf" value="([^"]+)"', shown).group(1)
    done = page.post("/teams", data={"action": "act_confirm", "org": org, "request": request_id, "csrf": csrf})
    assert "Hecho: Registrar factura recibida." in done.text and "Esperan tu confirmación" not in done.text
    [after] = app_session(base, [("company_actions", {"org_id": org})], storage=app)[0]
    assert f"{request_id} · Registrar factura recibida · done" in after
    space_page = page.get(f"/teams?org={org}&space={space}").text
    assert "ceo hizo «Registrar factura recibida» con Prueba: exp-7" in space_page
