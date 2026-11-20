"""The INFINITO Teams page (remote_teams_page.py), as people use it in a browser: create, invite, join, give access,
add a memory, and choose what an app may do. Made-up people and companies only."""
import re
from urllib.parse import parse_qs, urlparse

import pytest

from src.infinito3.remote_accounts import Accounts
from tests.test_remote_accounts_v3 import PASSWORD, app_session, server  # noqa: F401  # server is a fixture

httpx2 = pytest.importorskip("httpx2")
pytest.importorskip("cryptography")

SPANISH = {"accept-language": "es-ES"}
DECISION = "En Alfa usamos el pago en una sola pantalla."


@pytest.fixture
def on(monkeypatch, request):
    monkeypatch.setenv("INFINITO_ENTERPRISE_ENABLED", "1")
    return request.getfixturevalue("server")


def browser(base, name, invite=""):
    client = httpx2.Client(base_url=base, headers=SPANISH)
    answer = client.post("/teams", data={"action": "signin", "name": name, "password": PASSWORD, "invite": invite})
    assert answer.status_code == 303, answer.text
    return client


def send(browser_of, **form):
    """A form sent from the page itself, with its token against other sites' forms."""
    form["csrf"] = re.search(r'name="csrf" value="([^"]+)"', browser_of.get("/teams").text).group(1)
    return browser_of.post("/teams", data=form)


def where(answer) -> dict:
    return {name: values[0] for name, values in parse_qs(urlparse(answer.headers["location"]).query).items()}


def people(folder, *names):
    accounts = Accounts(folder)
    for name in names:
        accounts.create(name, PASSWORD)
    accounts.allow_teams("ceo")
    return accounts


def acme_with_alfa(base, folder):
    """The owner creates Acme and project Alfa, and invites Bea, who joins: what the page is for."""
    people(folder, "ceo", "bea", "carlos")
    ceo = browser(base, "ceo")
    org = where(send(ceo, action="create_org", name="Acme"))["org"]
    alfa = where(send(ceo, action="create_space", org=org, name="Proyecto Alfa"))["space"]
    invited = send(ceo, action="invite", org=org)
    link = re.search(r'class="code link">([^<]+)<', invited.text).group(1)
    token = parse_qs(urlparse(link).query)["invite"][0]
    bea = browser(base, "bea", invite=token)
    assert "Te han invitado" in bea.get(f"/teams?invite={token}").text
    assert "Ya estás en Acme" in send(bea, action="accept", invite=token).text
    return ceo, bea, org, alfa


def test_without_the_flag_there_is_no_teams_page(server, monkeypatch):
    monkeypatch.delenv("INFINITO_ENTERPRISE_ENABLED", raising=False)
    base, _, _ = server
    assert httpx2.get(f"{base}/teams").status_code == 404


def test_the_session_is_for_this_page_only_and_scripts_cannot_read_it(on):
    base, _, folder = on
    people(folder, "ceo")
    answer = httpx2.post(f"{base}/teams", data={"action": "signin", "name": "ceo", "password": PASSWORD})
    cookie = answer.headers["set-cookie"].lower()
    assert "path=/teams" in cookie and "httponly" in cookie and "samesite=strict" in cookie
    wrong = httpx2.post(f"{base}/teams", data={"action": "signin", "name": "ceo", "password": "no es esta"},
                        headers=SPANISH)
    assert wrong.status_code == 400 and "no son correctos" in wrong.text


def test_an_owner_creates_invites_and_gives_a_project_to_whoever_joined(on):
    base, _, folder = on
    ceo, bea, org, alfa = acme_with_alfa(base, folder)
    assert "Proyecto Alfa" not in bea.get(f"/teams?org={org}").text  # joining gives the general space only
    assert send(ceo, action="add_member", org=org, space=alfa, target="bea", role="member", write="1").status_code == 303
    assert send(bea, action="remember", org=org, space=alfa, statement=DECISION, kind="decision").status_code == 303
    for reader in (bea, ceo):
        page = reader.get(f"/teams?org={org}&space={alfa}").text
        assert DECISION in page and "decisión" in page and "· bea" in page


def test_someone_outside_a_project_sees_neither_it_nor_its_memories(on):
    base, _, folder = on
    ceo, bea, org, alfa = acme_with_alfa(base, folder)
    send(ceo, action="add_member", org=org, space=alfa, target="bea", role="member", write="1")
    send(bea, action="remember", org=org, space=alfa, statement=DECISION, kind="fact")
    carlos = browser(base, "carlos")
    page = carlos.get(f"/teams?org={org}&space={alfa}").text
    assert DECISION not in page and "No existe, o no es tuyo." in page


def test_a_form_without_the_pages_token_does_nothing(on):
    base, _, folder = on
    people(folder, "ceo")
    ceo = browser(base, "ceo")
    answer = ceo.post("/teams", data={"action": "create_org", "name": "Acme", "csrf": "otro"})
    assert answer.status_code == 403 and "Acme" not in ceo.get("/teams").text


def test_only_accounts_the_server_allowed_can_create_an_organization(on):
    base, _, folder = on
    people(folder, "ceo", "bea")
    bea = browser(base, "bea")
    assert "Crear una organización" not in bea.get("/teams").text
    assert "no puede crear organizaciones" in send(bea, action="create_org", name="Mía").text


def test_an_app_reaches_the_organization_only_once_the_person_chooses_it_on_the_page(on):
    base, _, folder = on
    ceo, bea, org, alfa = acme_with_alfa(base, folder)
    send(ceo, action="add_member", org=org, space=alfa, target="bea", role="member", write="1")
    [personal], _, app = app_session(base, [("remember", {"statement": "Mi perra se llama Nena."})], name="bea")
    page = bea.get(f"/teams?org={org}").text
    assert "Prueba" in page and 'value="none" selected' in page  # her app, with nothing yet
    [refused] = app_session(base, [("company_recall", {"org_id": org, "query": "pago"})], storage=app)[0]
    assert refused.startswith("Not allowed")
    send(bea, action="app_level", org=org, client=app.client.client_id, level="add")
    stored, recalled = app_session(base, [
        ("company_remember", {"org_id": org, "space_id": alfa, "statement": DECISION, "kind": "fact"}),
        ("company_recall", {"org_id": org, "query": "pago"})], storage=app)[0]
    assert stored.startswith("Remembered") and DECISION in recalled and "by: bea · Prueba" in recalled
    assert "Nena" not in recalled  # her own memory stays hers
    assert "Nena" not in ceo.get(f"/teams?org={org}&space={alfa}").text


def test_an_invitation_opened_by_two_people_at_once_lets_in_only_one(on, monkeypatch):
    import threading
    import time

    from src.infinito3.remote_teams import Teams

    base, _, folder = on
    people(folder, "ceo", "bea", "carlos")
    ceo = browser(base, "ceo")
    org = where(send(ceo, action="create_org", name="Acme"))["org"]
    link = re.search(r'class="code link">([^<]+)<', send(ceo, action="invite", org=org).text).group(1)
    token = parse_qs(urlparse(link).query)["invite"][0]
    slow, real = [], Teams.accept

    def accept_slowly(self, offer, account, key):
        slow.append(account)
        time.sleep(0.3)  # both requests are in at the same time
        return real(self, offer, account, key)

    monkeypatch.setattr(Teams, "accept", accept_slowly)
    guests = [browser(base, name, invite=token) for name in ("bea", "carlos")]
    forms = [re.search(r'name="csrf" value="([^"]+)"', g.get(f"/teams?invite={token}").text).group(1) for g in guests]
    answers = []
    threads = [threading.Thread(target=lambda g=g, c=c: answers.append(g.post(
        "/teams", data={"action": "accept", "invite": token, "csrf": c}))) for g, c in zip(guests, forms)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(a.status_code for a in answers) == [200, 400] and len(slow) == 1
    assert sum("Ya estás en Acme" in a.text for a in answers) == 1
