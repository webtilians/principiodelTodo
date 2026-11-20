"""INFINITO Teams through MCP: the company_* tools, as Claude or ChatGPT call them after signing in.

An app gets nothing of an organization until the person chooses (A16), no tool takes who is asking from its arguments
(A06), and a space id that is not the user's opens nothing (A03)."""
import pytest

from src.infinito3.remote_accounts import Accounts
from src.infinito3.remote_teams import NOT_FOUND, Teams
from tests.test_remote_accounts_v3 import PASSWORD, app_session, server  # noqa: F401  # server is a fixture

pytest.importorskip("cryptography")
pytest.importorskip("mcp.server.mcpserver")

DECISION = "En Alfa usamos el pago en una sola pantalla."


@pytest.fixture
def on(monkeypatch, request):
    monkeypatch.setenv("INFINITO_ENTERPRISE_ENABLED", "1")
    return request.getfixturevalue("server")


def acme(folder):
    """What the owner does on the Teams page (phase 4): Acme, with a project, made straight on the server's files."""
    accounts = Accounts(folder)
    ceo = accounts.open("ceo", PASSWORD)
    accounts.allow_teams("ceo")
    teams = Teams(accounts)
    org = teams.create_org("ceo", ceo, "Acme")
    return accounts, teams, org, teams.create_space(org, "ceo", ceo, "Proyecto Alfa"), ceo


def test_without_the_flag_there_is_no_company_tool(server, monkeypatch):
    monkeypatch.delenv("INFINITO_ENTERPRISE_ENABLED", raising=False)
    base, _, _ = server
    _, _, app = app_session(base, [("recall", {"query": "x"})], mode="signup", name="ceo")
    assert not [tool.name for tool in app.tools if tool.name.startswith("company_")]


def test_a_new_app_gets_nothing_of_an_organization_until_the_person_chooses(on):
    base, _, folder = on
    [none], _, app = app_session(base, [("company_list_orgs", {})], mode="signup", name="ceo")
    assert "in no organization" in none
    accounts, teams, org, alfa, _ = acme(folder)
    listed, spaces, recalled = app_session(base, [("company_list_orgs", {}), ("company_list_spaces", {"org_id": org}),
                                                  ("company_recall", {"org_id": org, "query": "pago"})], storage=app)[0]
    assert f"Acme (org_id {org}): the user is owner; this app may 'none'" in listed
    assert spaces.startswith("Not allowed") and recalled.startswith("Not allowed")

    teams.set_app_level(org, "ceo", app.client.client_id, "read")
    stored, recalled = app_session(base, [
        ("company_remember", {"org_id": org, "space_id": alfa, "statement": DECISION, "kind": "decision"}),
        ("company_recall", {"org_id": org, "query": "pago"})], storage=app)[0]
    assert stored.startswith("Not allowed") and "needs 'add'" in stored and DECISION not in recalled

    teams.set_app_level(org, "ceo", app.client.client_id, "add")
    stored, recalled = app_session(base, [
        ("company_remember", {"org_id": org, "space_id": alfa, "statement": DECISION, "kind": "decision",
                              "idempotency_key": "k-1"}),
        ("company_remember", {"org_id": org, "space_id": alfa, "statement": DECISION, "kind": "decision",
                              "idempotency_key": "k-1"}),
        ("company_recall", {"org_id": org, "query": "pago", "space_ids": [alfa]})], storage=app)[0][::2]
    assert stored.startswith("Remembered as") and "proposal" in stored
    assert recalled.count(DECISION) == 1 and "proposal, by: ceo · Prueba" in recalled
    assert f"## Proyecto Alfa (project, space_id {alfa})" in recalled


def test_no_company_tool_takes_who_is_asking(on):
    base, _, _ = on
    _, _, app = app_session(base, [("company_list_orgs", {})], mode="signup", name="ceo")
    for tool in (t for t in app.tools if t.name.startswith("company_")):
        assert not {"author", "account", "person", "role", "principal", "user", "owner"} & set(
            tool.input_schema.get("properties", {})), tool.name


def test_a_space_that_is_not_the_users_opens_nothing_whatever_the_app_asks(on):
    base, _, folder = on
    app_session(base, [("company_list_orgs", {})], mode="signup", name="ceo")
    accounts, teams, org, alfa, ceo = acme(folder)
    teams.remember(org, alfa, "ceo", ceo, DECISION, kind="fact")
    _, _, bea_app = app_session(base, [("company_list_orgs", {})], mode="signup", name="bea")
    offer = {}
    teams.invite(org, "ceo", ceo, lambda key, data: offer.update(key=key, data=data, account="ceo") or "token")
    teams.accept(offer, "bea", accounts.open("bea", PASSWORD))
    teams.set_app_level(org, "bea", bea_app.client.client_id, "add")
    general, alfa_asked, alfa_written = app_session(base, [
        ("company_recall", {"org_id": org, "query": "pago"}),
        ("company_recall", {"org_id": org, "query": "pago", "space_ids": [alfa]}),
        ("company_remember", {"org_id": org, "space_id": alfa, "statement": "Bea estuvo aquí."})],
        storage=bea_app)[0]
    assert "## General" in general and "Proyecto Alfa" not in general and DECISION not in general
    assert alfa_asked == NOT_FOUND and alfa_written == NOT_FOUND


def test_an_organization_that_does_not_exist_answers_like_one_that_is_not_the_users(on):
    base, _, folder = on
    app_session(base, [("company_list_orgs", {})], mode="signup", name="ceo")
    _, _, org, _, _ = acme(folder)
    _, _, bea_app = app_session(base, [("company_list_orgs", {})], mode="signup", name="bea")
    theirs, made_up = app_session(base, [("company_list_spaces", {"org_id": org}),
                                         ("company_list_spaces", {"org_id": "0123456789abcdef"})], storage=bea_app)[0]
    assert theirs == made_up and theirs.startswith("Not allowed")
