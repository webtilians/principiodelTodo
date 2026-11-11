"""The cognitive engine's reviews on the person's page: kept sealed, read in full, decided only by the person."""
import json
import re

import pytest

from src.infinito3 import remote_apps, remote_cognition as cognition
from src.infinito3.remote_backup import MEMBER

KEY = bytes(range(32))


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "accounts" / "ana"


def review(run="r202611111735", **extra):
    return {"run": run, "summary": "Un día con avances.",
            "reflection": {"id": "b25bc042", "text": "Hoy veo un avance real.", "advice": "Comprobar antes de creer.",
                           "questions": ["¿Durará?"], "sources": ["m1", "m2"], "model": "gpt-6-astra"},
            "coverage": {"read": 39, "reported": 39, "nobody": 11, "limits": []},
            "audit": [{"role": "observer", "issue": "omission", "note": "Se le pasó.", "ids": ["m2"]}],
            "hypotheses": [{"id": "H1", "statement": "El arranque podría fallar.", "falsifier": "Una ejecución real.",
                            "verdict": "supported", "why": "Consta.", "ids": ["m1"]}],
            "costs": {"spent_eur": 0.477, "cap_eur": 1.0},
            "proposals": [{"id": "P1", "target_role": "observer", "guideline": "Separa decisión y ejecución.",
                           "rationale": "Las confunde.", "test": "Reprocesar."},
                          {"id": "P2", "target_role": "nobody", "guideline": "x"}],
            **extra}


def change(folder, action):
    return cognition.change(folder, KEY, "ana", action)


def test_a_review_is_kept_sealed_with_its_guidelines_waiting(folder):
    kept = change(folder, lambda data: cognition.report(data, "INFINITO Motor cognitivo", review()))
    assert kept["by"] == "INFINITO Motor cognitivo" and [p["id"] for p in kept["proposals"]] == ["P1"]
    assert b"avance" not in (folder / cognition.FILE).read_bytes()  # sealed
    data = cognition.read(folder, KEY, "ana")
    assert cognition.unseen(data) == 1 and cognition.waiting(data) == 1
    with pytest.raises(cognition.Refused, match="Already kept"):
        change(folder, lambda data: cognition.report(data, "x", review()))
    with pytest.raises(ValueError):
        cognition.read(folder, bytes(32), "ana")  # another key opens nothing


def test_a_secret_never_goes_in(folder):
    with pytest.raises(cognition.Refused, match="secret"):
        change(folder, lambda data: cognition.report(data, "x", review(summary="Mi contraseña es limonero-secreto-77")))
    assert not (folder / cognition.FILE).exists() or cognition.read(folder, KEY, "ana")["reviews"] == []


def test_only_the_person_decides_and_the_engine_reads_it(folder):
    change(folder, lambda data: cognition.report(data, "x", review()))
    assert change(folder, lambda data: cognition.decide(data, "P1", "approve", "Vale")) == "approved"
    with pytest.raises(cognition.Refused):
        change(folder, lambda data: cognition.decide(data, "P1", "reject"))  # already decided
    [decided] = json.loads(cognition.decisions(cognition.read(folder, KEY, "ana")))
    assert decided["status"] == "approved" and decided["note"] == "Vale"
    assert change(folder, lambda data: cognition.decide(data, "P1", "revoke")) == "revoked"
    # No tool decides: the engine's tools only keep a review and read decisions.
    assert set(cognition.TOOL_DESCRIPTIONS) == {"cognition_report", "cognition_decisions"}
    assert remote_apps.allowed("add", "cognition_report") and remote_apps.allowed("read", "cognition_decisions")
    assert not remote_apps.allowed("read", "cognition_report")


def test_forgetting_takes_what_cites_the_forgotten(folder):
    change(folder, lambda data: cognition.report(data, "x", review()))
    assert change(folder, lambda data: cognition.drop_citing(data, ["m2"])) == 2
    kept = cognition.read(folder, KEY, "ana")["reviews"][0]
    assert kept["audit"] == [] and kept["reflection"]["text"] == "" and len(kept["hypotheses"]) == 1


def test_the_notice_shows_until_read_and_while_a_guideline_waits(folder):
    t = cognition.TEXT["es"]
    change(folder, lambda data: cognition.report(data, "x", review()))
    data = cognition.read(folder, KEY, "ana")
    assert "dejó una revisión nueva" in cognition.notice(t, data) and "1 pauta" in cognition.notice(t, data)
    change(folder, cognition.mark_seen)
    assert "esperando tu decisión" in cognition.notice(t, cognition.read(folder, KEY, "ana"))
    change(folder, lambda data: cognition.decide(data, "P1", "reject"))
    assert cognition.notice(t, cognition.read(folder, KEY, "ana")) == ""


def test_its_file_goes_in_the_backups():
    assert MEMBER.fullmatch("accounts/ana/cognition.sealed")


# The whole way: the engine reports over MCP, the person reads and decides on the page, the engine reads it back

def test_the_engine_reports_and_the_person_decides_on_the_page(server):  # noqa: F811
    from tests.test_remote_accounts_v3 import app_session
    from tests.test_remote_memory_page_v3 import browser, csrf_of, enter

    base, _, _ = server
    steps = [("remember", {"statement": "Decidimos probar el motor cognitivo.", "kind": "decision"}),
             ("cognition_report", {"review": review()})]
    (_, kept), _, _ = app_session(base, steps, mode="signup", name="iris")
    assert kept.startswith("Kept as review") and "1 guideline(s) waiting" in kept
    with browser() as client:
        home = enter(client, base, "iris")
        assert "Tu motor cognitivo dejó una revisión nueva" in home.text and "view=cognition" in home.text
        page = client.get(f"{base}/memory", params={"view": "cognition"})
        for words in ("Un día con avances.", "Hoy veo un avance real.", "Se le pasó.", "El arranque podría fallar.",
                      "Separa decisión y ejecución.", "Aceptar", "Rechazar", "0.477"):
            assert words in page.text, words
        decided = client.post(f"{base}/memory", data={"action": "cognition_approve", "id": "P1", "text": "",
                                                      "csrf": csrf_of(page)})
        assert "Pauta aceptada" in decided.text and "Retirar" in decided.text
        assert "dejó una revisión nueva" not in client.get(f"{base}/memory").text  # read, and nothing waits
        forged = client.post(f"{base}/memory", data={"action": "cognition_revoke", "id": "P1", "csrf": "not-it"})
        assert forged.status_code == 403
    (decisions,), _, _ = app_session(base, [("cognition_decisions", {})], name="iris")
    assert json.loads(decisions)[0]["status"] == "approved"


from tests.test_remote_accounts_v3 import server  # noqa: E402,F401  # the fixture
