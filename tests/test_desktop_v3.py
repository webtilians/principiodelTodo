"""INFINITO Desktop, phase 1 (desktop/infinito_desktop): the checks, the notifications and the sealed session.

The checks are pure functions of what they read, so they run here with made-up files that reproduce the real cases of
November 2026: the backups that failed in silence from 7/11, and the engine paused on 27/11. What touches Windows (DPAPI,
the tray) is tested on Windows only.
"""
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from cognition import config as settings
from cognition.diary import Diary
from cognition.status import summary
from desktop.infinito_desktop import alerts, checks, server
from desktop.infinito_desktop.checks import FAIL, OK, UNKNOWN, WARN
from desktop.infinito_desktop.session import Store
from desktop.infinito_desktop.state import State

MADRID = timezone(timedelta(hours=1))
NOW = datetime(2026, 11, 28, 1, 0, tzinfo=MADRID)
PAGE = "https://infinitomemory.com/memory?view=cognition"
PACKAGE = Path(__file__).resolve().parents[1] / "desktop" / "infinito_desktop"


def backup_folder(tmp_path, lines, copies=()):
    folder = tmp_path / "copias-servidor"
    folder.mkdir()
    (folder / "pull.log").write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    for name in copies:
        (folder / name).write_bytes(b"x")
    return folder


def test_a_backup_that_fails_is_a_failure_with_the_day_of_the_last_good_one(tmp_path):
    folder = backup_folder(tmp_path, [
        "2026-11-07 21:00 ok: 1 new, 30 kept",
        "2026-11-27 21:00 FAILED: C:\\x\\infinito-20261114-034511.tar.gz.part is NOT usable: unexpected entry "
        "'accounts/ana/cognition.sealed'.",
        "2026-11-27 21:00 INFINITO: tu copia de seguridad mas reciente tiene mas de 3 dias. Mira pull.log"],
        copies=["infinito-20261107-034500.tar.gz", "infinito-20261106-034500.tar.gz"])
    found = checks.backups(folder, NOW)
    assert found.state == FAIL and "unexpected entry" in found.sentence and "07/11/2026" in found.sentence
    assert ".." not in found.sentence and found.link.startswith("file:")


def test_a_backup_is_well_only_with_a_recent_copy(tmp_path):
    fresh = backup_folder(tmp_path, ["2026-11-27 21:00 ok: 1 new, 30 kept"], ["infinito-20261127-034500.tar.gz"])
    assert checks.backups(fresh, NOW).state == OK
    (fresh / "infinito-20261127-034500.tar.gz").rename(fresh / "infinito-20261120-034500.tar.gz")
    stale = checks.backups(fresh, NOW)  # «ok: 0 new» every day, and no copy arrives: the case to catch
    assert stale.state == FAIL and "20/11/2026" in stale.sentence


def test_backups_this_app_cannot_read_are_never_well(tmp_path):
    assert checks.backups(tmp_path / "nowhere", NOW).state == UNKNOWN
    assert checks.backups(backup_folder(tmp_path, ["what is this"]), NOW).state == UNKNOWN


def engine_folder(tmp_path):
    folder = tmp_path / "cognition"
    folder.mkdir()
    return folder, settings.load(folder)


def test_the_engine_is_read_through_its_own_summary(tmp_path):
    assert summary(tmp_path / "nowhere", {}) is None
    assert checks.engine(None, NOW, PAGE).state == UNKNOWN
    folder, conf = engine_folder(tmp_path)
    (folder / "session.json").write_text("{}")  # only whether it is there is ever looked at
    diary = Diary(folder)
    diary.append("error", run="r1", role="supervisor", code="403 insufficient_quota")
    diary.append("review_end", run="r1", live=False, outcome="reviewed", judged=False, at="2026-11-27T04:10:00+00:00")
    said = summary(folder, conf)
    assert said["signed_in"] and said["last_review"]["problem"] == "403 insufficient_quota"
    unjudged = checks.engine(said, NOW, PAGE)
    assert unjudged.state == WARN and "insufficient_quota" in unjudged.sentence  # never «ha terminado la revisión»
    (folder / "PAUSED").write_text("2026-11-27T19:26:32.365511", encoding="utf-8")
    paused = checks.engine(summary(folder, conf), NOW, PAGE)
    assert paused.state == WARN and paused.sentence == "Está en pausa desde el 27/11/2026 a las 19:26."
    (folder / "PAUSED").unlink()
    diary.append("review_end", run="r2", live=False, outcome="reviewed", judged=True, at="2026-11-27T04:10:00+00:00")
    assert checks.engine(summary(folder, conf), NOW, PAGE).state == OK
    assert checks.engine(summary(folder, conf), NOW + timedelta(days=3), PAGE).state == WARN  # no review since


def test_an_engine_whose_diary_does_not_add_up_is_a_failure(tmp_path):
    folder, conf = engine_folder(tmp_path)
    Diary(folder).append("review_end", run="r1", live=False, outcome="reviewed", judged=True)
    lines = (folder / "diary.jsonl").read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[0])
    entry["outcome"] = "rewritten"
    (folder / "diary.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
    assert checks.engine(summary(folder, conf), NOW, PAGE).state == FAIL


SPANISH_TASKS = '''"Nombre de host","Nombre de tarea","Hora próxima ejecución","Estado","Modo de inicio de sesión","Último tiempo de ejecución","Último resultado","Autor","Tarea que se ejecutará","Iniciar en","Comentario","Estado de tarea programada","Tiempo de inactividad"
"PC","\\INFINITO copias de seguridad","28/11/2026 21:00:00","Listo","Solo interactivo","27/11/2026 21:00:01","0","PC\\ana","powershell.exe","N/A","N/A","Habilitada","Deshabilitado"
"PC","\\INFINITO copias de seguridad","28/11/2026 21:00:00","Listo","Solo interactivo","27/11/2026 21:00:01","0","PC\\ana","powershell.exe","N/A","N/A","Habilitada","Deshabilitado"
"PC","\\INFINITO F9 Worker","N/A","En ejecución","Solo interactivo","26/11/2026 0:41:13","267009","PC\\ana","python.exe","N/A","N/A","Habilitada","Deshabilitado"
"PC","\\INFINITO Motor cognitivo","28/11/2026 1:05:00","Listo","Solo interactivo","28/11/2026 0:05:02","0","PC\\ana","pythonw.exe","N/A","N/A","Habilitada","Deshabilitado"
"PC","\\OneDrive Standalone Update Task","28/11/2026 3:00:00","Listo","Solo interactivo","N/A","1","Microsoft","OneDrive.exe","N/A","N/A","Habilitada","Deshabilitado"
"Nombre de host","Nombre de tarea","Hora próxima ejecución","Estado","Modo de inicio de sesión","Último tiempo de ejecución","Último resultado","Autor","Tarea que se ejecutará","Iniciar en","Comentario","Estado de tarea programada","Tiempo de inactividad"
'''


def test_the_scheduled_tasks_are_read_in_any_language_and_only_infinitos():
    rows = checks.scheduled_tasks(SPANISH_TASKS)
    assert [row["name"] for row in rows] == ["INFINITO copias de seguridad", "INFINITO F9 Worker",
                                             "INFINITO Motor cognitivo"]
    assert checks.tasks(rows).state == OK  # 267009 is «running», not a failure
    english = ('"HostName","TaskName","Next Run Time","Status","Logon Mode","Last Run Time","Last Result","Author",'
               '"Task To Run","Start In","Comment","Scheduled Task State","Idle Time"\n'
               '"PC","\\INFINITO copias de seguridad","N/A","Disabled","Interactive only","N/A","0","a","p","N/A",'
               '"N/A","Disabled","Disabled"\n'
               '"PC","\\INFINITO Motor cognitivo","N/A","Ready","Interactive only","11/28/2026 0:05","-2147024894",'
               '"a","p","N/A","N/A","Enabled","Disabled"\n')
    said = checks.tasks(checks.scheduled_tasks(english))
    assert said.state == WARN and said.sentence.startswith("«INFINITO copias de seguridad» está deshabilitada")
    assert "terminó con el código 0x80070002" in said.sentence
    assert checks.tasks(None).state == UNKNOWN and checks.tasks([]).state == UNKNOWN


def claude(tmp_path, hooks):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hooks": hooks}), encoding="utf-8")
    return path


def test_the_hooks_are_well_only_when_both_are_there_and_have_a_session(tmp_path):
    hooks = {"UserPromptSubmit": [{"hooks": [{"type": "command", "command": "infinito-remote", "args": ["context"]}]}],
             "Stop": [{"hooks": [{"type": "command",
                                  "command": "C:\\Users\\ana\\.infinito\\venv\\Scripts\\infinito-remote.exe checkpoint"}]}]}
    session = tmp_path / "remote-session.json"
    assert checks.hooks(claude(tmp_path, hooks), session).state == WARN  # no session
    session.write_text("{}")
    assert checks.hooks(claude(tmp_path, hooks), session).state == OK
    del hooks["Stop"]
    missing = checks.hooks(claude(tmp_path, hooks), session)
    assert missing.state == WARN and "checkpoint" in missing.sentence
    (tmp_path / "settings.json").write_text("{not json", encoding="utf-8")
    assert checks.hooks(tmp_path / "settings.json", session).state == UNKNOWN
    assert checks.hooks(tmp_path / "none.json", session).state == WARN


def test_the_server_and_the_session_say_plainly_what_they_found():
    def broken():
        raise OSError("no route")

    assert checks.server(lambda: 200, "https://x").state == OK
    assert checks.server(lambda: 502, "https://x").state == FAIL and checks.server(broken, "https://x").state == FAIL
    assert [checks.session(outcome, "https://x").state for outcome in ("ok", "none", "expired", "offline")] == [
        OK, WARN, FAIL, UNKNOWN]


def test_the_icon_is_green_only_when_everything_is_well():
    make = lambda *states: [checks.Check(str(i), s, "t", "s") for i, s in enumerate(states)]  # noqa: E731
    assert checks.worst(make(OK, OK)) == OK
    assert checks.worst(make(OK, UNKNOWN)) == WARN and checks.worst(make(WARN, FAIL, OK)) == FAIL
    assert checks.worst([]) == UNKNOWN


def test_the_person_is_told_only_when_something_gets_worse_and_once_a_day():
    def look(**states):
        return [checks.Check(key, state, key, "s") for key, state in states.items()]

    seen, told = {}, {}

    def notify(day, **states):
        found = look(**states)
        news = alerts.due(found, seen, told, day)
        alerts.remember(found, seen, told, news, day)
        return sorted(check.key for check in news)

    assert notify("2026-11-28", backups=FAIL, engine=WARN, tasks=OK, hooks=UNKNOWN) == ["backups", "engine"]
    assert notify("2026-11-28", backups=FAIL, engine=WARN, tasks=OK, hooks=UNKNOWN) == []  # still bad is not news
    # Worse, but the engine was already told of today: at most one notification a day for each check.
    assert notify("2026-11-28", backups=FAIL, engine=FAIL, tasks=WARN, hooks=OK) == ["tasks"]
    assert notify("2026-11-28", backups=OK, engine=FAIL, tasks=WARN, hooks=OK) == []
    assert notify("2026-11-28", backups=FAIL, engine=FAIL, tasks=WARN, hooks=OK) == []  # once a day for each
    assert notify("2026-11-29", backups=OK, engine=FAIL, tasks=WARN, hooks=OK) == []
    assert notify("2026-11-29", backups=FAIL, engine=FAIL, tasks=WARN, hooks=OK) == ["backups"]
    assert notify("2026-11-29", backups=FAIL, engine=FAIL, tasks=FAIL, hooks=OK) == ["tasks"]  # a new day


def fake_seal(data: bytes) -> bytes:
    return bytes(b ^ 0x5A for b in data)


def test_the_session_is_never_kept_in_plain_text_and_a_broken_one_is_no_session(tmp_path, monkeypatch):
    store = Store(tmp_path, fake_seal, fake_seal)
    assert store.renew() == "none" and store.load() is None
    session = {"server": "https://x", "client_id": "c", "token_endpoint": "https://x/token",
               "access_token": "secret-access-xyz", "refresh_token": "secret-refresh-xyz", "expires_at": 0}
    store.save(session)
    sealed = (tmp_path / "session.dpapi").read_bytes()
    assert b"secret" not in sealed and store.load() == session
    monkeypatch.setattr(server, "refresh", lambda kept: {**kept, "access_token": "new", "expires_at": 9e12})
    assert store.renew() == "ok" and store.load()["access_token"] == "new"
    store.save({**session, "expires_at": 0})

    def expired(kept):
        raise server.Expired()

    monkeypatch.setattr(server, "refresh", expired)
    assert store.renew() == "expired"

    def offline(kept):
        raise OSError("no network")

    monkeypatch.setattr(server, "refresh", offline)
    assert store.renew() == "offline"
    (tmp_path / "session.dpapi").write_bytes(b"garbage")
    assert store.load() is None and store.renew() == "none"
    revoked = []
    monkeypatch.setattr(server, "revoke", revoked.append)
    store.save(session)
    store.sign_out()
    assert not (tmp_path / "session.dpapi").exists() and revoked == [session]


@pytest.mark.skipif(os.name != "nt", reason="DPAPI is Windows'")
def test_dpapi_opens_only_what_it_sealed():
    from desktop.infinito_desktop import dpapi

    sealed = dpapi.protect(b'{"refresh_token": "secret-refresh-xyz"}')
    assert b"secret" not in sealed and dpapi.unprotect(sealed) == b'{"refresh_token": "secret-refresh-xyz"}'
    with pytest.raises(OSError):
        dpapi.unprotect(sealed[:-8] + b"tampered")


def test_the_app_talks_only_to_its_own_paths_and_from_one_module():
    for refused in ("https://infinitomemory.com/mcp", "https://infinitomemory.com/memory",
                    "https://infinitomemory.com/api/devices/v1/events", "https://elsewhere.example/token"):
        with pytest.raises(PermissionError):
            server._allowed("https://infinitomemory.com", refused)
    assert server._allowed("https://infinitomemory.com", "https://infinitomemory.com/token")
    assert server.SCOPE == "infinito:device"  # a device's sign-in: no part of the memory, not even reading it
    reaching = [path.name for path in PACKAGE.glob("*.py")
                if any(word in path.read_text(encoding="utf-8") for word in ("urllib", "http.client", "urlopen"))]
    assert reaching == ["server.py"]
    for path in PACKAGE.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "forget" not in text and "vault_" not in text, path.name


def test_the_check_command_never_shows_the_session(tmp_path, monkeypatch, capsys):
    from desktop.infinito_desktop import __main__ as command, look, places

    monkeypatch.setenv("INFINITO_HOME", str(tmp_path))
    store = Store(places.home(), fake_seal, fake_seal)
    store.save({"server": "https://x", "client_id": "c", "token_endpoint": "https://x/token",
                "access_token": "secret-access-xyz", "refresh_token": "secret-refresh-xyz", "expires_at": 9e12})
    monkeypatch.setattr(command, "Store", lambda folder: Store(folder, fake_seal, fake_seal))
    monkeypatch.setattr(server, "answers", lambda address: 200)
    monkeypatch.setattr(look, "_schtasks", lambda: None)
    monkeypatch.setattr(look, "_engine_summary", lambda: None)
    assert command.main(["--check"]) == 0
    out = capsys.readouterr().out
    said = json.loads(out)
    assert "secret" not in out and {check["key"] for check in said["checks"]} == {
        "session", "server", "hooks", "engine", "backups", "tasks"}
    assert next(c for c in said["checks"] if c["key"] == "session")["state"] == OK


def test_the_state_keeps_a_device_id_and_starts_with_everything_off(tmp_path):
    state = State(tmp_path)
    assert state["start_with_windows"] is False and state["paused"] is False
    assert State(tmp_path)["device_id"] == state["device_id"]
    (tmp_path / "state.json").write_text("{broken", encoding="utf-8")
    assert State(tmp_path)["budget"]["never_on_battery"] is True


def test_the_engine_also_says_its_state_as_data_from_its_command(tmp_path, monkeypatch, capsys):
    from cognition.__main__ import main as cognition

    monkeypatch.setenv("INFINITO_COGNITION_HOME", str(tmp_path / "cognition"))
    (tmp_path / "cognition").mkdir()
    (tmp_path / "cognition" / "PAUSED").write_text("2026-11-27T19:26:32", encoding="utf-8")
    assert cognition(["status", "--json"]) in (None, 0)
    said = json.loads(capsys.readouterr().out)
    assert said["paused"] and said["paused_since"] == "2026-11-27T19:26:32" and said["signed_in"] is False
