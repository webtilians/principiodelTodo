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
    session = {"server": "https://x", "client_id": "c", "token_endpoint": "https://x/token", "scope": server.SCOPE,
               "access_token": "secret-access-xyz", "refresh_token": "secret-refresh-xyz", "expires_at": 0}
    store.save(session)
    sealed = (tmp_path / "session.dpapi").read_bytes()
    assert b"secret" not in sealed and store.load() == session
    monkeypatch.setattr(server, "refresh", lambda kept: {**kept, "access_token": "new", "expires_at": 9e12})
    assert store.renew() == "ok" and store.load()["access_token"] == "new"
    store.save({**session, "scope": "infinito:device"})  # signed in with phase 1: no notes yet
    assert store.renew() == "old" and not store.may_send_notes()
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
                    "https://infinitomemory.com/api/devices/v2/events", "https://elsewhere.example/token",
                    "https://elsewhere.example/api/devices/v1/events"):
        with pytest.raises(PermissionError):
            server._allowed("https://infinitomemory.com", refused)
    assert server._allowed("https://infinitomemory.com", "https://infinitomemory.com/token")
    assert server._allowed("https://infinitomemory.com", "https://infinitomemory.com/api/devices/v1/events")
    # A device's sign-in that may add notes and nothing else: no part of the memory, not even reading it.
    assert server.SCOPE.split() == ["infinito:device", "infinito:device:note"]
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
    store.save({"server": "https://x", "client_id": "c", "token_endpoint": "https://x/token", "scope": server.SCOPE,
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


# Phase 2: the quick capture.

from desktop.infinito_desktop import capture, outbox  # noqa: E402
from desktop.infinito_desktop.waiting import FIRST_WAIT, MAX_DAYS, MAX_ITEMS, Full, Queue  # noqa: E402

DEVICE = "3b1e2f60-1c2d-4e5f-8a9b-0c1d2e3f4a5b"


def test_a_note_is_the_persons_words_with_its_kind_and_project():
    event, why = capture.note("  Llamar a Juanma por la reunión del viernes  ", "pending", " INFINITO ", DEVICE, NOW)
    assert why is None
    assert event["capability"] == "note" and event["type"] == "text" and event["origin"] == "declared"
    assert event["payload"] == {"text": "Llamar a Juanma por la reunión del viernes", "kind": "pending",
                                "project": "INFINITO"}
    assert event["observed_to"] == "2026-11-28T01:00:00+01:00" and event["device_id"] == DEVICE
    plain, _ = capture.note("Una idea", None, "", DEVICE, NOW)
    assert plain["payload"] == {"text": "Una idea"} and plain["event_id"] != event["event_id"]


def test_a_secret_never_leaves_the_pc_and_the_window_says_why():
    for secret, said in (("Mi tarjeta es 4111 1111 1111 1111", "un número de tarjeta"),
                         ("La contraseña del router es limonero-secreto-77", "una contraseña")):
        event, why = capture.note(secret, None, "", DEVICE, NOW)
        assert event is None and said in why and "1111" not in why and "limonero" not in why
    assert capture.note("   ", None, "", DEVICE, NOW)[1] == "No hay nada que guardar."
    assert "2000" in capture.note("x" * 2001, None, "", DEVICE, NOW)[1]


def test_the_queue_is_sealed_bounded_and_waits_longer_each_time(tmp_path):
    queue = Queue(tmp_path, fake_seal, fake_seal)
    event, _ = capture.note("Comprar pilas para el reloj", None, "", DEVICE, NOW)
    queue.add(event, now=1000)
    assert b"pilas" not in (tmp_path / "queue.dpapi").read_bytes()
    assert [item["event"]["event_id"] for item in queue.due(1000)] == [event["event_id"]]
    queue.later(event["event_id"], now=1000)
    assert queue.due(1000 + FIRST_WAIT - 1) == [] and len(queue.due(1000 + FIRST_WAIT)) == 1
    queue.later(event["event_id"], now=1000)
    assert queue.due(1000 + FIRST_WAIT) == [] and len(queue.due(1000 + 2 * FIRST_WAIT)) == 1  # it doubles
    assert queue.items(1000 + MAX_DAYS * 86400) == []  # a week later, it has gone
    queue.clear()
    for number in range(MAX_ITEMS):
        queue.add(capture.note(f"Nota {number}", None, "", DEVICE, NOW)[0], now=1000)
    with pytest.raises(Full):
        queue.add(event, now=1000)
    queue.clear()
    assert queue.items(1000) == [] and not (tmp_path / "queue.dpapi").exists()


def test_each_note_leaves_the_queue_only_when_the_server_has_said_its_last_word(tmp_path):
    queue = Queue(tmp_path, fake_seal, fake_seal)
    notes = [capture.note(f"Nota {n}", None, "", DEVICE, NOW)[0] for n in range(4)]
    for note in notes:
        queue.add(note, now=1000)
    answers = iter([(200, {"saved": True, "memory_id": "m1"}), (200, {"saved": True, "duplicate": True}),
                    (400, {"error": "not_saved", "detail": "Not stored: it looks like a password, and secrets..."}),
                    (429, {"error": "too_fast", "retry_after": "3"})])
    happened = outbox.deliver(queue, lambda event: next(answers), now=1000)
    assert [item["outcome"] for item in happened] == ["kept", "kept", "refused", "later"]
    assert happened[1]["said"] == "Ya estaba en tu memoria." and "contraseña" not in happened[2]["said"]
    assert "tu memoria no lo guarda" in happened[2]["said"]
    assert [item["event"]["event_id"] for item in queue.items(1000)] == [notes[3]["event_id"]]
    assert queue.due(1002) == [] and len(queue.due(1003)) == 1  # it waits what the server asked

    def offline(event):
        raise OSError("no network")

    assert outbox.deliver(queue, offline, now=1003)[0]["outcome"] == "later"
    assert len(queue.items(1003)) == 1  # it stays, to go when the network is back
    assert outbox.deliver(queue, lambda event: (401, {"error": "invalid_token"}), now=2000)[0]["outcome"] == "later"


def test_a_note_waits_while_the_app_cannot_send_it(tmp_path, monkeypatch):
    store = Store(tmp_path, fake_seal, fake_seal)
    with pytest.raises(OSError):
        store.send({"event_id": "x"})  # never signed in
    session = {"server": "https://x", "client_id": "c", "token_endpoint": "https://x/token", "scope": "infinito:device",
               "access_token": "a", "refresh_token": "r", "expires_at": 9e12}
    store.save(session)
    with pytest.raises(OSError):
        store.send({"event_id": "x"})  # signed in before it could send notes
    store.save({**session, "scope": server.SCOPE, "expires_at": 0})

    def expired(kept):
        raise server.Expired()

    monkeypatch.setattr(server, "refresh", expired)
    assert store.send({"event_id": "x"}) == (401, {"error": "invalid_token"})
    sent = []
    monkeypatch.setattr(server, "refresh", lambda kept: {**kept, "access_token": "fresh", "expires_at": 9e12})
    monkeypatch.setattr(server, "send", lambda kept, event: sent.append((kept["access_token"], event)) or (200, {}))
    assert store.send({"event_id": "x"}) == (200, {}) and sent == [("fresh", {"event_id": "x"})]


def test_signing_in_again_ends_the_session_before(tmp_path, monkeypatch):
    store = Store(tmp_path, fake_seal, fake_seal)
    old = {"server": "https://x", "client_id": "old", "token_endpoint": "https://x/token", "scope": "infinito:device",
           "access_token": "a", "refresh_token": "r", "expires_at": 9e12}
    store.save(old)
    revoked = []
    monkeypatch.setattr(server, "revoke", revoked.append)
    monkeypatch.setattr(server, "sign_in", lambda address, **kw: {**old, "client_id": "new", "scope": server.SCOPE})
    store.sign_in("https://x")
    assert store.load()["client_id"] == "new" and [s["client_id"] for s in revoked] == ["old"]
    assert store.may_send_notes()
