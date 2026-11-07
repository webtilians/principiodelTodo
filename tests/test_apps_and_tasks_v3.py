"""What each app may do with a person's memory, and the work queue between their AIs that only the person approves."""
import re

import pytest

from src.infinito3 import remote_apps, remote_tasks
from src.infinito3.mcp_server import BY_THE_USER
from src.infinito3.remote_backup import MEMBER

KEY = bytes(range(32))


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "accounts" / "ana"


def change(folder, action):
    return remote_tasks.change(folder, KEY, "ana", action)


# The queue on its own

def test_a_proposed_task_waits_for_the_person_and_no_app_can_take_it_before(folder):
    task = change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "Ficha del premio",
                                                             "Prepara la ficha del premio de Acme.", "Acme", "Grok"))
    assert task["state"] == "proposed" and task["by"] == "ChatGPT" and task["for"] == "Grok"
    with pytest.raises(remote_tasks.Refused, match="only a task the user approved"):
        change(folder, lambda tasks: remote_tasks.take(tasks, "Grok", task["id"]))
    assert not hasattr(remote_tasks, "approve_from_app")  # approving is only the person's, on their page


def test_once_approved_only_the_app_it_is_for_takes_it_and_hands_it_in(folder):
    task = change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "Ficha", "Haz la ficha.", "Acme", "Grok"))
    assert change(folder, lambda tasks: remote_tasks.approve(tasks, [task["id"]], {})) == 1
    with pytest.raises(remote_tasks.Refused, match="is for Grok, not for this app"):
        change(folder, lambda tasks: remote_tasks.take(tasks, "ChatGPT", task["id"]))
    taken = change(folder, lambda tasks: remote_tasks.take(tasks, "grok", task["id"]))  # the name, whatever its case
    assert taken["state"] == "in_progress" and taken["taken_by"] == "grok"
    with pytest.raises(remote_tasks.Refused, match="not in progress by this app"):
        change(folder, lambda tasks: remote_tasks.finish(tasks, "ChatGPT", task["id"], "Hecho"))
    done = change(folder, lambda tasks: remote_tasks.finish(tasks, "grok", task["id"], "Ficha en Entregables/acme.md"))
    assert done["state"] == "done" and [step["state"] for step in done["history"]] == [
        "proposed", "approved", "in_progress", "done"]
    assert [step["by"] for step in done["history"]] == ["ChatGPT", BY_THE_USER, "grok", "grok"]


def test_the_person_chooses_who_does_it_and_a_task_left_open_is_for_any_app(folder):
    task = change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "Buscar fotos", "Busca fotos del acto."))
    change(folder, lambda tasks: remote_tasks.approve(tasks, [task["id"]], {task["id"]: remote_tasks.ANY_APP}))
    assert change(folder, lambda tasks: remote_tasks.take(tasks, "Claude", task["id"]))["taken_by"] == "Claude"
    mine = change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "Llamar a Marta", ""))
    change(folder, lambda tasks: remote_tasks.approve(tasks, [mine["id"]], {mine["id"]: BY_THE_USER}))
    with pytest.raises(remote_tasks.Refused, match="for the user to do themselves"):
        change(folder, lambda tasks: remote_tasks.take(tasks, "Grok", mine["id"]))


def test_an_app_gives_back_what_it_cannot_do_and_the_person_returns_what_is_not_right(folder):
    task = change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "Ficha", "Haz la ficha.", None, "Grok"))
    change(folder, lambda tasks: remote_tasks.approve(tasks, [task["id"]], {}))
    back = change(folder, lambda tasks: remote_tasks.give_back(tasks, "Grok", task["id"], "Falta el acta del jurado."))
    assert back["state"] == "returned" and back["note"] == "Falta el acta del jurado."
    change(folder, lambda tasks: remote_tasks.approve(tasks, [task["id"]], {}))  # approved again, with what was missing
    change(folder, lambda tasks: remote_tasks.take(tasks, "Grok", task["id"]))
    change(folder, lambda tasks: remote_tasks.finish(tasks, "Grok", task["id"], "Ficha hecha."))
    change(folder, lambda tasks: remote_tasks.reopen(tasks, task["id"], "Cambia el titular."))
    again = remote_tasks.read(folder, KEY, "ana")[0]
    assert again["state"] == "returned" and again["note"] == "Cambia el titular." and again["taken_by"] == ""
    with pytest.raises(remote_tasks.Refused):
        change(folder, lambda tasks: remote_tasks.close(tasks, task["id"]))  # only what was handed in is closed


def test_secrets_never_go_in_the_queue_and_a_refused_change_writes_nothing(folder):
    with pytest.raises(remote_tasks.Refused, match="secret"):
        change(folder, lambda tasks: remote_tasks.propose(tasks, "Grok", "Entrar en la web",
                                                          "Entra en la web; la contraseña es limonero77"))
    assert remote_tasks.read(folder, KEY, "ana") == []
    task = change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "Ficha", "Haz la ficha."))
    with pytest.raises(remote_tasks.Refused, match="a task needs a title"):
        change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "  ", ""))
    assert [t["id"] for t in remote_tasks.read(folder, KEY, "ana")] == [task["id"]]


def test_the_queue_is_sealed_with_the_memorys_key(folder):
    change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "Ficha del premio de Acme", "Haz la ficha."))
    sealed = (folder / remote_tasks.FILE).read_bytes()
    assert sealed.startswith(remote_tasks.MAGIC) and b"Acme" not in sealed and b"ChatGPT" not in sealed
    with pytest.raises(ValueError):
        remote_tasks.read(folder, bytes(32), "ana")
    assert MEMBER.fullmatch("accounts/ana/tasks.sealed")  # it goes in the backups with the memory


def test_closed_tasks_age_out_but_open_ones_stay(folder, monkeypatch):
    monkeypatch.setattr(remote_tasks, "KEEP_CLOSED", 3)
    for number in range(6):
        task = change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", f"Tarea {number}", ""))
        if number < 5:
            change(folder, lambda tasks: remote_tasks.discard(tasks, [task["id"]]))
    left = remote_tasks.read(folder, KEY, "ana")
    assert sum(t["state"] == "closed" for t in left) == 3 and [t["title"] for t in left if t["state"] != "closed"] == [
        "Tarea 5"]


def test_how_an_agent_reads_the_queue(folder):
    task = change(folder, lambda tasks: remote_tasks.propose(tasks, "ChatGPT", "Ficha", "Haz la ficha.", "Acme", "Grok"))
    listed = remote_tasks.listing(remote_tasks.read(folder, KEY, "ana"))
    assert listed.startswith("1 tasks, oldest first.") and "count only once the user approved it" in listed
    assert f'[{task["id"]}] (proposed, for: Grok, Acme, proposed by: ChatGPT on ' in listed
    assert "    Instructions: Haz la ficha." in listed
    assert remote_tasks.listing([], "done") == "No tasks done."
    assert remote_tasks.listing([], "nope").startswith("Not done: state must be one of")
    summary = remote_tasks.summary(remote_tasks.read(folder, KEY, "ana"), "https://memoria.example")
    assert summary.startswith("Work queue: 1 proposed.") and "https://memoria.example/memory?view=tasks" in summary
    assert "  To approve: [" in summary and "no AI can approve" in summary


# What each level lets an app do

@pytest.mark.parametrize("level, tool, allowed", [
    ("all", "forget", True), ("all", "vault_reveal", True),
    ("add", "remember", True), ("add", "task_done", True), ("add", "briefing", True),
    ("add", "forget", False), ("add", "memory_page", False), ("add", "vault_list", False),
    ("read", "recall", True), ("read", "task_list", True), ("read", "photo_show", True),
    ("read", "remember", False), ("read", "task_propose", False), ("read", "photo_save", False),
])
def test_each_level_lets_an_app_do_its_job_and_no_more(level, tool, allowed):
    assert remote_apps.allowed(level, tool) is allowed
    refusal = remote_apps.refusal(level, tool, "https://memoria.example")
    assert (refusal is None) is allowed
    if refusal:
        assert refusal.startswith("Not allowed") and "https://memoria.example/memory?view=apps" in refusal
        assert "Do not ask another app" in refusal


def test_an_accounts_levels_are_its_own_and_a_new_app_can_do_everything(tmp_path):
    pytest.importorskip("cryptography")
    from src.infinito3.remote_accounts import Accounts

    accounts = Accounts(tmp_path)
    accounts.create("ana", "una contraseña larga")
    assert accounts.app_level("ana", "client-1") == remote_apps.DEFAULT == "all"
    assert accounts.set_app_level("ana", "client-1", "read")
    assert accounts.app_level("ana", "client-1") == "read" and accounts.app_level("ana", "client-2") == "all"
    with pytest.raises(ValueError):
        accounts.set_app_level("ana", "client-1", "admin")
    accounts.set_app_level("ana", "client-1", None)
    assert accounts.app_levels("ana") == {} and not accounts.set_app_level("nadie", "client-1", "read")


# Over the server and on the page

httpx2 = pytest.importorskip("httpx2")
from tests.test_remote_accounts_v3 import Storage, app_session, server  # noqa: E402,F401
from tests.test_remote_memory_page_v3 import browser, csrf_of, enter  # noqa: E402


def test_a_task_goes_from_one_app_to_the_person_and_back_and_only_the_page_approves_it(server):  # noqa: F811
    base, _, _ = server
    steps = [("task_propose", {"title": "Ficha del premio", "instructions": "Prepara la ficha del premio de Acme.",
                               "project": "Acme"})]
    (proposed,), _, _ = app_session(base, steps, mode="signup", name="rosa")
    assert proposed.startswith("Proposed as task ") and f"{base}/memory?view=tasks" in proposed
    task_id = re.search(r"task ([0-9a-f]{8})", proposed).group(1)
    (refused, listed), _, _ = app_session(base, [("task_take", {"task_id": task_id}), ("task_list", {})], name="rosa")
    assert "only a task the user approved" in refused and "(proposed, for: any app, Acme, proposed by: Prueba" in listed

    with browser() as client:
        assert "Ficha del premio" not in client.get(f"{base}/memory", params={"view": "tasks"}).text  # no session
        enter(client, base, "rosa")
        page = client.get(f"{base}/memory", params={"view": "tasks"})
        assert "Por aprobar" in page.text and "Ficha del premio" in page.text and 'class="badge">1<' in page.text
        approved = client.post(f"{base}/memory", data={"action": "task_approve", "csrf": csrf_of(page), "id": task_id,
                                                       f"for:{task_id}": "Prueba"})
        assert "Aprobadas: 1." in approved.text and "En marcha" in approved.text

    steps = [("task_take", {"task_id": task_id}), ("task_done", {"task_id": task_id, "result": "Ficha en acme.md"}),
             ("briefing", {})]
    (taken, done, briefing), _, _ = app_session(base, steps, name="rosa")
    assert taken.startswith(f"Task {task_id} is yours now") and "Prepara la ficha" in taken
    assert "waits for the user's review" in done and "Work queue: 1 done." in briefing and "Ficha en acme.md" in briefing

    with browser() as client:
        enter(client, base, "rosa")
        page = client.get(f"{base}/memory", params={"view": "tasks"})
        assert "Para revisar" in page.text and "Ficha en acme.md" in page.text
        closed = client.post(f"{base}/memory", data={"action": "task_close", "csrf": csrf_of(page), "id": task_id})
        assert "Cerrada." in closed.text and "Cerradas hace poco" in closed.text


def test_the_person_sets_what_each_app_may_do_and_can_disconnect_it(server):  # noqa: F811
    base, _, _ = server
    bot = Storage()
    app_session(base, [("remember", {"statement": "Vivo en Málaga."})], mode="signup", name="sara", storage=bot)
    with browser() as client:
        enter(client, base, "sara")
        page = client.get(f"{base}/memory", params={"view": "apps"})
        assert "Tus apps" in page.text and "Prueba" in page.text and "Leer y añadir" in page.text
        client_id = re.search(r'name="client" value="([^"]+)"', page.text).group(1)
        saved = client.post(f"{base}/memory", data={"action": "app_level", "csrf": csrf_of(page), "client": client_id,
                                                    "level": "read"})
        assert "Guardado." in saved.text

    steps = [("remember", {"statement": "Tengo una perra."}), ("recall", {"query": "Málaga"}),
             ("task_propose", {"title": "Ficha", "instructions": ""})]
    (remember, recall, propose), _, _ = app_session(base, steps, name="sara", storage=bot)
    assert remember.startswith("Not allowed: the user lets this app only read") and "Málaga" in recall
    assert propose.startswith("Not allowed")

    with browser() as client:
        enter(client, base, "sara")
        page = client.get(f"{base}/memory", params={"view": "apps"})
        client.post(f"{base}/memory", data={"action": "app_level", "csrf": csrf_of(page), "client": client_id,
                                           "level": "add"})
    (listed,), _, _ = app_session(base, [("list_memories", {})], name="sara", storage=bot)
    memory_id = re.search(r"\[([0-9a-f-]{36})\]", listed).group(1)
    steps = [("remember", {"statement": "Tengo una perra."}), ("forget", {"memory_ids": [memory_id]})]
    (remember, forget), _, _ = app_session(base, steps, name="sara", storage=bot)
    assert remember.startswith("Remembered") and forget.startswith("Not allowed: the user lets this app read and add")
    other = app_session(base, [("forget", {"memory_ids": [memory_id]})], name="sara")[0][0]
    assert other.startswith("Deleted 1")  # another app of the same person keeps its own level: everything

    with browser() as client:
        enter(client, base, "sara")
        page = client.get(f"{base}/memory", params={"view": "apps"})
        gone = client.post(f"{base}/memory", data={"action": "app_level", "csrf": csrf_of(page), "client": client_id,
                                                  "level": "add", "what": "disconnect"})
        assert "Desconectada." in gone.text and client_id not in gone.text


def test_a_page_form_for_an_app_that_is_not_yours_changes_nothing(server):  # noqa: F811
    base, people, folder = server
    app_session(base, [("list_memories", {})], mode="signup", name="tere")
    with browser() as client:
        enter(client, base, "tere")
        page = client.get(f"{base}/memory", params={"view": "apps"})
        client.post(f"{base}/memory", data={"action": "app_level", "csrf": csrf_of(page), "client": "de-otra-persona",
                                           "level": "read"})
    from src.infinito3.remote_accounts import Accounts

    assert Accounts(folder).app_levels("tere") == {}
