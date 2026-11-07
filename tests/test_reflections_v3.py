"""Reflective memory: the agent's own opinions of what it lived with the person, kept as written, never facts or
guidelines, and gone when what they rest on is forgotten."""
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from src.infinito3 import remote_apps, remote_reflections as reflections
from src.infinito3.remote_backup import MEMBER

KEY = bytes(range(32))
KNOWN = ["m1", "m2", "m3"]


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "accounts" / "ana"


def change(folder, action):
    return reflections.change(folder, KEY, "ana", action)


def first(data):
    return reflections.add(data, "ChatGPT", "Creo que dejamos F11 demasiado pronto.", ["m1"], "decision", KNOWN,
                           project="INFINITO", advice="Antes de cerrar un experimento, mirar el coste de seguir.",
                           questions=["¿Habría cambiado algo con otra réplica?"], model="GPT-5")


# Keeping one

def test_a_reflection_rests_on_memories_and_is_kept_as_written(folder):
    kept = change(folder, first)
    assert kept["by"] == "ChatGPT" and kept["model"] == "GPT-5" and kept["sources"] == ["m1"]
    data = reflections.read(folder, KEY, "ana")
    assert data["reflections"][0]["text"] == "Creo que dejamos F11 demasiado pronto."
    assert reflections.status(data, data["reflections"][0]) == "historical"


@pytest.mark.parametrize("sources, scope, relates, refused", [
    ([], "decision", [], "cite in sources"),
    (["nope"], "decision", [], "these are not memories: nope"),
    (["m1"], "mood", [], "scope must be one of"),
    (["m1"], "decision", [{"id": "zzzz", "how": "reconsiders"}], "there is no reflection"),
])
def test_a_reflection_without_evidence_or_with_a_wrong_link_is_not_kept(folder, sources, scope, relates, refused):
    with pytest.raises(reflections.Refused, match=refused):
        change(folder, lambda data: reflections.add(data, "Claude", "Opino algo.", sources, scope, KNOWN,
                                                    relates=relates))
    assert reflections.read(folder, KEY, "ana")["reflections"] == []


def test_the_same_reflection_twice_or_a_secret_is_not_kept(folder):
    change(folder, first)
    with pytest.raises(reflections.Refused, match="word for word"):
        change(folder, first)
    with pytest.raises(reflections.Refused, match="secret"):
        change(folder, lambda data: reflections.add(data, "Claude", "Su contraseña es limonero77, creo.", ["m2"],
                                                    "user", KNOWN))


def test_a_later_reflection_reconsiders_an_earlier_one_and_both_stay(folder):
    one = change(folder, first)
    two = change(folder, lambda data: reflections.add(
        data, "Claude", "Ahora creo que cerrar F11 fue lo correcto: no había señal.", ["m2"], "decision", KNOWN,
        relates=[{"id": one["id"], "how": "reconsiders"}]))
    data = reflections.read(folder, KEY, "ana")
    assert [r["text"] for r in data["reflections"]] == ["Creo que dejamos F11 demasiado pronto.",
                                                        "Ahora creo que cerrar F11 fue lo correcto: no había señal."]
    assert reflections.status(data, data["reflections"][0]) == "reconsidered"
    told = reflections.history(data, one["id"])
    assert told.index(one["id"]) < told.index(two["id"]) and f"reconsiders {one['id']}" in told
    assert reflections.history(data, two["id"]) == told  # the same story, from either end


def test_forgetting_a_memory_takes_every_reflection_resting_on_it(folder):
    change(folder, first)
    change(folder, lambda data: reflections.add(data, "Claude", "Hoy fue un buen día de trabajo.", ["m2", "m3"],
                                                "relationship", KNOWN))
    assert change(folder, lambda data: reflections.drop_citing(data, ["m3"])) == 1
    assert [r["sources"] for r in reflections.read(folder, KEY, "ana")["reflections"]] == [["m1"]]


def test_the_person_comments_hides_deletes_and_asks(folder):
    one = change(folder, first)
    change(folder, lambda data: reflections.comment(data, one["id"], "F11 no se cerró: quedó aparcado."))
    change(folder, lambda data: reflections.hide(data, one["id"], True))
    data = reflections.read(folder, KEY, "ana")
    assert data["reflections"][0]["notes"][0]["text"] == "F11 no se cerró: quedó aparcado."
    assert data["reflections"][0]["text"] == "Creo que dejamos F11 demasiado pronto."  # the opinion stays as it was
    assert reflections.listing(data) == "No reflections yet."  # hidden from the AIs
    change(folder, reflections.request)
    assert reflections.read(folder, KEY, "ana")["requested"]
    change(folder, lambda data: reflections.add(data, "Claude", "Otra cosa que pienso.", ["m2"], "self", KNOWN))
    assert reflections.read(folder, KEY, "ana")["requested"] == ""  # a reflection answers the request
    change(folder, lambda data: reflections.delete(data, one["id"]))
    assert [r["id"] for r in reflections.read(folder, KEY, "ana")["reflections"]] != [one["id"]]


def test_they_are_sealed_and_go_in_the_backups(folder):
    change(folder, first)
    sealed = (folder / reflections.FILE).read_bytes()
    assert sealed.startswith(reflections.MAGIC) and b"F11" not in sealed
    with pytest.raises(ValueError):
        reflections.read(folder, bytes(32), "ana")
    assert MEMBER.fullmatch("accounts/ana/reflections.sealed")


# When it is due, and what the agent reads

def memory(identifier, kind="", days_ago=0):
    created = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days_ago)
    return SimpleNamespace(id=identifier, content=f"Recuerdo {identifier}", created_at=created,
                           metadata={"kind": kind} if kind else {})


def test_the_server_says_when_a_reflection_is_due(folder):
    few = [memory(f"m{n}") for n in range(3)]
    assert reflections.due(reflections.read(folder, KEY, "ana"), few).startswith("No reflection is due")
    many = [memory(f"m{n}") for n in range(6)]
    assert "there is no reflection yet" in reflections.due(reflections.read(folder, KEY, "ana"), many)
    change(folder, reflections.request)
    assert "the user asked for a reflection" in reflections.due(reflections.read(folder, KEY, "ana"), few)


def test_a_decision_since_the_last_reflection_makes_one_due(folder):
    change(folder, first)
    data = reflections.read(folder, KEY, "ana")
    later = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=5)
    new = [SimpleNamespace(id=f"n{n}", content="x", created_at=later, metadata={"kind": k})
           for n, k in enumerate(["decision", "fact", "fact"])]
    assert "1 decision or agreement among 3 new memories" in reflections.due(data, new)
    assert reflections.due(data, new[:2]).startswith("No reflection is due")


def test_what_to_reflect_on_is_whats_new_and_the_earlier_reflections(folder):
    from src.infinito3.types import MemoryKind, MemoryRecord

    change(folder, first)
    data = reflections.read(folder, KEY, "ana")
    later = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=5)
    new = [MemoryRecord(content="Retomamos F11 con otra semilla.", kind=MemoryKind.SEMANTIC, importance=.6,
                        created_at=later, metadata={"kind": "decision", "project": "INFINITO"})]
    told = reflections.context(data, new)
    assert "Retomamos F11 con otra semilla." in told and "Creo que dejamos F11 demasiado pronto." in told
    assert told.rstrip().endswith(reflections.HOW_TO) and "relate to them rather than repeat them" in told


def test_reading_reflections_needs_only_read_and_writing_one_needs_add():
    for tool in ("reflection_due", "reflection_context", "reflection_list", "reflection_history"):
        assert remote_apps.allowed("read", tool)
    assert remote_apps.allowed("add", "reflection_add") and not remote_apps.allowed("read", "reflection_add")


# The MVP of the specification, over the server and on the page

httpx2 = pytest.importorskip("httpx2")
from tests.test_remote_accounts_v3 import Storage, app_session, server  # noqa: E402,F401
from tests.test_remote_memory_page_v3 import browser, csrf_of, enter  # noqa: E402


def ids_of(listed):
    return re.findall(r"\[([0-9a-f-]{36})\]", listed)


def test_a_reflection_is_written_in_one_app_read_in_another_reconsidered_and_shown_in_order(server):  # noqa: F811
    base, _, _ = server
    steps = [("remember", {"statement": "Decidimos aparcar F11 por falta de señal.", "kind": "decision",
                           "project": "INFINITO"}),
             ("remember", {"statement": "Prefiero iniciativa con riesgo de error a exceso de prudencia."}),
             ("list_memories", {})]
    (_, _, listed), _, _ = app_session(base, steps, mode="signup", name="hugo")
    f11, initiative = ids_of(listed)
    steps = [("reflection_context", {}),
             ("reflection_add", {"reflection": "Creo que aparcar F11 fue prudente, quizá demasiado.",
                                 "sources": [f11], "scope": "decision", "project": "INFINITO",
                                 "advice": "Proponer una réplica barata antes de aparcar.",
                                 "questions": ["¿Faltaba señal o faltaba tiempo?"], "model": "GPT-5"}),
             ("recall", {"query": "F11"})]
    (context, added, recalled), _, _ = app_session(base, steps, name="hugo")
    assert "Decidimos aparcar F11" in context and reflections.HOW_TO in context
    assert added.startswith("Kept as your reflection ") and "not a guideline" in added
    first_id = re.search(r"reflection ([0-9a-f]{8})", added).group(1)
    assert "quizá demasiado" not in recalled  # an opinion is not a memory, nor a guideline
    assert "The user's own guidelines" not in recalled

    steps = [("reflection_list", {}),
             ("reflection_add", {"reflection": "Ya no lo veo así: aparcar F11 encaja con cómo prefiere trabajar.",
                                 "sources": [f11, initiative], "scope": "decision",
                                 "relates_to": [{"id": first_id, "how": "reconsiders"}]}),
             ("reflection_history", {"reflection_id": first_id})]
    (listed, second, story), _, _ = app_session(base, steps, name="hugo")  # another app, another sign-in
    assert "quizá demasiado" in listed and "(GPT-5)" in listed and reflections.HEADER in listed
    second_id = re.search(r"reflection ([0-9a-f]{8})", second).group(1)
    assert story.index(first_id) < story.index(second_id) and f"reconsiders {first_id}" in story

    with browser() as client:
        enter(client, base, "hugo")
        page = client.get(f"{base}/memory", params={"view": "reflections"})
        text = page.text
        assert text.index("quizá demasiado") < text.index("Ya no lo veo así")  # in order
        assert "reconsiderada después" in text and "Reconsidera" in text and "Proponer una réplica barata" in text
        assert "Decidimos aparcar F11" in text  # what it rests on
        commented = client.post(f"{base}/memory", data={"action": "reflection_comment", "csrf": csrf_of(page),
                                                       "id": first_id, "text": "F11 quedó aparcado, no cerrado."})
        assert "Comentario guardado." in commented.text and "quizá demasiado" in commented.text
        deleted = client.post(f"{base}/memory", data={"action": "reflection_delete", "csrf": csrf_of(commented),
                                                     "id": second_id})
        assert "Borrada." in deleted.text and "Ya no lo veo así" not in deleted.text

    (answer, listed), _, _ = app_session(base, [("forget", {"memory_ids": [f11]}), ("reflection_list", {})], name="hugo")
    assert "Deleted 1 of your reflections that rested on them." in answer and listed == "No reflections yet."


def test_an_app_that_may_only_read_cannot_write_a_reflection(server):  # noqa: F811
    base, _, _ = server
    reader = Storage()
    app_session(base, [("remember", {"statement": "Vivo en Málaga."})], mode="signup", name="ines", storage=reader)
    with browser() as client:
        enter(client, base, "ines")
        page = client.get(f"{base}/memory", params={"view": "apps"})
        client_id = re.search(r'name="client" value="([^"]+)"', page.text).group(1)
        client.post(f"{base}/memory", data={"action": "app_level", "csrf": csrf_of(page), "client": client_id,
                                           "level": "read"})
    steps = [("list_memories", {}), ("reflection_due", {})]
    (listed, due), _, _ = app_session(base, steps, name="ines", storage=reader)
    (refused,), _, _ = app_session(base, [("reflection_add", {"reflection": "Opino.", "sources": ids_of(listed),
                                                              "scope": "user"})], name="ines", storage=reader)
    assert due.startswith("No reflection is due") and refused.startswith("Not allowed")
