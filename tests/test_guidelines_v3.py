"""The person's own guidelines for all their AIs: written on their page, first in what every app reads."""
import io
import json
import re
from types import SimpleNamespace

import pytest

from src.infinito3 import remote_apps, remote_client, remote_guidelines as guidelines
from src.infinito3.remote_backup import MEMBER

KEY = bytes(range(32))


@pytest.fixture
def folder(tmp_path):
    return tmp_path / "accounts" / "ana"


def change(folder, action):
    return guidelines.change(folder, KEY, "ana", action)


def test_what_the_person_types_becomes_one_guideline_per_line(folder):
    change(folder, lambda data: guidelines.replace(data, "- Respóndeme en español.\n\n2) Firma con Hilo.\n"
                                                         "respondeme en espanol\n   Sé breve.  "))
    assert guidelines.read(folder, KEY, "ana")["items"] == ["Respóndeme en español.", "Firma con Hilo.", "Sé breve."]
    change(folder, lambda data: guidelines.replace(data, ""))
    assert guidelines.read(folder, KEY, "ana")["items"] == []


@pytest.mark.parametrize("text, reason", [
    ("\n".join(f"Pauta {n}" for n in range(guidelines.MAX_ITEMS + 1)), "too_many"),
    ("x" * (guidelines.MAX_CHARS + 1), "too_long"),
    ("Firma con Hilo.\nLa contraseña es limonero77", "secret"),
])
def test_too_many_too_long_or_a_secret_changes_nothing(folder, text, reason):
    change(folder, lambda data: guidelines.replace(data, "Firma con Hilo."))
    with pytest.raises(guidelines.Refused, match=reason):
        change(folder, lambda data: guidelines.replace(data, text))
    assert guidelines.read(folder, KEY, "ana")["items"] == ["Firma con Hilo."]


def test_an_app_only_proposes_and_the_person_accepts_or_discards(folder):
    assert change(folder, lambda data: guidelines.propose(data, "ChatGPT", "Respóndeme en español.")) == "proposed"
    assert change(folder, lambda data: guidelines.propose(data, "Grok", "respondeme en espanol")) == "waiting"
    data = guidelines.read(folder, KEY, "ana")
    assert data["items"] == [] and data["proposed"][0]["by"] == "ChatGPT"
    change(folder, lambda data: guidelines.accept(data, data["proposed"][0]["id"]))
    assert guidelines.read(folder, KEY, "ana") == {**guidelines.read(folder, KEY, "ana"), "items": [
        "Respóndeme en español."], "proposed": []}
    assert change(folder, lambda data: guidelines.propose(data, "Grok", "Respóndeme en español.")) == "already"
    change(folder, lambda data: guidelines.propose(data, "Grok", "Usa emojis."))
    proposal = guidelines.read(folder, KEY, "ana")["proposed"][0]["id"]
    change(folder, lambda data: guidelines.discard(data, proposal))
    with pytest.raises(guidelines.Refused, match="gone"):
        change(folder, lambda data: guidelines.discard(data, proposal))
    with pytest.raises(guidelines.Refused, match="secret"):
        change(folder, lambda data: guidelines.propose(data, "Grok", "La contraseña es limonero77"))


def test_what_every_app_reads_first(folder):
    assert guidelines.block(guidelines.read(folder, KEY, "ana"), "https://m.example") == ""
    change(folder, lambda data: guidelines.replace(data, "Respóndeme en español.\nFirma con Hilo."))
    change(folder, lambda data: guidelines.propose(data, "Grok", "Sé breve."))
    written = guidelines.block(guidelines.read(folder, KEY, "ana"), "https://m.example")
    assert written.splitlines() == [guidelines.HEADER, "- Respóndeme en español.", "- Firma con Hilo.",
                                    "1 proposed guideline waits for the user to accept at "
                                    "https://m.example/memory?view=guidelines; it is not a guideline yet."]


def test_they_are_sealed_with_the_memorys_key_and_go_in_the_backups(folder):
    change(folder, lambda data: guidelines.replace(data, "Firma con Hilo."))
    sealed = (folder / guidelines.FILE).read_bytes()
    assert sealed.startswith(guidelines.MAGIC) and b"Hilo" not in sealed
    with pytest.raises(ValueError):
        guidelines.read(folder, bytes(32), "ana")
    assert MEMBER.fullmatch("accounts/ana/guidelines.sealed")


def test_what_the_memory_already_holds_that_reads_like_a_guideline_is_offered():
    def record(text):
        return SimpleNamespace(id=text[:8], content=text)

    memories = [record("Prefiero las explicaciones precisas y en español."), record("Vivo en Málaga."),
                record("Cuando uses la memoria, firmarás con Hilo al final."),
                record("Quiero que INFINITO también pueda guardar fotos."),
                record("Quiero que en cada conversación con Grok lo primero que hagas sea consultar la memoria.")]
    offered = [r.content for r in guidelines.suggestions(memories, {"items": ["Cuando uses la memoria, firmarás con "
                                                                              "Hilo al final."]})]
    assert offered == ["Quiero que en cada conversación con Grok lo primero que hagas sea consultar la memoria.",
                       "Prefiero las explicaciones precisas y en español."]


def test_an_app_that_may_add_may_propose_one_and_one_that_only_reads_may_not():
    assert remote_apps.allowed("add", "guideline_propose") and not remote_apps.allowed("read", "guideline_propose")


def test_the_hook_brings_the_guidelines_under_their_own_heading(monkeypatch):
    answer = f"{guidelines.HEADER}\n- Firma con Hilo.\n\nMemories about the user, oldest first.\n[1] (2026-11-07) \"Vivo en Málaga.\""
    monkeypatch.setattr(remote_client, "recall", lambda *args, **kwargs: answer)
    out = io.StringIO()
    remote_client.context(io.BytesIO(json.dumps({"prompt": "hola"}).encode()), out)
    text = json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]
    assert text.startswith(f"{guidelines.HEADER}\n- Firma con Hilo.\n\n{remote_client.CONTEXT_HEADER}")
    assert text.count("Firma con Hilo") == 1 and "Vivo en Málaga" in text


# Over the server and on the page

httpx2 = pytest.importorskip("httpx2")
from tests.test_remote_accounts_v3 import Storage, app_session, server  # noqa: E402,F401
from tests.test_remote_memory_page_v3 import browser, csrf_of, enter  # noqa: E402


def test_written_on_the_page_they_reach_every_app_first(server):  # noqa: F811
    base, _, _ = server
    bot = Storage()
    steps = [("remember", {"statement": "Prefiero las explicaciones precisas y en español."}),
             ("recall", {"query": "explicaciones"})]
    (_, before), _, _ = app_session(base, steps, mode="signup", name="vera", storage=bot)
    assert not before.startswith(guidelines.HEADER)
    with browser() as client:
        enter(client, base, "vera")
        page = client.get(f"{base}/memory", params={"view": "guidelines"})
        assert "Tus pautas" in page.text and "Una pauta por línea" in page.text
        assert "Prefiero las explicaciones precisas y en español." in page.text  # offered from the memory
        saved = client.post(f"{base}/memory", data={"action": "guidelines_save", "csrf": csrf_of(page),
                                                    "text": "Firma con Hilo cuando uses mi memoria."})
        assert "Guardadas." in saved.text
        memory_id = re.search(r'name="id" value="([0-9a-f-]{36})"', saved.text).group(1)
        added = client.post(f"{base}/memory", data={"action": "guideline_from_memory", "csrf": csrf_of(saved),
                                                    "id": memory_id})
        assert "Añadida." in added.text

    steps = [("recall", {"query": "Málaga"}), ("briefing", {}), ("list_memories", {}), ("overview", {})]
    answers, _, _ = app_session(base, steps, name="vera", storage=bot)
    for answer in answers:
        assert answer.startswith(f"{guidelines.HEADER}\n- Firma con Hilo cuando uses mi memoria.\n"
                                 "- Prefiero las explicaciones precisas y en español.\n\n"), answer


def test_an_app_proposes_and_only_the_page_makes_it_a_guideline(server):  # noqa: F811
    base, _, _ = server
    steps = [("guideline_propose", {"guideline": "Respóndeme siempre en español."}), ("recall", {"query": "x"})]
    (proposed, recalled), _, _ = app_session(base, steps, mode="signup", name="wanda")
    assert proposed.startswith("Proposed.") and f"{base}/memory?view=guidelines" in proposed
    assert recalled.startswith("The user has no guidelines for their AIs yet.\n1 proposed guideline wait")
    with browser() as client:
        enter(client, base, "wanda")
        page = client.get(f"{base}/memory", params={"view": "guidelines"})
        assert "Propuestas por tus IAs" in page.text and "propuesta por Prueba" in page.text
        proposal = re.search(r'name="id" value="([0-9a-f]{8})"', page.text).group(1)
        accepted = client.post(f"{base}/memory", data={"action": "guideline_accept", "csrf": csrf_of(page),
                                                       "id": proposal})
        assert "Añadida." in accepted.text and "Propuestas por tus IAs" not in accepted.text
    (recalled,), _, _ = app_session(base, [("recall", {"query": "x"})], name="wanda")
    assert recalled.startswith(f"{guidelines.HEADER}\n- Respóndeme siempre en español.\n\n")


def test_a_secret_typed_on_the_page_is_not_kept_nor_sent_back(server):  # noqa: F811
    base, _, _ = server
    app_session(base, [("list_memories", {})], mode="signup", name="xenia")
    with browser() as client:
        enter(client, base, "xenia")
        page = client.get(f"{base}/memory", params={"view": "guidelines"})
        refused = client.post(f"{base}/memory", data={"action": "guidelines_save", "csrf": csrf_of(page),
                                                      "text": "La contraseña es limonero77"})
        assert refused.status_code == 400 and "parece una clave" in refused.text and "limonero77" not in refused.text


# What every app is told, and the page that shows it

def test_every_app_is_told_how_to_use_a_memory_in_under_two_kilobytes():
    from src.infinito3.mcp_server import INSTRUCTIONS

    assert len(INSTRUCTIONS) <= 2048
    for rule in ("again when the topic changes", "never invent memories, results or dates",
                 "do not make the user repeat it", "Never present a proposal as a decision",
                 "if sources differ, show both", "never restore it from old summaries",
                 "unless a tool confirmed it", "only the user's own guidelines, first in recall",
                 # The memory is the agent's own, and it speaks from it without narrating its tools.
                 "INFINITO is your long-term memory", "Speak from it as your own memory",
                 "without narrating tools or lookups unless one failed", "Earlier work is yours even if another app did it"):
        assert rule in INSTRUCTIONS, rule


def test_recall_no_longer_says_the_latest_always_wins(tmp_path):
    from src.infinito3.mcp_server import AgentMemory

    memory = AgentMemory(str(tmp_path / "memory.db"))
    memory.remember("Vivo en Málaga.")
    assert "the later is usually current; if they come from different sources, say both" in memory.recall("Málaga")
    memory.close()


def test_a_server_with_sign_in_tells_apps_exactly_what_the_page_shows():
    from src.infinito3 import remote_memory, remote_tasks
    from src.infinito3.mcp_server import INSTRUCTIONS

    told = remote_memory.server_instructions()
    assert told.startswith(INSTRUCTIONS) and guidelines.INSTRUCTIONS in told and remote_tasks.INSTRUCTIONS in told
    assert remote_memory.INSTRUCTIONS in told


def test_the_page_shows_what_the_ais_receive(server):  # noqa: F811
    from src.infinito3 import remote_memory

    base, _, _ = server
    app_session(base, [("list_memories", {})], mode="signup", name="zoe")
    with browser() as client:
        enter(client, base, "zoe")
        page = client.get(f"{base}/memory", params={"view": "guidelines"})
        assert "Vista previa: lo que reciben tus IAs" in page.text and "Aún no tienes pautas" in page.text
        assert "iguales para todas las personas" in page.text
        assert "again when the topic changes" in page.text and "steer how you behave." in page.text
        saved = client.post(f"{base}/memory", data={"action": "guidelines_save", "csrf": csrf_of(page),
                                                    "text": "Firma con Hilo."})
        assert f"{guidelines.HEADER}\n- Firma con Hilo.</pre>" in saved.text.replace("&#x27;", "'")
    assert remote_memory.server_instructions().count("\n- ") >= 12
