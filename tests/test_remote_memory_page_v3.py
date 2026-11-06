"""A person's page for their memory: nothing opens without the password, and it can do what an app refuses to."""
import re

import pytest

from src.infinito3.remote_accounts import Grants
from src.infinito3.remote_memory import MAX_IDS, PAGE_SIZE, TOOL_HINTS, set_session_cookie
from tests.test_remote_accounts_v3 import PASSWORD, app_session, server  # noqa: F401  # server is a fixture

httpx2 = pytest.importorskip("httpx2")

SPANISH = {"accept-language": "es-ES"}
SECRET = "Mi contraseña es limonero-secreto-77"


def browser():
    return httpx2.Client(headers=SPANISH, follow_redirects=True)


def enter(client, base, name, *, password=PASSWORD, link=""):
    return client.post(f"{base}/memory", data={"action": "signin", "name": name, "password": password, "r": link})


def csrf_of(page):
    return re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)


def ids_in(answer):
    return re.findall(r"\[([0-9a-f-]{36})\]", answer)


def link_in(answer):
    return re.search(r"http\S+/memory\?r=(\S+)", answer)


def memory_of(base, name, statements):
    """A new account with these memories; returns what list_memories says, with their ids."""
    steps = [("remember", {"statement": s}) for s in statements] + [("list_memories", {})]
    answers, _, _ = app_session(base, steps, mode="signup", name=name)
    assert all(answer.startswith("Remembered") for answer in answers[:-1]), answers
    return answers[-1]


def test_a_link_alone_shows_nothing_and_the_page_always_asks_for_the_password(server):  # noqa: F811
    base, _, _ = server
    listed = memory_of(base, "lena", ["Mi dentista se llama Marta."])
    (answer,), _, _ = app_session(base, [("memory_page", {"memory_ids": ids_in(listed)})], name="lena")
    link = link_in(answer).group(0)
    assert "Nothing is deleted until" in answer and "Marta" not in answer

    opened = httpx2.get(link, headers=SPANISH)  # as a chat app that fetches the link first would
    assert opened.status_code == 200 and "Marta" not in opened.text
    assert 'type="password"' in opened.text and "<script" not in opened.text
    assert "form-action 'self'" in opened.headers["content-security-policy"]
    assert "Marta" not in httpx2.get(f"{base}/memory", headers=SPANISH).text


def test_each_person_sees_only_their_memory_and_a_wrong_password_opens_nothing(server):  # noqa: F811
    base, _, _ = server
    memory_of(base, "nuria", ["Mi dentista se llama Marta."])
    memory_of(base, "oscar", ["Mi fontanero se llama Paco."])
    with browser() as client:
        refused = enter(client, base, "nuria", password="esta no es la contraseña")
        assert refused.status_code == 400 and "no abren ninguna memoria" in refused.text and "Marta" not in refused.text
        assert "Marta" not in client.get(f"{base}/memory").text
    with browser() as client:
        page = enter(client, base, "nuria")
        assert page.status_code == 200 and "Marta" in page.text and "Paco" not in page.text and "1 recuerdo<" in page.text


def test_what_an_app_cannot_forget_is_confirmed_on_the_page_and_only_what_stays_marked_goes(server):  # noqa: F811
    base, _, _ = server
    listed = memory_of(base, "pau", ["Mi dentista se llama Marta.", "Mi fontanero se llama Paco.",
                                     "Mi color favorito es el verde."])
    dentist, plumber, colour = ids_in(listed)
    (answer,), _, _ = app_session(base, [("memory_page", {"memory_ids": [dentist, plumber]})], name="pau")
    token = link_in(answer).group(1)
    assert "Marta" in app_session(base, [("list_memories", {})], name="pau")[0][0]  # the tool deletes nothing

    with browser() as client:
        confirm = enter(client, base, "pau", link=token)
        assert "Se borrarán 2 recuerdos" in confirm.text and "Marta" in confirm.text and "Paco" in confirm.text
        assert "verde" not in confirm.text and "14 días" in confirm.text
        assert confirm.text.count(" checked") == 2
        done = client.post(f"{base}/memory", data={"action": "delete", "csrf": csrf_of(confirm), "r": token,
                                                   "id": [dentist]})  # the person unmarked the plumber
        assert done.status_code == 200 and "Se ha borrado 1 recuerdo." in done.text and "volver al chat" in done.text
        again = client.get(f"{base}/memory", params={"r": token})
        assert again.status_code == 400 and "caducado" in again.text  # used up only when the page is sent

    (left,), _, _ = app_session(base, [("list_memories", {})], name="pau")
    assert "Marta" not in left and "Paco" in left and colour in left


def test_the_tool_asks_for_known_ids_and_without_ids_gives_the_page(server):  # noqa: F811
    base, _, _ = server
    memory_of(base, "quim", ["Mi dentista se llama Marta."])
    (unknown, whole, many), _, storage = app_session(
        base, [("memory_page", {"memory_ids": ["no-existe"]}), ("memory_page", {}),
               ("memory_page", {"memory_ids": [str(i) for i in range(MAX_IDS + 1)]})], name="quim")
    assert "unknown ids: no-existe" in unknown and "/memory" not in unknown
    assert whole.split("\n")[0].endswith(f"{base}/memory") and "?r=" not in whole
    assert "Not done" in many
    tool = next(t for t in storage.tools if t.name == "memory_page")
    assert set(tool.input_schema["properties"]) == {"memory_ids"} and "forget" in tool.description
    assert TOOL_HINTS["readOnlyHint"] is True and tool.annotations.read_only_hint is True


def test_a_private_address_has_no_page(server):  # noqa: F811
    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    base, people, _ = server
    token = people.add("eva")

    async def run():
        async with streamable_http_client(f"{base}/u/{token}/mcp") as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                return (await client.call_tool("memory_page", {})).content[0].text

    assert "needs an INFINITO account" in anyio.run(run)


def test_someone_elses_link_shows_nothing_of_theirs_and_deletes_nothing(server):  # noqa: F811
    base, _, _ = server
    listed = memory_of(base, "rita", ["Mi dentista se llama Marta."])
    memory_of(base, "saul", ["Mi fontanero se llama Paco."])
    (answer,), _, _ = app_session(base, [("memory_page", {"memory_ids": ids_in(listed)})], name="rita")
    token = link_in(answer).group(1)
    with browser() as client:
        page = enter(client, base, "saul", link=token)
        assert page.status_code == 403 and "otra memoria" in page.text and "Marta" not in page.text
        own = client.get(f"{base}/memory")
        done = client.post(f"{base}/memory", data={"action": "delete", "csrf": csrf_of(own), "r": token,
                                                   "id": ids_in(listed)})
        assert "Se han borrado 0" in done.text
    assert "Marta" in app_session(base, [("list_memories", {})], name="rita")[0][0]
    with browser() as client:
        assert "Marta" in enter(client, base, "rita", link=token).text  # and the link still works for Rita


def test_a_correction_keeps_what_the_memory_was_and_replaces_its_words(server):  # noqa: F811
    base, _, _ = server
    answers, _, _ = app_session(base, [
        ("remember", {"statement": "Con Acme lanzamos la web el 15 de enero.", "kind": "decision", "project": "Acme",
                      "source": "reunión con Ana", "date": "2026-10-26"}), ("list_memories", {})],
        mode="signup", name="tina")
    [memory_id] = ids_in(answers[1])
    with browser() as client:
        page = enter(client, base, "tina")
        assert "Decisión" in page.text and "reunión con Ana" in page.text and "2026-10-26" in page.text
        form = client.get(f"{base}/memory", params={"edit": memory_id})
        assert "<textarea" in form.text and "15 de enero" in form.text
        done = client.post(f"{base}/memory", data={"action": "correct", "csrf": csrf_of(form), "id": memory_id,
                                                   "text": "Con Acme lanzamos la web el 22 de enero."})
        assert done.status_code == 200 and "Corregido" in done.text and "22 de enero" in done.text
        assert "15 de enero" not in done.text
    (listed,), _, _ = app_session(base, [("list_memories", {})], name="tina")
    assert "22 de enero" in listed and "15 de enero" not in listed and memory_id not in listed
    assert all(part in listed for part in ("decision", "Acme", "reunión con Ana", "2026-10-26"))


def test_correcting_a_fact_with_one_value_keeps_the_new_one(server):  # noqa: F811
    base, _, _ = server
    [memory_id] = ids_in(memory_of(base, "ulises", ["Vivo en Madrid."]))
    with browser() as client:
        page = enter(client, base, "ulises")
        client.post(f"{base}/memory", data={"action": "correct", "csrf": csrf_of(page), "id": memory_id,
                                            "text": "Vivo en Valencia."})
    (listed,), _, _ = app_session(base, [("list_memories", {})], name="ulises")
    assert "Valencia" in listed and "Madrid" not in listed


@pytest.mark.parametrize("typed", [SECRET, "¿Dónde vivo?", "x" * 3000])
def test_what_cannot_be_a_memory_is_refused_is_not_sent_back_and_nothing_changes(server, capfd, typed):  # noqa: F811
    base, _, folder = server
    [memory_id] = ids_in(memory_of(base, "vera", ["Mi dentista se llama Marta."]))
    with browser() as client:
        page = enter(client, base, "vera")
        refused = client.post(f"{base}/memory", data={"action": "correct", "csrf": csrf_of(page), "id": memory_id,
                                                      "text": typed})
        assert refused.status_code == 400 and "No se ha guardado" in refused.text
        assert typed not in refused.text and "Marta" in refused.text
    (listed,), _, _ = app_session(base, [("list_memories", {})], name="vera")
    assert memory_id in listed and "Marta" in listed
    logged = capfd.readouterr()
    assert typed not in logged.out + logged.err
    on_disk = b"".join(p.read_bytes() for p in folder.rglob("*") if p.is_file())
    assert typed.encode() not in on_disk and b"Marta" not in on_disk


def test_search_ignores_accents_filters_by_client_and_kind_and_shows_text_as_text(server):  # noqa: F811
    base, _, _ = server
    answers, _, _ = app_session(base, [
        ("remember", {"statement": "El camión de Acme llega el lunes.", "kind": "pending", "project": "Acme"}),
        ("remember", {"statement": "Con Bosque cerramos el precio.", "kind": "decision", "project": "Bosque"}),
        ("remember", {"statement": "Mi libro favorito es <b>Dune</b> <script>alert(1)</script>."}),
    ], mode="signup", name="wanda")
    assert all(answer.startswith("Remembered") for answer in answers)
    with browser() as client:
        page = enter(client, base, "wanda")
        assert "<b>Dune</b>" not in page.text and "<script" not in page.text and "&lt;b&gt;Dune" in page.text
        assert "3 recuerdos" in page.text

        def shown(**params):
            return client.get(f"{base}/memory", params=params).text

        found = shown(q="CAMION")
        assert "camión" in found and "Bosque cerramos" not in found and "1 de 3" in found
        assert "Bosque cerramos" in shown(project="Bosque") and "camión" not in shown(project="Bosque")
        assert "camión" in shown(kind="pending") and "Bosque cerramos" not in shown(kind="pending")
        assert "No hay nada" in shown(q="nadie dijo esto")
        assert "Bosque" in shown(project="Bosque", kind="decision") and "<option" in shown()


def test_deleting_all_of_a_filter_takes_only_that_and_still_asks(server):  # noqa: F811
    base, _, _ = server
    app_session(base, [("remember", {"statement": "Acme paga en enero.", "project": "Acme"}),
                       ("remember", {"statement": "Acme firma en marzo.", "project": "Acme"}),
                       ("remember", {"statement": "Bosque paga en mayo.", "project": "Bosque"})],
                mode="signup", name="xenia")
    with browser() as client:
        enter(client, base, "xenia")
        filtered = client.get(f"{base}/memory", params={"project": "Acme"})
        assert "Borrar los 2 de este filtro" in filtered.text
        confirm = client.post(f"{base}/memory", data={"action": "review_found", "csrf": csrf_of(filtered),
                                                      "project": "Acme", "q": "", "kind": ""})
        assert "enero" in confirm.text and "marzo" in confirm.text and "mayo" not in confirm.text
        assert "Acme" in app_session(base, [("list_memories", {})], name="xenia")[0][0]  # not yet
        marked = re.findall(r'name="id" value="([^"]+)"', confirm.text)
        client.post(f"{base}/memory", data={"action": "delete", "csrf": csrf_of(confirm), "r": "", "id": marked})
    (listed,), _, _ = app_session(base, [("list_memories", {})], name="xenia")
    assert "Acme" not in listed and "Bosque" in listed


def test_marking_nothing_deletes_nothing(server):  # noqa: F811
    base, _, _ = server
    memory_of(base, "yago", ["Mi dentista se llama Marta."])
    with browser() as client:
        page = enter(client, base, "yago")
        for action in ("review", "delete"):
            again = client.post(f"{base}/memory", data={"action": action, "csrf": csrf_of(page)})
            assert "No has marcado ninguno" in again.text and "Marta" in again.text
    assert "Marta" in app_session(base, [("list_memories", {})], name="yago")[0][0]


def test_a_long_memory_comes_in_pages_newest_first(server):  # noqa: F811
    base, _, _ = server
    items = [{"statement": f"El dato número {i} es importante."} for i in range(PAGE_SIZE + 5)]
    [answer], _, _ = app_session(base, [("remember_many", {"items": items})], mode="signup", name="zoe")
    assert answer.startswith(f"Remembered {PAGE_SIZE + 5}")
    with browser() as client:
        first = enter(client, base, "zoe")
        assert first.text.count('class="memo"') == PAGE_SIZE and f"{PAGE_SIZE + 5} recuerdos" in first.text
        assert "p=2" in first.text and "Más antiguos" in first.text and "Más recientes" not in first.text
        second = client.get(f"{base}/memory", params={"p": 2})
        assert second.text.count('class="memo"') == 5 and "Más recientes" in second.text
        assert client.get(f"{base}/memory", params={"p": "ni idea"}).status_code == 200


def test_changing_anything_needs_the_page_of_this_session(server):  # noqa: F811
    base, _, _ = server
    [memory_id] = ids_in(memory_of(base, "alba", ["Mi dentista se llama Marta."]))
    with browser() as client:
        enter(client, base, "alba")
        forged = client.post(f"{base}/memory", data={"action": "delete", "csrf": "otra", "id": [memory_id]})
        assert forged.status_code == 403
        assert client.post(f"{base}/memory", data={"action": "delete", "id": [memory_id]}).status_code == 403
    no_cookie = httpx2.post(f"{base}/memory", data={"action": "delete", "csrf": "x", "id": [memory_id]}, headers=SPANISH)
    assert no_cookie.status_code == 401 and "Vuelve a entrar" in no_cookie.text
    assert "Marta" in app_session(base, [("list_memories", {})], name="alba")[0][0]


def test_the_session_is_short_is_not_readable_by_scripts_and_ends_when_asked(server):  # noqa: F811
    base, _, folder = server
    memory_of(base, "bruno", ["Mi dentista se llama Marta."])
    with httpx2.Client(headers=SPANISH) as client:
        answer = enter(client, base, "bruno")
        assert answer.status_code == 303
        cookie = answer.headers["set-cookie"].lower()
        assert "httponly" in cookie and "samesite=strict" in cookie and "path=/memory" in cookie and "max-age=900" in cookie
        token = client.cookies["infinito_memory"]
        on_disk = (folder / "grants.db").read_bytes()
        assert token.encode() not in on_disk  # only its hash is kept

        page = client.get(f"{base}/memory")
        assert "Marta" in page.text
        out = client.post(f"{base}/memory", data={"action": "signout", "csrf": csrf_of(page)})
        assert out.status_code == 303
    gone = httpx2.get(f"{base}/memory", cookies={"infinito_memory": token}, headers=SPANISH)
    assert "Marta" not in gone.text and 'type="password"' in gone.text


def test_a_new_password_also_ends_the_page_sessions(server):  # noqa: F811
    base, _, folder = server
    memory_of(base, "carla", ["Mi dentista se llama Marta."])
    with httpx2.Client(headers=SPANISH, follow_redirects=True) as client:
        enter(client, base, "carla")
        Grants(folder).drop_account("carla")  # what changing the password does
        page = client.get(f"{base}/memory")
        assert "Marta" not in page.text and 'type="password"' in page.text


def test_over_https_the_cookie_only_travels_encrypted():
    from starlette.responses import Response

    response = Response()
    set_session_cookie(response, "token", secure=True)
    assert "secure" in response.headers["set-cookie"].lower()


def test_too_many_wrong_passwords_make_the_name_wait(server):  # noqa: F811
    base, _, _ = server
    memory_of(base, "diego", ["Mi dentista se llama Marta."])
    with browser() as client:
        for _ in range(3):
            assert enter(client, base, "diego", password="otra cosa distinta").status_code == 400
        waiting = enter(client, base, "diego")
        assert waiting.status_code == 429 and "Espera" in waiting.text and "Marta" not in waiting.text


def test_nothing_read_or_typed_on_the_page_reaches_the_log(server, capfd):  # noqa: F811
    base, _, _ = server
    [memory_id] = ids_in(memory_of(base, "elena", ["Mi dentista se llama Marta."]))
    capfd.readouterr()
    with browser() as client:
        page = enter(client, base, "elena")
        client.post(f"{base}/memory", data={"action": "delete", "csrf": csrf_of(page), "id": [memory_id]})
    logged = capfd.readouterr()
    text = logged.out + logged.err
    assert "/memory" in text
    assert all(hidden not in text for hidden in ("Marta", PASSWORD, "elena", memory_id))


def test_the_front_page_leads_to_it_in_both_languages(server):  # noqa: F811
    base, _, _ = server
    for headers, title in ((SPANISH, "Tu memoria, a la vista"), ({"accept-language": "en"}, "Your memory, in plain sight")):
        home = httpx2.get(f"{base}/", headers=headers).text
        assert 'href="/memory"' in home and "memory_page" in home and title in home


def test_the_page_speaks_both_languages_with_the_same_texts(server):  # noqa: F811
    from src.infinito3.remote_memory import TEXT
    from src.infinito3.remote_vault import TEXT as VAULT

    assert TEXT["es"].keys() == TEXT["en"].keys() and VAULT["es"].keys() == VAULT["en"].keys()
    assert TEXT["es"]["kinds"].keys() == TEXT["en"]["kinds"].keys()
    base, _, _ = server
    memory_of(base, "fran", ["My dentist is called Marta."])
    with httpx2.Client(headers={"accept-language": "en"}, follow_redirects=True) as client:
        assert "Type your password" in client.get(f"{base}/memory").text
        page = client.post(f"{base}/memory", data={"action": "signin", "name": "fran", "password": PASSWORD, "r": ""})
        assert "Your memory" in page.text and "1 memory" in page.text and "Marta" in page.text
