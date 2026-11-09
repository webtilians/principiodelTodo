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
    """Sign in and, with no link to confirm, open the list of memories."""
    answer = client.post(f"{base}/memory", data={"action": "signin", "name": name, "password": password, "r": link})
    return client.get(f"{base}/memory", params={"view": "memories"}) if 'class="dash"' in answer.text and not link else answer


def stamp_on_the_form(base):
    return re.search(r'name="stamp" value="([^"]+)"', httpx2.get(f"{base}/memory", params={"new": "1"}).text).group(1)


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
        assert "Bosque" in shown(project="Bosque", kind="decision") and "<option" in shown(view="memories")


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
    from src.infinito3.remote_portal import TEXT as PORTAL
    from src.infinito3.remote_vault import TEXT as VAULT

    assert TEXT["es"].keys() == TEXT["en"].keys() and VAULT["es"].keys() == VAULT["en"].keys()
    assert PORTAL["es"].keys() == PORTAL["en"].keys()
    assert TEXT["es"]["kinds"].keys() == TEXT["en"]["kinds"].keys()
    base, _, _ = server
    memory_of(base, "fran", ["My dentist is called Marta."])
    with httpx2.Client(headers={"accept-language": "en"}, follow_redirects=True) as client:
        assert "Type your password" in client.get(f"{base}/memory").text
        page = enter(client, base, "fran")
        assert "<h1>Memories</h1>" in page.text and "1 memory" in page.text and "Marta" in page.text


def kinds_of(base, name):
    """An account with memories of several kinds, for two clients."""
    steps = [("remember", {"statement": "Acme aprueba el presupuesto.", "kind": "decision", "project": "Acme"}),
             ("remember", {"statement": "Acme cambia de logo.", "kind": "decision", "project": "Acme"}),
             ("remember", {"statement": "Globex decide el color.", "kind": "decision", "project": "Globex"}),
             ("remember", {"statement": "Reclamar a Marta los textos.", "kind": "pending", "project": "Acme"}),
             ("remember", {"statement": "Mi dentista se llama Marta."})]
    answers, _, _ = app_session(base, steps, mode="signup", name=name)
    assert all(answer.startswith("Remembered") for answer in answers), answers


def home_of(client, base, name):
    client.post(f"{base}/memory", data={"action": "signin", "name": name, "password": PASSWORD, "r": ""})
    return client.get(f"{base}/memory")


def neuron(page, kind):
    return re.search(rf'<a class="neuron"[^>]*kind={kind}".*?</a>', page.text, re.S).group(0)


def test_signing_in_lands_on_the_vault_home_with_the_design_and_real_numbers(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "aina")
    with browser() as client:
        page = home_of(client, base, "aina")
    assert 'class="dash"' in page.text and "<script" not in page.text and "<h1>Bóveda</h1>" in page.text
    for word in ("Inicio", "Recuerdos", "Proyectos", "Búsqueda", "Ajustes", "Memorias recientes", "Claves protegidas",
                 "Actividad", "Nodos de memoria", "Buscar en tu memoria"):
        assert word in page.text, word
    assert "<b>5</b><span>Recuerdos</span>" in page.text  # the memories
    assert "<b>7</b><span>Nodos de memoria</span>" in page.text  # and the two clients
    assert "Memoria sellada" in page.text and "La bóveda es por invitación" in page.text  # this account has no vault
    assert "Mi dentista" in page.text or "Reclamar a Marta" in page.text
    assert "img-src 'self'" in page.headers["content-security-policy"]


def test_it_never_promises_more_than_it_does(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "ona")
    with browser() as client:
        page = home_of(client, base, "ona")
    assert "extremo a extremo" not in page.text and "256-bit" in page.text and "en reposo" in page.text


def test_each_kind_is_a_bubble_of_the_brain_that_leads_to_its_memories(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "bruna")
    with browser() as client:
        page = home_of(client, base, "bruna")
        assert page.text.count('<a class="neuron"') == 8
        assert 'aria-label="Decisión: 3"' in neuron(page, "decision") and "Decisión · 3" in neuron(page, "decision")
        assert 'aria-label="Sobre ti: 1"' in neuron(page, "none") and 'aria-label="Propuesta: 0"' in neuron(page, "proposal")
        chosen = client.get(f"{base}/memory", params={"kind": "pending"})
        assert "Reclamar a Marta" in chosen.text and "presupuesto" not in chosen.text


def test_the_bubbles_are_on_the_picture(server):  # noqa: F811
    from src.infinito3.remote_portal import NEURON_SPOTS

    assert len(NEURON_SPOTS) == 8
    for x, y, radius in NEURON_SPOTS.values():
        assert radius < x < 725 - radius and radius < y < 455 - radius


def test_the_brain_picture_needs_no_sign_in_and_says_nothing_of_anyone(server):  # noqa: F811
    base, _, _ = server
    picture = httpx2.get(f"{base}/memory/brain.webp")
    assert picture.status_code == 200 and picture.headers["content-type"] == "image/webp"
    assert picture.content[:4] == b"RIFF" and picture.content[8:12] == b"WEBP"
    assert "set-cookie" not in picture.headers and "public" in picture.headers["cache-control"]


def test_the_memories_saved_with_no_kind_can_be_chosen(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "delia")
    with browser() as client:
        enter(client, base, "delia")
        chosen = client.get(f"{base}/memory", params={"kind": "none"})
        assert "Mi dentista" in chosen.text and "presupuesto" not in chosen.text and "Reclamar" not in chosen.text
        assert '<option value="none" selected>Sobre ti</option>' in chosen.text


def test_the_top_search_finds_and_the_bell_leads_to_what_is_pending(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "elsa")
    with browser() as client:
        page = home_of(client, base, "elsa")
        assert 'aria-label="Lo pendiente: 1"' in page.text and 'class="dot"' in page.text
        found = client.get(f"{base}/memory", params={"q": "presupuesto"})
        assert "Acme aprueba" in found.text and "Mi dentista" not in found.text and 'value="presupuesto"' in found.text


def test_a_search_cannot_break_out_of_the_page(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "flora")
    with browser() as client:
        enter(client, base, "flora")
        for params in ({"project": '"><script>alert(1)</script>'}, {"q": '"><script>alert(1)</script>'}):
            assert "<script>" not in client.get(f"{base}/memory", params=params).text


def test_what_a_memory_says_cannot_run_on_the_home(server):  # noqa: F811
    base, _, _ = server
    statement = "Mi gato se llama <script>alert(1)</script> Tom."
    answers, _, _ = app_session(base, [("remember", {"statement": statement, "kind": "fact", "project": "<b>Acme</b>"})],
                                mode="signup", name="tina")
    assert answers[0].startswith("Remembered"), answers
    with browser() as client:
        page = home_of(client, base, "tina")
    assert "<script>" not in page.text and "<b>Acme</b>" not in page.text and "&lt;script&gt;" in page.text


def test_deleting_what_a_kind_shows_takes_only_that_kind(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "gala")
    with browser() as client:
        enter(client, base, "gala")
        chosen = client.get(f"{base}/memory", params={"kind": "none"})
        confirm = client.post(f"{base}/memory", data={"action": "review_found", "csrf": csrf_of(chosen), "kind": "none",
                                                      "q": "", "project": ""})
        assert "Mi dentista" in confirm.text and "presupuesto" not in confirm.text


def test_with_nothing_kept_the_home_says_so(server):  # noqa: F811
    base, _, _ = server
    app_session(base, [], mode="signup", name="hada")
    with browser() as client:
        page = home_of(client, base, "hada")
    assert page.text.count('<a class="neuron"') == 8 and "Lo que le cuentes a tus IAs" in page.text
    assert "<b>0</b><span>Recuerdos</span>" in page.text


def test_nothing_of_the_portal_appears_before_signing_in(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "iris")
    for params in ({}, {"kind": "decision"}, {"view": "projects"}, {"view": "settings"}, {"view": "vault"}):
        page = httpx2.get(f"{base}/memory", params=params, headers=SPANISH)
        assert 'class="neuron' not in page.text and 'class="side"' not in page.text and "presupuesto" not in page.text
        assert 'type="password"' in page.text


def test_the_projects_page_shows_each_client_and_what_is_still_open(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "kira")
    with browser() as client:
        enter(client, base, "kira")
        page = client.get(f"{base}/memory", params={"view": "projects"})
    assert "Acme" in page.text and "Globex" in page.text and "Reclamar a Marta" in page.text
    assert 'href="/memory?project=Acme"' in page.text and "2 decisión" in page.text


def test_with_no_clients_the_projects_page_says_so(server):  # noqa: F811
    base, _, _ = server
    memory_of(base, "lila", ["Mi dentista se llama Marta."])
    with browser() as client:
        enter(client, base, "lila")
        assert "Ningún recuerdo tiene cliente" in client.get(f"{base}/memory", params={"view": "projects"}).text


def test_the_settings_page_says_whose_account_it_is_and_nothing_false(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "mara")
    with browser() as client:
        enter(client, base, "mara")
        page = client.get(f"{base}/memory", params={"view": "settings"})
    assert "mara" in page.text and "AES-256-GCM" in page.text and "extremo a extremo" not in page.text


def test_the_menu_marks_where_you_are_and_it_all_speaks_english_too(server):  # noqa: F811
    base, _, _ = server
    kinds_of(base, "jana")
    with httpx2.Client(headers={"accept-language": "en"}, follow_redirects=True) as client:
        page = home_of(client, base, "jana")
        assert "<h1>Vault</h1>" in page.text and "Recent memories" in page.text and "Memory sealed" in page.text
        assert 'aria-label="About you: 1"' in neuron(page, "none")
        assert re.search(r'<a href="/memory" class=on aria-current=page>', page.text)
        listing = client.get(f"{base}/memory", params={"view": "memories"})
        assert re.search(r'<a href="/memory\?view=memories" class=on aria-current=page>', listing.text)
        assert client.post(f"{base}/memory", data={"action": "signout", "csrf": csrf_of(page)}).status_code == 200
        assert 'class="side"' not in client.get(f"{base}/memory").text


def test_whoever_wants_creates_a_memory_on_the_page_with_no_invitation_and_no_app(server):  # noqa: F811
    base, _, _ = server

    def create(client, name, password=PASSWORD, repeat=None):
        return client.post(f"{base}/memory", data={"action": "signup", "name": name, "password": password,
                                                   "repeat": password if repeat is None else repeat,
                                                   "stamp": stamp_on_the_form(base)})

    with browser() as client:
        entry = client.get(f"{base}/memory")
        assert 'href="/memory?new=1"' in entry.text and "Primera vez" in entry.text
        form = client.get(f"{base}/memory", params={"new": "1"})
        assert 'name="action" value="signup"' in form.text and 'name="repeat"' in form.text and "invite" not in form.text
        assert "ni quien administra" not in form.text.lower()  # a claim the front page's tests refuse

        assert "no coinciden" in create(client, "nadia", repeat="otra-frase-distinta").text
        assert "al menos" in create(client, "nadia", password="corta").text
        wrong_name = create(client, "NO VALE!")
        assert wrong_name.status_code == 400 and "Elige un nombre" in wrong_name.text
        assert "nadia" not in client.get(f"{base}/memory").text  # none of those created anything

        done = create(client, "nadia")
        assert done.status_code == 200 and re.search(r'class="code">[^<]+<', done.text)
        assert 'action="/#connect"' in done.text and "conectar mi primera IA" in done.text
        opened = client.get(f"{base}/memory")
        assert "Lo que le cuentes" in opened.text and 'type="password"' not in opened.text  # inside, nothing more to type
        assert 'value="signup"' not in client.get(f"{base}/memory", params={"new": "1"}).text  # and no form once inside

    with browser() as stranger:
        taken = create(stranger, "nadia")
        assert taken.status_code == 400 and "ya está en uso" in taken.text and 'class="code"' not in taken.text
        assert "Abrir mi memoria" not in enter(stranger, base, "nadia").text  # the password opens it from anywhere

    (listed,), _, _ = app_session(base, [("list_memories", {})], name="nadia")  # and an app can connect to it
    assert isinstance(listed, str)


def test_a_link_to_confirm_something_does_not_offer_to_create_a_memory(server):  # noqa: F811
    base, _, _ = server
    listed = memory_of(base, "lucas", ["Mi taller se llama Motos Vega."])
    (answer,), _, _ = app_session(base, [("memory_page", {"memory_ids": ids_in(listed)})], name="lucas")
    assert "/memory?new=1" not in httpx2.get(link_in(answer).group(0), headers=SPANISH).text


def try_to_sign_up(base, name, **fields):
    form = {"action": "signup", "name": name, "password": PASSWORD, "repeat": PASSWORD, **fields}
    return httpx2.post(f"{base}/memory", data=form, headers=SPANISH)


def test_what_a_program_sends_creates_no_memory_and_a_person_is_only_asked_to_retry(server, monkeypatch):  # noqa: F811
    base, _, folder = server
    good = stamp_on_the_form(base)
    for fields in ({"stamp": good, "homepage": "http://spam.example"}, {}, {"stamp": "123.forged"}, {"stamp": "no"}):
        refused = try_to_sign_up(base, "robot", **fields)
        assert refused.status_code == 400 and 'class="code"' not in refused.text
        assert "Espera unos segundos" in refused.text and 'name="stamp"' in refused.text  # a fresh form to try again

    monkeypatch.setattr("src.infinito3.remote_accounts.SIGNUP_MIN_SECONDS", 60)
    assert try_to_sign_up(base, "robot", stamp=stamp_on_the_form(base)).status_code == 400  # at once: no person is so fast
    monkeypatch.setattr("src.infinito3.remote_accounts.SIGNUP_MAX_SECONDS", -1)
    monkeypatch.setattr("src.infinito3.remote_accounts.SIGNUP_MIN_SECONDS", -60)
    assert try_to_sign_up(base, "robot", stamp=stamp_on_the_form(base)).status_code == 400  # a form kept for too long
    accounts = folder / "accounts.json"
    assert not accounts.exists() or "robot" not in accounts.read_text()


def test_the_field_only_a_program_fills_in_is_out_of_a_persons_way(server):  # noqa: F811
    base, _, _ = server
    form = httpx2.get(f"{base}/memory", params={"new": "1"}, headers=SPANISH).text
    assert '<div class="leave" aria-hidden="true">' in form and 'name="homepage"' in form and 'tabindex="-1"' in form
    assert ".leave { position: absolute; left: -9999px; }" in form


def test_new_memories_are_limited_by_address_and_in_all_and_failed_tries_do_not_count(server, monkeypatch):  # noqa: F811
    base, _, folder = server
    monkeypatch.setattr("src.infinito3.remote_accounts.SIGNUPS_PER_ADDRESS", 2)
    monkeypatch.setattr("src.infinito3.remote_accounts.SIGNUPS_IN_ALL", 3)

    def create(name, address, **fields):
        form = {"action": "signup", "name": name, "password": PASSWORD, "repeat": PASSWORD,
                "stamp": stamp_on_the_form(base), **fields}
        return httpx2.post(f"{base}/memory", data=form, headers={**SPANISH, "x-forwarded-for": address})

    assert create("nueva-a", "203.0.113.7", repeat="otra-frase-distinta").status_code == 400  # a mistake is not a memory
    assert create("nueva-a", "203.0.113.7").status_code == 200
    assert create("nueva-b", "203.0.113.7").status_code == 200
    third = create("nueva-c", "203.0.113.7")
    assert third.status_code == 429 and "Vuelve a probar dentro de 60 minutos" in third.text and 'class="code"' not in third.text
    assert create("nueva-d", "198.51.100.9").status_code == 200  # someone else, somewhere else
    assert create("nueva-e", "192.0.2.44").status_code == 429  # and now the whole server has had enough for this hour
    accounts = (folder / "accounts.json").read_text()
    assert "nueva-c" not in accounts and "nueva-e" not in accounts
    assert "203.0.113.7" not in accounts and "198.51.100.9" not in accounts  # an address never reaches the disk
