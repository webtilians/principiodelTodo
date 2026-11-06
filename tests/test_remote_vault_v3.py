"""The vault on the server: reachable from any app, and no value ever reaches the agent, the chat or a log."""
import re
import tarfile

import pytest

from src.infinito3.remote_accounts import Accounts
from tests.test_remote_accounts_v3 import PASSWORD, app_session, server  # noqa: F401  # server is a fixture

pykeepass = pytest.importorskip("pykeepass")
httpx2 = pytest.importorskip("httpx2")

MASTER = "una frase maestra bastante larga"
VALUE = "valor-secreto-limonero-77"
SPANISH = {"accept-language": "es-ES"}


def invited(base, people, name, steps=()):
    """A new account the server's owner has let in, and what its app gets from these tools."""
    app_session(base, [], mode="signup", name=name)
    assert Accounts(people.folder).allow_vault(name)
    return app_session(base, list(steps), name=name)[0]


def link_in(answer):
    return re.search(r"http\S+/vault\?r=\S+", answer).group(0)


def token_of(link):
    return link.split("r=")[1]


def send(base, link, **fields):
    return httpx2.post(f"{base}/vault", data={"r": token_of(link), **fields}, headers=SPANISH)


def keep(base, people, name, entry="Router de casa", value=VALUE, master=MASTER):
    """The person keeps a value through the page; returns what the app can do next."""
    (answer,) = invited(base, people, name, [("vault_add", {"entry": entry})])
    page = send(base, link_in(answer), master=master, repeat=master, username="admin", value=value)
    assert page.status_code == 200 and "ya está en tu bóveda" in page.text
    return link_in(answer)


def test_an_account_not_invited_gets_no_link_and_is_told_where_to_keep_it(server):  # noqa: F811
    base, _, _ = server
    answers, _, _ = app_session(base, [("vault_add", {"entry": "Router"}), ("vault_reveal", {"entry": "Router"}),
                                       ("vault_list", {})], mode="signup", name="vera")
    assert all("by invitation" in answer and "password manager" in answer for answer in answers)
    assert not any("/vault?r=" in answer for answer in answers)


def test_a_private_address_has_no_vault(server):  # noqa: F811
    import anyio
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    base, people, _ = server
    token = people.add("eva")

    async def run():
        async with streamable_http_client(f"{base}/u/{token}/mcp") as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                return [(await client.call_tool(name, arguments)).content[0].text
                        for name, arguments in (("vault_add", {"entry": "Router"}), ("vault_list", {}))]

    assert all("needs an INFINITO account" in answer for answer in anyio.run(run))


def test_no_tool_has_a_way_to_receive_a_value(server):  # noqa: F811
    base, _, _ = server
    _, _, app = app_session(base, [], mode="signup", name="tool")
    fields = {t.name: set(t.input_schema["properties"]) for t in app.tools if t.name.startswith("vault_")}
    assert fields == {"vault_add": {"entry"}, "vault_reveal": {"entry"}, "vault_delete": {"entry"}, "vault_list": set()}


def test_a_value_kept_on_the_page_never_reaches_the_agent_and_opens_with_keepass(server, capfd):  # noqa: F811
    base, people, folder = server
    keep(base, people, "nora")
    answers, _, _ = app_session(base, [("vault_list", {}), ("vault_reveal", {"entry": "Router de casa"}),
                                       ("recall", {"query": "router"}), ("list_memories", {})], name="nora")
    assert "- Router de casa" in answers[0]
    assert not any(VALUE in answer or MASTER in answer for answer in answers)
    assert "No memories" in answers[3]  # the vault does not write to the memory

    vault = folder / "accounts" / "nora" / "vault.kdbx"
    assert VALUE.encode() not in vault.read_bytes()
    names = folder / "accounts" / "nora" / "vault-names.sealed"
    assert b"Router" not in names.read_bytes()
    db = pykeepass.PyKeePass(str(vault), password=MASTER)  # a standard file: KeePassXC opens the same
    entry = db.find_entries(title="Router de casa", first=True)
    assert (entry.username, entry.password) == ("admin", VALUE)

    logged = capfd.readouterr()
    assert VALUE not in logged.out + logged.err and MASTER not in logged.out + logged.err
    assert not list((folder / "accounts" / "nora").glob("*.tmp")) and not list((folder / "accounts" / "nora").glob("*.new"))


def test_the_value_is_shown_only_on_the_page_once_and_the_page_leaves_by_itself(server):  # noqa: F811
    base, people, _ = server
    keep(base, people, "iris")
    (answer,), _, _ = app_session(base, [("vault_reveal", {"entry": "router DE casa"})], name="iris")
    link = link_in(answer)
    assert VALUE not in answer
    opened = httpx2.get(link, headers=SPANISH)
    assert opened.status_code == 200 and VALUE not in opened.text and 'name="value"' not in opened.text
    assert "autofocus" not in opened.text  # it would scroll past the warning to check the address

    shown = send(base, link, master=MASTER)
    assert shown.status_code == 200 and VALUE in shown.text and "admin" in shown.text
    assert shown.headers["cache-control"] == "no-store" and shown.headers["refresh"].startswith("60;")
    assert "script" not in shown.headers["content-security-policy"]
    assert send(base, link, master=MASTER).status_code == 400  # used up
    assert httpx2.get(link).status_code == 400


def test_opening_a_link_does_not_use_it_up_because_chat_apps_may_fetch_it_first(server):  # noqa: F811
    base, people, _ = server
    (answer,) = invited(base, people, "fetch", [("vault_add", {"entry": "Wifi"})])
    link = link_in(answer)
    assert httpx2.get(link + "&utm_source=chatgpt.com").status_code == 200  # as ChatGPT opens it
    assert httpx2.get(link).status_code == 200
    assert send(base, link, master=MASTER, repeat=MASTER, value=VALUE).status_code == 200


def test_a_wrong_master_password_opens_nothing_keeps_the_link_and_makes_you_wait(server):  # noqa: F811
    base, people, _ = server
    keep(base, people, "lola")
    (answer,), _, _ = app_session(base, [("vault_reveal", {"entry": "Router de casa"})], name="lola")
    link = link_in(answer)
    wrong = send(base, link, master="no es esta la clave maestra")
    assert wrong.status_code == 400 and "no abre tu bóveda" in wrong.text and VALUE not in wrong.text
    for _ in range(2):
        send(base, link, master="tampoco es esta la clave maestra")
    waits = send(base, link, master=MASTER)  # the right one, too late: after three failures the account waits
    assert waits.status_code == 400 and "Demasiados intentos" in waits.text and VALUE not in waits.text


def test_a_new_vault_needs_a_long_master_password_typed_twice_and_a_value(server):  # noqa: F811
    base, people, folder = server
    (answer,) = invited(base, people, "nuevo", [("vault_add", {"entry": "Wifi"})])
    link = link_in(answer)
    first = httpx2.get(link, headers=SPANISH)
    assert 'name="repeat"' in first.text and "primera vez" in first.text and "En pruebas" in first.text
    assert "al menos 14" in send(base, link, master="corta", repeat="corta", value=VALUE).text
    assert "no coinciden" in send(base, link, master=MASTER, repeat=MASTER + "x", value=VALUE).text
    assert "Escribe la clave" in send(base, link, master=MASTER, repeat=MASTER, value="").text
    assert "no se pueden guardar" in send(base, link, master=MASTER, repeat=MASTER, value="x" * 4097).text
    assert "no se pueden guardar" in send(base, link, master=MASTER, repeat=MASTER, value=VALUE, username="a\x01").text
    assert not (folder / "accounts" / "nuevo" / "vault.kdbx").exists()


def test_keeping_a_name_again_replaces_it_and_says_so_first(server):  # noqa: F811
    base, people, folder = server
    keep(base, people, "dani", entry="Wifi", value="valor-viejo-1")
    (answer,), _, _ = app_session(base, [("vault_add", {"entry": "WIFI"})], name="dani")
    link = link_in(answer)
    assert "se sustituirá" in httpx2.get(link, headers=SPANISH).text
    assert send(base, link, master=MASTER, value="valor-nuevo-2").status_code == 200
    (listed,), _, _ = app_session(base, [("vault_list", {})], name="dani")
    assert listed.count("- ") == 1
    db = pykeepass.PyKeePass(str(folder / "accounts" / "dani" / "vault.kdbx"), password=MASTER)
    assert [e.password for e in db.entries] == ["valor-nuevo-2"]


def test_the_person_names_the_entry_and_says_what_it_is_for(server):  # noqa: F811
    base, people, folder = server
    (answer,) = invited(base, people, "ruth", [("vault_add", {"entry": "Clave nueva"})])
    link = link_in(answer)
    page = httpx2.get(link, headers=SPANISH)
    assert 'name="name"' in page.text and 'value="Clave nueva"' in page.text and 'name="note"' in page.text
    assert "No pongas aquí ninguna clave" in page.text
    saved = send(base, link, name="  Panel de  Hetzner ", note="Cuenta de pagos", master=MASTER, repeat=MASTER,
                 username="root", value=VALUE)
    assert saved.status_code == 200 and "Panel de Hetzner" in saved.text

    (listed, reveal), _, _ = app_session(base, [("vault_list", {}), ("vault_reveal", {"entry": "panel de hetzner"})],
                                         name="ruth")
    assert "- Panel de Hetzner: Cuenta de pagos" in listed and "Clave nueva" not in listed and VALUE not in listed
    db = pykeepass.PyKeePass(str(folder / "accounts" / "ruth" / "vault.kdbx"), password=MASTER)
    assert [(e.title, e.notes, e.username, e.password) for e in db.entries] == [
        ("Panel de Hetzner", "Cuenta de pagos", "root", VALUE)]
    assert "Cuenta de pagos".encode() not in (folder / "accounts" / "ruth" / "vault-names.sealed").read_bytes()
    shown = send(base, link_in(reveal), master=MASTER).text
    assert VALUE in shown and "Cuenta de pagos" in shown


def test_a_note_or_a_name_holding_the_secret_is_refused_and_not_sent_back(server):  # noqa: F811
    base, people, folder = server
    (answer,) = invited(base, people, "alba", [("vault_add", {"entry": "Wifi"})])
    link = link_in(answer)
    common = {"master": MASTER, "repeat": MASTER, "value": VALUE}
    for field, text in (("note", f"la clave es {VALUE}"), ("note", "La clave del wifi es limonero2024"),
                        ("note", "x" * 201),
                        ("name", f"Wifi {VALUE}"), ("name", "La clave del wifi es limonero2024"), ("name", "   ")):
        page = send(base, link, **common, **{field: text})
        assert page.status_code == 400 and ("La nota" in page.text or "Elige un nombre" in page.text), text
        assert VALUE not in page.text and "limonero2024" not in page.text
    assert not (folder / "accounts" / "alba" / "vault.kdbx").exists()


def test_renaming_onto_another_entry_is_refused_instead_of_replacing_it(server):  # noqa: F811
    base, people, folder = server
    keep(base, people, "cris", entry="Banco", value="valor-del-banco-1")
    (answer,), _, _ = app_session(base, [("vault_add", {"entry": "Wifi"})], name="cris")
    link = link_in(answer)
    taken = send(base, link, name="BANCO", master=MASTER, value="valor-del-wifi-2")
    assert taken.status_code == 400 and "otra entrada con ese nombre" in taken.text
    assert send(base, link, name="Wifi de casa", master=MASTER, value="valor-del-wifi-2").status_code == 200
    db = pykeepass.PyKeePass(str(folder / "accounts" / "cris" / "vault.kdbx"), password=MASTER)
    assert {e.title: e.password for e in db.entries} == {"Banco": "valor-del-banco-1", "Wifi de casa": "valor-del-wifi-2"}


def test_a_note_can_be_added_later_without_typing_the_secret_again(server):  # noqa: F811
    base, people, folder = server
    keep(base, people, "tere", entry="Wifi")
    (answer,), _, _ = app_session(base, [("vault_add", {"entry": "Wifi"})], name="tere")
    link = link_in(answer)
    page = httpx2.get(link, headers=SPANISH)
    assert "se conservan los que tiene" in page.text and 'name="repeat"' not in page.text
    assert send(base, link, name="Wifi", note="El de casa", master=MASTER).status_code == 200
    (listed,), _, _ = app_session(base, [("vault_list", {})], name="tere")
    assert "- Wifi: El de casa" in listed
    db = pykeepass.PyKeePass(str(folder / "accounts" / "tere" / "vault.kdbx"), password=MASTER)
    assert [(e.username, e.password, e.notes) for e in db.entries] == [("admin", VALUE, "El de casa")]


def test_a_new_name_with_no_secret_is_refused(server):  # noqa: F811
    base, people, _ = server
    keep(base, people, "pau", entry="Wifi")
    (answer,), _, _ = app_session(base, [("vault_add", {"entry": "Wifi"})], name="pau")
    page = send(base, link_in(answer), name="Otra", master=MASTER)
    assert page.status_code == 400 and "Escribe la clave" in page.text


def test_notes_written_elsewhere_never_reach_the_agent(server):  # noqa: F811
    base, people, folder = server
    keep(base, people, "kim", entry="Wifi")
    path = folder / "accounts" / "kim" / "vault.kdbx"
    db = pykeepass.PyKeePass(str(path), password=MASTER)  # as if it had been edited in KeePassXC
    db.add_entry(db.root_group, "Banco", "yo", "valor-banco", notes="codigos de rescate 1234")
    db.save()
    (answer,), _, _ = app_session(base, [("vault_add", {"entry": "Correo"})], name="kim")
    assert send(base, link_in(answer), master=MASTER, value="valor-correo").status_code == 200
    (listed,), _, _ = app_session(base, [("vault_list", {})], name="kim")
    assert "- Banco" in listed and "rescate" not in listed


def test_a_list_written_before_notes_existed_still_reads(server):  # noqa: F811
    import json

    from src.infinito3.remote_accounts import _seal
    from src.infinito3.remote_vault import ServerVault, list_answer

    _, _, folder = server
    key = b"k" * 32
    vault = ServerVault(folder / "viejo", "viejo")
    vault.folder.mkdir()
    vault.names_path.write_bytes(_seal(key, json.dumps(["Wifi"]).encode(), "vault-names:viejo"))
    assert vault.listed(key) == [{"name": "Wifi", "note": ""}]
    assert "- Wifi" in list_answer(vault.listed(key))


def test_the_agent_is_told_not_to_invent_a_name(server):  # noqa: F811
    base, _, _ = server
    _, _, app = app_session(base, [], mode="signup", name="desc")
    tools = {t.name: t.description for t in app.tools}
    assert "ask the user first" in tools["vault_add"] and "do not invent a name" in tools["vault_add"]
    assert "vault_list first" in tools["vault_reveal"] and "notes" in tools["vault_list"]


def test_asking_to_see_what_is_not_there_says_so_without_asking_for_the_master_password(server):  # noqa: F811
    base, people, _ = server
    keep(base, people, "marc")
    (answer,), _, _ = app_session(base, [("vault_reveal", {"entry": "Banco"})], name="marc")
    page = httpx2.get(link_in(answer), headers=SPANISH)
    assert "No está en tu bóveda" in page.text and 'name="master"' not in page.text


def test_one_persons_link_and_vault_are_not_anothers(server):  # noqa: F811
    base, people, folder = server
    keep(base, people, "uno", value="valor-de-uno-11")
    (answer,) = invited(base, people, "dos", [("vault_add", {"entry": "Router de casa"})])
    assert "- Router" not in app_session(base, [("vault_list", {})], name="dos")[0][0]
    assert send(base, link_in(answer), master=MASTER, repeat=MASTER, value="valor-de-dos-22").status_code == 200
    one = pykeepass.PyKeePass(str(folder / "accounts" / "uno" / "vault.kdbx"), password=MASTER)
    assert [e.password for e in one.entries] == ["valor-de-uno-11"]


def test_a_vault_link_that_is_wrong_shows_nothing_and_a_denied_account_loses_its_link(server):  # noqa: F811
    base, people, _ = server
    assert httpx2.get(f"{base}/vault?r=no-such-link", headers=SPANISH).status_code == 400
    assert httpx2.get(f"{base}/vault").status_code == 400
    (answer,) = invited(base, people, "sale", [("vault_add", {"entry": "Wifi"})])
    Accounts(people.folder).allow_vault("sale", False)
    assert httpx2.get(link_in(answer)).status_code == 400


def test_a_secret_given_as_an_entrys_name_gets_no_link_and_is_not_repeated(server):  # noqa: F811
    base, people, _ = server
    answers = invited(base, people, "fina", [("vault_add", {"entry": "La clave del wifi es limonero2024"}),
                                             ("vault_add", {"entry": "   "}), ("vault_reveal", {"entry": "x" * 81}),
                                             ("vault_add", {"entry": "a\x00b"})])
    assert "looks like the secret itself" in answers[0] and "limonero2024" not in answers[0]
    assert "give the entry a name" in answers[1] and "at most 80" in answers[2] and "control" in answers[3]
    assert not any("/vault?r=" in answer for answer in answers)


def test_a_name_or_a_value_cannot_inject_anything_into_the_page(server):  # noqa: F811
    base, people, _ = server
    name = "<script>alert(1)</script>"
    (answer,) = invited(base, people, "html", [("vault_add", {"entry": name})])
    link = link_in(answer)
    page = httpx2.get(link)
    assert "<script>alert" not in page.text and "&lt;script&gt;" in page.text
    value = "<img src=x onerror=alert(2)>"
    assert send(base, link, master=MASTER, repeat=MASTER, value=value).status_code == 200
    (shown,), _, _ = app_session(base, [("vault_reveal", {"entry": name})], name="html")
    text = send(base, link_in(shown), master=MASTER).text
    assert "<img src=x" not in text and "&lt;img src=x" in text


def test_a_copy_holds_the_vault_locked_and_restores_it(server, tmp_path):  # noqa: F811
    from src.infinito3 import remote_backup

    base, people, folder = server
    keep(base, people, "copia")
    saved = tmp_path / "copias"
    remote_backup.backup(folder, saved)
    copy = next(saved.iterdir())
    with tarfile.open(copy) as archive:
        names = archive.getnames()
        everything = b"".join(archive.extractfile(m).read() for m in archive.getmembers())
    assert "accounts/copia/vault.kdbx" in names and "accounts/copia/vault-names.sealed" in names
    assert VALUE.encode() not in everything and b"Router de casa" not in everything

    elsewhere = tmp_path / "otro"
    remote_backup.restore(copy, elsewhere)
    db = pykeepass.PyKeePass(str(elsewhere / "accounts" / "copia" / "vault.kdbx"), password=MASTER)
    assert [e.password for e in db.entries] == [VALUE]


def test_a_copy_with_something_else_as_a_vault_is_refused(tmp_path):
    from src.infinito3 import remote_backup

    data = tmp_path / "datos"
    (data / "accounts" / "ana").mkdir(parents=True)
    (data / "accounts" / "ana" / "vault.kdbx").write_bytes(b"esto no es KeePass")
    with pytest.raises(ValueError, match="not a KeePass file"):
        remote_backup.backup(data, tmp_path / "copias")


def test_an_entry_is_deleted_only_on_the_page_with_the_master_password(server):  # noqa: F811
    base, people, folder = server
    keep(base, people, "hugo")
    (other,), _, _ = app_session(base, [("vault_add", {"entry": "Panel"})], name="hugo")
    send(base, link_in(other), master=MASTER, value="otro-valor-distinto-55")
    (answer,), _, _ = app_session(base, [("vault_delete", {"entry": "router de casa"})], name="hugo")
    link = link_in(answer)
    assert "Nothing changes until" in answer and VALUE not in answer
    opened = httpx2.get(link, headers=SPANISH)
    assert opened.status_code == 200 and "para siempre" in opened.text and 'name="value"' not in opened.text

    vault = folder / "accounts" / "hugo" / "vault.kdbx"
    wrong = send(base, link, master="esta clave maestra no es")
    assert wrong.status_code == 400 and "no abre tu bóveda" in wrong.text
    assert "- Router de casa" in app_session(base, [("vault_list", {})], name="hugo")[0][0]

    done = send(base, link, master=MASTER)
    assert done.status_code == 200 and "ya no está en tu bóveda" in done.text and VALUE not in done.text
    assert "caducado" in httpx2.get(link, headers=SPANISH).text  # a link works once
    (listed,), _, _ = app_session(base, [("vault_list", {})], name="hugo")
    assert "Router" not in listed and "- Panel" in listed
    db = pykeepass.PyKeePass(str(vault), password=MASTER)  # still a standard file, without that entry
    assert db.find_entries(title="Router de casa", first=True) is None
    assert db.find_entries(title="Panel", first=True).password == "otro-valor-distinto-55"


def test_deleting_what_is_not_there_says_so_and_an_account_without_invitation_gets_no_link(server):  # noqa: F811
    base, people, _ = server
    keep(base, people, "lola")
    (answer,), _, _ = app_session(base, [("vault_delete", {"entry": "No existe"})], name="lola")
    assert "No está en tu bóveda" in httpx2.get(link_in(answer), headers=SPANISH).text
    (refused,), _, _ = app_session(base, [("vault_delete", {"entry": "Router"})], mode="signup", name="mina")
    assert "by invitation" in refused and "/vault?r=" not in refused


def test_the_memory_page_lists_names_only_and_leads_to_the_vault_page(server):  # noqa: F811
    base, people, _ = server
    keep(base, people, "olga")
    with httpx2.Client(headers=SPANISH, follow_redirects=True) as client:
        page = client.post(f"{base}/memory", data={"action": "signin", "name": "olga", "password": PASSWORD, "r": ""})
        assert "Router de casa" in page.text and VALUE not in page.text
        assert "Bóveda activa" in page.text and 'href="/memory?view=vault"' in page.text  # the home leads to all of them
        assert "Router de casa" in client.get(f"{base}/memory", params={"view": "vault"}).text
        csrf = re.search(r'name="csrf" value="([^"]+)"', page.text).group(1)
        asked = client.post(f"{base}/memory", data={"action": "vault", "csrf": csrf, "entry": "Router de casa",
                                                    "what": "delete"})
        assert str(asked.url).startswith(f"{base}/vault?r=") and "para siempre" in asked.text and VALUE not in asked.text
        assert "Te lo pide tu IA" not in asked.text  # the person asked for this one themselves
        token = str(asked.url).split("r=")[1]
        gone = client.post(f"{base}/vault", data={"r": token, "master": MASTER}).text
        assert "ya no está" in gone and 'href="/memory"' in gone
        assert "Router de casa" not in client.get(f"{base}/memory").text
        made_up = client.post(f"{base}/memory", data={"action": "vault", "csrf": csrf, "entry": "Otra", "what": "reveal"})
        assert "/vault" not in str(made_up.url)


def test_an_account_without_invitation_sees_no_vault_on_the_memory_page(server):  # noqa: F811
    base, _, _ = server
    app_session(base, [], mode="signup", name="paula")
    with httpx2.Client(headers=SPANISH, follow_redirects=True) as client:
        page = client.post(f"{base}/memory", data={"action": "signin", "name": "paula", "password": PASSWORD, "r": ""})
        assert page.status_code == 200 and "Tu bóveda" not in page.text
