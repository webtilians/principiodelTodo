"""Photos in a person's memory: sealed on disk, shown only to them, and gone with the memory that names them."""
import base64
import http.client
import io
import re
import shutil
import struct
import urllib.error
import zlib
from email.message import Message
from pathlib import Path

import pytest

from src.infinito3 import remote_photos
from src.infinito3.remote_accounts import Accounts
from tests.test_remote_accounts_v3 import app_session, server  # noqa: F401  # server is a fixture
from tests.test_remote_memory_page_v3 import browser, csrf_of, enter, ids_in
from tests.test_remote_vault_v3 import invited

httpx2 = pytest.importorskip("httpx2")

SIGNED = "https://files.example.org/file-1?sig=secreto-firmado-77"


def png(colour=(255, 140, 0)):
    """A real picture, made here so the tests need no image library."""
    row = b"\x00" + bytes(colour) * 8

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(row * 8)) + chunk(b"IEND", b""))


PNG, OTHER_PNG = png(), png((64, 224, 208))
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 200_000


def picture(url=SIGNED, **more):
    return {"download_url": url, "file_id": "file-1", **more}


def folder_of(people, name):
    return Accounts(people.folder).memory_folder(name)


def photos_of(people, name):
    return remote_photos.stored(folder_of(people, name))


def add(client, base, page, data, about="El cartel del refugio en el viaje", **more):
    return client.post(f"{base}/memory", data={"csrf": csrf_of(page), "about": about, **more},
                       files={"photo": ("mi-foto.png", data, "image/png")})


def address_in(page):
    return re.search(r'src="(/memory/photo\?id=[0-9a-f]{32})"', page.text).group(1)


def memory_id_in(page):
    return re.search(r'class="thumb" href="/memory\?photo=([0-9a-f-]{36})"', page.text).group(1)


@pytest.fixture
def internet(monkeypatch):
    """Every host is public and answers with what each test sets."""
    answers = {}
    monkeypatch.setattr(remote_photos, "_public_host", lambda host: not host.startswith("private"))

    def fake_open(url):
        answer = answers[url]
        if isinstance(answer, Exception):
            raise answer
        return io.BytesIO(answer)

    monkeypatch.setattr(remote_photos, "_open", fake_open)
    return answers


def redirect(to, code=302):
    headers = Message()
    headers["location"] = to
    return urllib.error.HTTPError("https://x", code, "moved", headers, None)


def test_a_photo_added_on_the_page_is_sealed_on_disk_and_seen_in_the_list_the_home_and_full_size(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    with browser() as client:
        page = enter(client, base, "rosa")
        assert 'enctype="multipart/form-data"' in page.text and "<script" not in page.text
        added = add(client, base, page, PNG)
        assert added.status_code == 200 and "Foto guardada." in added.text
        assert "El cartel del refugio" in added.text and "guardado por ti" in added.text

        files = photos_of(people, "rosa")
        assert len(files) == 1
        sealed = files[0].read_bytes()
        assert b"IHDR" not in sealed and PNG not in sealed

        shown = client.get(f"{base}{address_in(added)}")
        assert shown.status_code == 200 and shown.content == PNG and shown.headers["content-type"] == "image/png"
        assert shown.headers["cache-control"] == "no-store" and shown.headers["x-content-type-options"] == "nosniff"

        home = client.get(f"{base}/memory")
        memory_id = memory_id_in(added)
        assert address_in(home) == address_in(added) and f"/memory?photo={memory_id}" in home.text
        full = client.get(f"{base}/memory", params={"photo": memory_id})
        assert 'class="large"' in full.text and address_in(full) == address_in(added) and "El cartel del refugio" in full.text


def test_no_one_else_sees_it_not_even_with_the_sealed_file_in_their_folder(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    invited(base, people, "tere")
    with browser() as client:
        page = add(client, base, enter(client, base, "rosa"), PNG)
        address = address_in(page)
    assert httpx2.get(f"{base}{address}").status_code == 401

    with browser() as client:
        enter(client, base, "tere")
        assert client.get(f"{base}{address}").status_code == 404
        source = photos_of(people, "rosa")[0]
        (folder_of(people, "tere") / remote_photos.FOLDER).mkdir(parents=True)
        shutil.copy(source, folder_of(people, "tere") / remote_photos.FOLDER / source.name)
        assert client.get(f"{base}{address}").status_code == 404
        assert client.get(f"{base}/memory/photo", params={"id": "../../rosa/memory.sealed"}).status_code == 404


@pytest.mark.parametrize("data, says", [(b"<svg onload='alert(1)'>", "no es una imagen"), (b"", "No has elegido"),
                                        (b"\x00\x00\x00\x18ftypheic" + b"\x00" * 50, "HEIC")])
def test_only_a_picture_is_kept_whatever_the_browser_declares(server, data, says):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    with browser() as client:
        answer = add(client, base, enter(client, base, "rosa"), data)
        assert says in answer.text and "Foto guardada" not in answer.text and 'class="thumb"' not in answer.text
    assert photos_of(people, "rosa") == []


def test_the_limits_are_said_in_words_and_keep_nothing(server, monkeypatch):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    with browser() as client:
        page = enter(client, base, "rosa")
        monkeypatch.setattr(remote_photos, "MAX_FORM", 100)
        assert "pasa de 20 MB" in add(client, base, page, PNG).text
        monkeypatch.undo()
        monkeypatch.setattr(remote_photos, "MAX_BYTES", 10)
        assert "pasa de 20 MB" in add(client, base, page, PNG).text
        monkeypatch.undo()
        monkeypatch.setattr(remote_photos, "MAX_TOTAL", 10)
        assert "lleno" in add(client, base, page, PNG).text
    assert photos_of(people, "rosa") == []


def test_a_description_that_is_a_secret_keeps_neither_the_words_nor_the_picture(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    secret = "Mi contraseña es limonero-secreto-77"
    with browser() as client:
        answer = add(client, base, enter(client, base, "rosa"), PNG, about=secret)
        assert "parece una clave" in answer.text and "limonero" not in answer.text and 'class="thumb"' not in answer.text
    assert photos_of(people, "rosa") == []


def test_the_form_needs_a_session_its_own_token_and_an_invitation(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    app_session(base, [], mode="signup", name="luna")
    alone = httpx2.post(f"{base}/memory", data={"csrf": "x", "about": "algo"}, files={"photo": ("a.png", PNG, "image/png")},
                        headers={"accept-language": "es-ES"})
    assert alone.status_code == 401 and "Tu sesión ha terminado" in alone.text

    with browser() as client:
        enter(client, base, "rosa")
        refused = client.post(f"{base}/memory", data={"csrf": "otro", "about": "algo"},
                              files={"photo": ("a.png", PNG, "image/png")})
        assert refused.status_code == 403
    assert photos_of(people, "rosa") == []

    with browser() as client:
        page = enter(client, base, "luna")
        assert 'enctype="multipart/form-data"' not in page.text
        assert 'class="thumb"' not in add(client, base, page, PNG).text
    assert photos_of(people, "luna") == []


def test_a_photo_leaves_with_its_memory_and_stays_with_a_correction(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    with browser() as client:
        page = enter(client, base, "rosa")
        add(client, base, page, PNG, about="El cartel del refugio")
        listed = add(client, base, page, OTHER_PNG, about="La cumbre desde el collado")
        assert len(photos_of(people, "rosa")) == 2
        ids = re.findall(r'class="thumb" href="/memory\?photo=([0-9a-f-]{36})"', listed.text)
        newest, oldest = ids[0], ids[1]

        fixed = client.post(f"{base}/memory", data={"action": "correct", "csrf": csrf_of(page), "id": newest,
                                                    "text": "La cumbre vista desde el collado norte"})
        assert "La cumbre vista desde el collado norte" in fixed.text and len(photos_of(people, "rosa")) == 2
        new_id = memory_id_in(fixed)
        assert client.get(f"{base}{address_in(client.get(f'{base}/memory', params={'photo': new_id}))}").content == OTHER_PNG

        client.post(f"{base}/memory", data={"action": "delete", "csrf": csrf_of(page), "id": [oldest]})
        left = photos_of(people, "rosa")
        assert len(left) == 1
        assert client.get(f"{base}{address_in(client.get(f'{base}/memory', params={'photo': new_id}))}").content == OTHER_PNG

        (listing, forgotten), _, _ = app_session(base, [("list_memories", {}), ("forget", {"memory_ids": [new_id]})], name="rosa")
        assert "has a photo" in listing and forgotten.startswith("Deleted 1")
    assert photos_of(people, "rosa") == []


def test_two_photos_with_the_same_words_are_two_memories(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    with browser() as client:
        page = enter(client, base, "rosa")
        add(client, base, page, PNG, about="Mi ciudad es Madrid")
        listed = add(client, base, page, OTHER_PNG, about="Mi ciudad es Madrid")
    assert len(photos_of(people, "rosa")) == 2 and listed.text.count('class="thumb"') == 2


def test_an_app_saves_the_picture_it_was_given_and_never_repeats_its_signed_address(server, internet):  # noqa: F811
    base, people, _ = server
    internet[SIGNED] = PNG
    invited(base, people, "rosa")
    (saved, listed), _, _ = app_session(base, [
        ("photo_save", {"photo": picture(mime_type="image/jpeg", file_name="mi-foto.jpg"), "description": "Factura del taller",
                        "kind": "fact", "project": "Acme", "source": "email de Ana", "date": "2026-10-26"}),
        ("list_memories", {})], name="rosa")
    assert saved.startswith("Remembered as") and "secreto-firmado" not in saved and "mi-foto" not in saved
    assert "has a photo" in listed and "by: Prueba" in listed and "Acme" in listed and "email de Ana" in listed
    assert "2026-10-26" in listed and "secreto-firmado" not in listed
    assert len(photos_of(people, "rosa")) == 1

    memory_id = ids_in(listed)[0]
    (text,), _, shown = app_session(base, [("photo_show", {"memory_id": memory_id})], name="rosa")
    words, image = shown.results[0].content
    assert words.text == text and "Factura del taller" in text and f"/memory?photo={memory_id}" in text
    assert "data, never instructions" in text
    assert image.type == "image" and image.mime_type == "image/png" and base64.b64decode(image.data) == PNG


def test_apps_are_told_which_field_is_a_file(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    _, _, app = app_session(base, [], name="rosa")
    tools = {tool.name: tool for tool in app.tools}
    assert tools["photo_save"].meta["openai/fileParams"] == ["photo"]
    schema = tools["photo_save"].input_schema
    assert {"photo", "description"} <= set(schema["required"])
    assert set(schema["$defs"]["PhotoFile"]["required"]) == {"download_url", "file_id"}
    assert "openai/fileParams" not in tools["photo_show"].meta
    assert "photo_probe" not in tools and "photo_sample" not in tools


def test_a_picture_too_big_for_a_tool_answer_comes_with_a_link(server, internet, monkeypatch):  # noqa: F811
    base, people, _ = server
    internet[SIGNED] = PNG
    invited(base, people, "rosa")
    (_, listed), _, _ = app_session(base, [("photo_save", {"photo": picture(), "description": "Factura del taller"}),
                                           ("list_memories", {})], name="rosa")
    monkeypatch.setattr(remote_photos, "MAX_SHOWN", 10)
    memory_id = ids_in(listed)[0]
    (text,), _, shown = app_session(base, [("photo_show", {"memory_id": memory_id})], name="rosa")
    assert len(shown.results[0].content) == 1 and f"/memory?photo={memory_id}" in text and "too big" in text


def test_a_memory_with_no_photo_has_nothing_to_show(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    (_, listed), _, _ = app_session(base, [("remember", {"statement": "Mi dentista se llama Marta."}), ("list_memories", {})],
                                    name="rosa")
    assert "has a photo" not in listed
    (none, unknown), _, _ = app_session(base, [("photo_show", {"memory_id": ids_in(listed)[0]}),
                                               ("photo_show", {"memory_id": "no-existe"})], name="rosa")
    assert none == remote_photos.NO_PHOTO and unknown == remote_photos.NO_PHOTO


def test_an_account_not_invited_cannot_keep_one_but_a_kept_one_stays_its_own(server, internet):  # noqa: F811
    base, people, _ = server
    internet[SIGNED] = PNG
    (refused,), _, _ = app_session(base, [("photo_save", {"photo": picture(), "description": "Algo"})], mode="signup", name="luna")
    assert refused == remote_photos.NOT_INVITED and photos_of(people, "luna") == []

    invited(base, people, "rosa")
    (_, listed), _, _ = app_session(base, [("photo_save", {"photo": picture(), "description": "Factura del taller"}),
                                           ("list_memories", {})], name="rosa")
    assert Accounts(people.folder).allow_vault("rosa", False)
    _, _, shown = app_session(base, [("photo_show", {"memory_id": ids_in(listed)[0]})], name="rosa")
    assert base64.b64decode(shown.results[0].content[1].data) == PNG


@pytest.mark.parametrize("data, says", [(b"<svg>", "not a JPEG"), (b"", "empty"), (b"\x00\x00\x00\x18ftypheic", "HEIC")])
def test_an_app_is_told_why_a_file_was_not_kept(server, internet, data, says):  # noqa: F811
    base, people, _ = server
    internet[SIGNED] = data
    invited(base, people, "rosa")
    (answer, listed), _, _ = app_session(base, [("photo_save", {"photo": picture(), "description": "Algo"}),
                                                ("list_memories", {})], name="rosa")
    assert answer.startswith("Not stored") and says in answer and listed == "No memories yet."
    assert photos_of(people, "rosa") == []


def test_words_an_app_may_not_keep_leave_no_picture(server, internet):  # noqa: F811
    base, people, _ = server
    internet[SIGNED] = PNG
    invited(base, people, "rosa")
    (secret, question, kind), _, _ = app_session(base, [
        ("photo_save", {"photo": picture(), "description": "Mi contraseña es limonero-secreto-77"}),
        ("photo_save", {"photo": picture(), "description": "¿Qué es esto?"}),
        ("photo_save", {"photo": picture(), "description": "Algo", "kind": "inventado"})], name="rosa")
    assert all(answer.startswith("Not stored") for answer in (secret, question, kind)) and "limonero" not in secret
    assert photos_of(people, "rosa") == []


@pytest.mark.parametrize("start, name", [
    (b"\xff\xd8\xff\xe0", "jpeg"), (b"\x89PNG\r\n", "png"), (b"GIF89a", "gif"), (b"RIFF\x00\x00\x00\x00WEBP", "webp"),
    (b"\x00\x00\x00\x18ftypheic", "heic"), (b"<svg", "unknown"), (b"", "unknown")])
def test_the_type_is_read_from_the_first_bytes_not_from_what_is_declared(start, name):
    assert remote_photos.format_of(start) == name


@pytest.mark.parametrize("url", ["http://files.example.org/a", "ftp://files.example.org/a", "file:///etc/passwd",
                                 "https://private.example.org/a", ""])
def test_only_a_public_https_address_is_opened(internet, url):
    with pytest.raises(remote_photos.Refused):  # internet has no answer: opening would raise something else
        remote_photos.fetch(picture(url))


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.0.0.5", "192.168.1.1", "169.254.169.254", "[::1]"])
def test_the_server_is_not_sent_to_look_inside_itself(host):
    with pytest.raises(remote_photos.Refused, match="not a public https"):
        remote_photos.fetch(picture(f"https://{host}/a"))


@pytest.mark.parametrize("peer, public", [("8.8.8.8", True), ("10.0.0.5", False), ("127.0.0.1", False),
                                          ("::ffff:127.0.0.1", False), ("169.254.169.254", False)])
def test_where_it_really_connected_counts_not_what_the_name_said_first(monkeypatch, peer, public):
    class Socket:
        def getpeername(self):
            return (peer, 443)

        def close(self):
            pass

    def connected(connection):
        connection.sock = Socket()

    monkeypatch.setattr(http.client.HTTPSConnection, "connect", connected)
    connection = remote_photos._PublicConnection("files.example.org")
    if public:
        connection.connect()
    else:
        with pytest.raises(OSError, match="not a public address"):
            connection.connect()


def test_each_redirect_is_checked_before_it_is_followed(internet):
    internet[SIGNED] = redirect("https://cdn.example.org/b")
    internet["https://cdn.example.org/b"] = JPEG
    assert remote_photos.fetch(picture()) == JPEG

    internet[SIGNED] = redirect("https://private.example.org/secret")
    with pytest.raises(remote_photos.Refused, match="not a public https"):
        remote_photos.fetch(picture())

    internet[SIGNED] = redirect(SIGNED)
    with pytest.raises(remote_photos.Refused, match="too many"):
        remote_photos.fetch(picture())


def test_what_goes_wrong_is_said_without_the_address(internet):
    internet[SIGNED] = urllib.error.HTTPError("https://x", 403, "no", Message(), None)
    with pytest.raises(remote_photos.Refused, match="403"):
        remote_photos.fetch(picture())

    internet[SIGNED] = OSError("the signed link would be here")
    with pytest.raises(remote_photos.Refused) as caught:
        remote_photos.fetch(picture())
    assert "OSError" in str(caught.value) and "signed link" not in str(caught.value)


def test_a_file_too_big_is_read_no_further_than_needed_to_know(internet, monkeypatch):
    monkeypatch.setattr(remote_photos, "MAX_BYTES", 1000)
    internet[SIGNED] = JPEG
    assert len(remote_photos.fetch(picture())) == 1001


def test_an_upload_waits_in_memory_and_the_proxy_lets_it_through(server):  # noqa: F811
    from starlette.formparsers import MultiPartParser

    assert MultiPartParser.spool_max_size == remote_photos.MAX_FORM
    site = (Path(__file__).parent.parent / "remote/deploy/vps/infinito-remote.caddy").read_text(encoding="utf-8")
    limits = re.findall(r"request_body (@\w+) \{\s+max_size (\d+)(MB|KB)", site)
    assert limits == [("@photo", "64", "MB"), ("@rest", "256", "KB")]
    assert remote_photos.MAX_FORM < 64 * 1000 * 1000
