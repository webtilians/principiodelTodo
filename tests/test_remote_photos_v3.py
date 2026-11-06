"""The trial of pictures: it asks the real apps what they do with one, and it keeps nothing."""
import base64
import io
import json
import struct
import urllib.error
import zlib
from email.message import Message

import pytest

from src.infinito3 import remote_photos
from tests.test_remote_accounts_v3 import app_session, server  # noqa: F401  # server is a fixture
from tests.test_remote_vault_v3 import invited

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 200_000
SIGNED = "https://files.example.org/file-1?sig=secreto-firmado-77"


def picture(url=SIGNED, **more):
    return {"download_url": url, "file_id": "file-1", **more}


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


def test_the_sample_is_a_real_picture_with_its_colours_only_in_the_pixels():
    data = remote_photos.sample_png()
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    width, height = struct.unpack(">II", data[16:24])
    start = data.index(b"IDAT") + 4
    length = struct.unpack(">I", data[start - 8:start - 4])[0]
    pixels = zlib.decompress(data[start:start + length])
    row = pixels[:1 + width * 3]
    assert (width, height) == (96, 48) and len(pixels) == height * len(row)
    assert tuple(row[1:4]) == remote_photos.LEFT_COLOUR and tuple(row[-3:]) == remote_photos.RIGHT_COLOUR
    for text in (remote_photos.SAMPLE_TEXT, remote_photos.SAMPLE_DESCRIPTION):
        assert "orange" not in text.lower() and "turquoise" not in text.lower()


@pytest.mark.parametrize("start, name", [
    (b"\xff\xd8\xff\xe0", "jpeg"), (b"\x89PNG\r\n", "png"), (b"GIF89a", "gif"), (b"RIFF\x00\x00\x00\x00WEBP", "webp"),
    (b"\x00\x00\x00\x18ftypheic", "heic or other ftyp"), (b"<svg", "unknown"), (b"", "unknown")])
def test_the_type_is_read_from_the_first_bytes_not_from_what_the_app_declares(start, name):
    assert remote_photos.format_of(start) == name


def test_a_file_is_counted_and_dropped_and_its_signed_address_never_repeated(internet, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    internet[SIGNED] = JPEG
    text = remote_photos.inspect_file(picture(mime_type="image/png", file_name="mi-foto.jpg"))
    report = json.loads(text)
    assert report["arrived"] is True and report["bytes"] == len(JPEG) and report["format"] == "jpeg"
    assert report["declared_type"] == "image/png" and report["hosts"] == ["files.example.org"]
    assert report["kept"] is False and report["has_file_name"] is True
    assert "secreto-firmado" not in text and "mi-foto" not in text
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("url", ["http://files.example.org/a", "ftp://files.example.org/a", "file:///etc/passwd",
                                 "https://private.example.org/a", ""])
def test_only_a_public_https_address_is_opened(internet, url):
    report = json.loads(remote_photos.inspect_file(picture(url)))  # internet has no answer: opening would raise
    assert report["arrived"] is False and report["why"]


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "10.0.0.5", "192.168.1.1", "169.254.169.254", "[::1]"])
def test_the_server_is_not_sent_to_look_inside_itself(host):
    report = json.loads(remote_photos.inspect_file(picture(f"https://{host}/a")))
    assert report["arrived"] is False and report["why"] == "not a public https address"


def test_each_redirect_is_checked_before_it_is_followed(internet):
    internet[SIGNED] = redirect("https://cdn.example.org/b")
    internet["https://cdn.example.org/b"] = JPEG
    report = json.loads(remote_photos.inspect_file(picture()))
    assert report["arrived"] is True and report["hosts"] == ["files.example.org", "cdn.example.org"]

    internet[SIGNED] = redirect("https://private.example.org/secret")
    report = json.loads(remote_photos.inspect_file(picture()))
    assert report["arrived"] is False and report["hosts"][-1] == "private.example.org"

    internet[SIGNED] = redirect(SIGNED)
    assert json.loads(remote_photos.inspect_file(picture()))["why"] == "too many redirects"


def test_too_big_or_unreachable_files_are_only_reported(internet, monkeypatch):
    monkeypatch.setattr(remote_photos, "MAX_BYTES", 1000)
    internet[SIGNED] = JPEG
    report = json.loads(remote_photos.inspect_file(picture()))
    assert report["arrived"] is False and "bigger" in report["why"]

    internet[SIGNED] = urllib.error.HTTPError("https://x", 403, "no", Message(), None)
    assert "403" in json.loads(remote_photos.inspect_file(picture()))["why"]

    internet[SIGNED] = OSError("the signed link would be here")
    text = remote_photos.inspect_file(picture())
    assert json.loads(text)["why"] == "OSError" and "signed link" not in text


def test_an_invited_app_gets_the_picture_as_an_image_and_chatgpt_is_told_which_field_is_a_file(server):  # noqa: F811
    base, people, _ = server
    invited(base, people, "rosa")
    _, _, app = app_session(base, [("photo_sample", {})], name="rosa")
    text, image = app.results[0].content
    assert text.text == remote_photos.SAMPLE_TEXT
    assert image.type == "image" and image.mime_type == "image/png"
    assert base64.b64decode(image.data) == remote_photos.sample_png()

    tools = {tool.name: tool for tool in app.tools}
    assert tools["photo_probe"].meta["openai/fileParams"] == ["photo"]
    schema = tools["photo_probe"].input_schema
    assert "photo" in schema["required"]
    file_schema = schema["$defs"]["PhotoFile"]
    assert set(file_schema["required"]) == {"download_url", "file_id"}
    assert {"mime_type", "file_name"} <= set(file_schema["properties"])
    assert "openai/fileParams" not in tools["photo_sample"].meta


def test_an_account_not_invited_gets_neither_tool(server):  # noqa: F811
    base, _, _ = server
    (probe, sample), _, _ = app_session(base, [("photo_probe", {"photo": picture("https://127.0.0.1/a")}),
                                               ("photo_sample", {})], mode="signup", name="luna")
    assert probe == remote_photos.NOT_INVITED and sample == remote_photos.NOT_INVITED
