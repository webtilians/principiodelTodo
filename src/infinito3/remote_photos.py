"""A trial, not a feature: what do the apps really do with a picture?

Keeping photos only makes sense if an app can hand one over and look at one again, and nothing we can rely on
says they do. ChatGPT documents file parameters of its own, and reports disagree on images coming back from a
tool. So, like the first step of the vault, this asks the real apps before anything is built:

- photo_probe declares its file the way ChatGPT documents and reports what reached the server.
- photo_sample answers with a small picture whose colours appear in no text, so an app can only name them if
  its model really saw it.

Nothing is kept. The file is read to count it and to look at its first bytes, then dropped, and the address it
came from is never repeated: it is a signed link. Both tools are by invitation, like the vault, and go away when
the photo tools arrive or the trial says there is no point."""
import ipaddress
import json
import socket
import struct
import urllib.error
import urllib.request
import zlib
from typing import Optional
from urllib.parse import urljoin, urlsplit

try:
    from typing_extensions import TypedDict
except ImportError:
    from typing import TypedDict

MAX_BYTES = 20 * 1024 * 1024
MAX_REDIRECTS = 3
TIMEOUT_SECONDS = 20
REDIRECTS = (301, 302, 303, 307, 308)

PROBE_DESCRIPTION = ("Trial, only when the person asks to test pictures: pass the picture they attached. Reports "
                     "whether the file reached INFINITO, its size and its type. Nothing is kept.")
SAMPLE_DESCRIPTION = ("Trial, only when the person asks to test pictures: answers with a small test picture. Say "
                      "exactly what you see in it; if no picture reached you, say so and do not guess.")
SAMPLE_TEXT = "Here is the test picture. Say what you see in it, or that no picture reached you."
NOT_INVITED = "The trial of pictures is by invitation and this account has none, which cannot be enabled from here."
HINTS = {"readOnlyHint": True, "openWorldHint": False}
PROBE_HINTS = {"readOnlyHint": True, "openWorldHint": True}  # it reads a file from the app that sends it

LEFT_COLOUR, RIGHT_COLOUR = (255, 140, 0), (64, 224, 208)  # orange and turquoise: never written in any answer


class _FileReference(TypedDict):
    download_url: str
    file_id: str


class PhotoFile(_FileReference, total=False):
    """What ChatGPT sends for a parameter listed in the tool's openai/fileParams."""

    mime_type: str
    file_name: str


def sample_png(width: int = 96, height: int = 48) -> bytes:
    """Two halves of colour, made here so the trial needs no image library."""
    row = b"\x00" + bytes(LEFT_COLOUR) * (width // 2) + bytes(RIGHT_COLOUR) * (width - width // 2)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * height))
            + chunk(b"IEND", b""))


def format_of(start: bytes) -> str:
    if start.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if start.startswith(b"\x89PNG"):
        return "png"
    if start.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if start[:4] == b"RIFF" and start[8:12] == b"WEBP":
        return "webp"
    if start[4:8] == b"ftyp":
        return "heic or other ftyp"
    return "unknown"


def _public_host(host: str) -> bool:
    """Only what the whole internet can reach: this server must not be sent to look inside itself."""
    try:
        found = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
    except (OSError, UnicodeError):
        return False
    return bool(found) and all(ipaddress.ip_address(address[4][0].split("%")[0]).is_global for address in found)


class _KeepRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # each hop is checked below before it is followed


_opener = urllib.request.build_opener(_KeepRedirects)


def _open(url: str):
    return _opener.open(urllib.request.Request(url, headers={"User-Agent": "INFINITO"}), timeout=TIMEOUT_SECONDS)


def inspect_file(photo: Optional[dict]) -> str:
    """What reached the server, as text for the app: never the address, never a byte of the file."""
    photo = photo or {}
    report = {"arrived": False, "hosts": [], "declared_type": photo.get("mime_type"),
              "has_file_id": bool(photo.get("file_id")), "has_file_name": bool(photo.get("file_name")), "kept": False}

    def finish(**found) -> str:
        return json.dumps({**report, **found})

    url = str(photo.get("download_url") or "")
    if not url:
        return finish(why="no download_url came with the picture")
    for _ in range(MAX_REDIRECTS + 1):
        parts = urlsplit(url)
        host = parts.hostname or ""
        report["hosts"].append(host)
        if parts.scheme != "https" or not _public_host(host):
            return finish(why="not a public https address")
        try:
            with _open(url) as answer:
                start, size = b"", 0
                while True:
                    piece = answer.read(64 * 1024)
                    if not piece:
                        break
                    start = start or piece[:16]
                    size += len(piece)
                    if size > MAX_BYTES:
                        return finish(why=f"bigger than {MAX_BYTES} bytes, stopped reading")
                return finish(arrived=True, bytes=size, format=format_of(start))
        except urllib.error.HTTPError as answer:
            location = answer.headers.get("location")
            if answer.code not in REDIRECTS or not location:
                return finish(why=f"the address answered {answer.code}")
            url = urljoin(url, location)
        except OSError as problem:
            return finish(why=type(problem).__name__)
    return finish(why="too many redirects")
