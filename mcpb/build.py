"""Build the Claude Desktop extension (an MCP Bundle): python mcpb/build.py

Writes dist/infinito-memory.mcpb, a zip with the manifest, the extension's
pyproject.toml, its icon and server/, where the INFINITO core is copied from
src/infinito3. Claude Desktop installs it with a double click and runs it with
uv, so users need no Python of their own. The manifest, the extension's
pyproject and the package must carry the same version.
"""
import json
import re
import shutil
import struct
import sys
import zipfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "mcpb"
DIST = ROOT / "dist"
STAGE = DIST / "mcpb-build"
OUTPUT = DIST / "infinito-memory.mcpb"


def version_of(pyproject: Path) -> str:
    return re.search(r'^version\s*=\s*"([^"]+)"', pyproject.read_text(encoding="utf-8"), re.M).group(1)


def icon_png(size: int = 512) -> bytes:
    """The INFINITO icon (as demo/static/icon.svg): a cobalt tile with three lines, the last one fading."""
    def inside_rounded(x, y, r):
        cx = min(max(x, r), size - 1 - r)
        cy = min(max(y, r), size - 1 - r)
        return (x - cx) ** 2 + (y - cy) ** 2 <= r * r

    scale = size / 32
    lines = [(9, 23, 11, 255), (9, 18, 16, 255), (9, 14, 21, 255), (19, 23, 21, 90)]  # x0, x1, y, alpha
    rows = []
    for y in range(size):
        row = bytearray([0])
        for x in range(size):
            if not inside_rounded(x, y, 7 * scale):
                row += bytes((0, 0, 0, 0))
                continue
            pixel = (42, 68, 212, 255)
            for x0, x1, ly, alpha in lines:
                px, py = x / scale, y / scale
                nx = min(max(px, x0), x1)
                if (px - nx) ** 2 + (py - ly) ** 2 <= 1.5 ** 2:
                    a = alpha / 255
                    pixel = tuple(round(c * (1 - a) + 255 * a) for c in pixel[:3]) + (255,)
            row += bytes(pixel)
        rows.append(bytes(row))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header)
            + chunk(b"IDAT", zlib.compress(b"".join(rows), 9)) + chunk(b"IEND", b""))


def main() -> int:
    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    versions = {"manifest.json": manifest["version"], "mcpb/pyproject.toml": version_of(HERE / "pyproject.toml"),
                "pyproject.toml": version_of(ROOT / "pyproject.toml")}
    if len(set(versions.values())) != 1:
        print(f"Versions differ: {versions}", file=sys.stderr)
        return 1

    shutil.rmtree(STAGE, ignore_errors=True)
    (STAGE / "server").mkdir(parents=True)
    for name in ("manifest.json", "pyproject.toml"):
        shutil.copy2(HERE / name, STAGE / name)
    shutil.copy2(HERE / "server" / "main.py", STAGE / "server" / "main.py")
    shutil.copytree(ROOT / "src" / "infinito3", STAGE / "server" / "infinito3",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (STAGE / "icon.png").write_bytes(icon_png())

    OUTPUT.unlink(missing_ok=True)
    with zipfile.ZipFile(OUTPUT, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                bundle.write(path, path.relative_to(STAGE).as_posix())
    print(f"{OUTPUT} ({OUTPUT.stat().st_size // 1024} KB), version {manifest['version']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
