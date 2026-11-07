"""Copies of a remote INFINITO server's memories, which nobody can read and which can really be restored.

    infinito-remote backup <folder> [--keep 14]   # one more copy; the oldest beyond --keep are deleted
    infinito-remote verify-backup <file>          # is this copy whole?
    infinito-remote restore <file>                # puts a copy back, with the service stopped

A copy holds what is already sealed on the server's disk: each memory.sealed with its photos, guidelines and work
queue, each vault (a KeePass file locked under its master password, with the sealed list of its names), and the files
that hold each person's key wrapped under their password or address (accounts.json, people.json). Without the password
or the address a copy opens nothing, so it can be kept anywhere. What can be read in one is the usernames
and how large each memory and each photo is. A plain memory left from before sealing is never copied, and neither are
the apps' sign-ins (grants.db): after a restore each app simply signs in again. Photos make copies larger: up to
remote_photos.MAX_TOTAL per person in each one.

A forgotten memory stays, sealed, in the copies made before it was forgotten until they age out, so
restoring an old copy brings it back: keep few, and say how long (remote/README.md).

Every copy lists the checksum of each file. It is checked right after writing and before any restore, so
a damaged copy is found out when it can still be replaced.
"""
import hashlib
import io
import json
import os
import re
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

SEALED = b"INF1"  # how a sealed memory starts (remote.py)
MANIFEST = "MANIFEST.json"
NAME = re.compile(r"infinito-\d{8}-\d{6}\.tar\.gz")
KEY_FILES = ("accounts.json", "people.json")
KDBX_MAGIC = b"\x03\xd9\xa2\x9a"  # how a vault starts: a KeePass file, locked under its master password
PERSON_FILES = ("memory.sealed", "vault.kdbx", "vault-names.sealed", "tasks.sealed", "guidelines.sealed",
                "reflections.sealed", "photos/*.sealed")
# Every name a copy may hold, and so the only places a restore may write: a photo is named by its 32-character id.
MEMBER = re.compile(r"(accounts\.json|people\.json|(accounts|people)/[a-z0-9][a-z0-9-]*/"
                    r"(memory\.sealed|vault\.kdbx|vault-names\.sealed|tasks\.sealed|guidelines\.sealed|"
                    r"reflections\.sealed|"
                    r"photos/[0-9a-f]{32}\.sealed))")


def _checksum(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _files_to_copy(data_dir: Path) -> Dict[str, bytes]:
    files = {name: (data_dir / name).read_bytes() for name in KEY_FILES if (data_dir / name).exists()}
    for kept in PERSON_FILES:
        for path in sorted(data_dir.glob(f"*/*/{kept}")):
            relative = path.relative_to(data_dir).as_posix()
            if MEMBER.fullmatch(relative):
                files[relative] = path.read_bytes()
    return files


def backup(data_dir: Path, folder: Path, keep: int = 14) -> str:
    """Write one more copy of data_dir into folder, check it, and delete the oldest beyond keep."""
    if keep < 1:
        raise ValueError("Keep at least one copy.")
    data_dir, folder = Path(data_dir), Path(folder)
    files = _files_to_copy(data_dir)
    now = datetime.now(timezone.utc)
    manifest = {"created": now.isoformat(timespec="seconds"), "files": {n: _checksum(d) for n, d in files.items()}}
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / f"infinito-{now:%Y%m%d-%H%M%S}.tar.gz"
    temporary = target.with_name(target.name + ".tmp")
    with tarfile.open(temporary, "w:gz") as archive:
        for name, data in [(MANIFEST, json.dumps(manifest, indent=1).encode())] + sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = len(data), 0o600, int(now.timestamp())
            archive.addfile(member, io.BytesIO(data))
    os.chmod(temporary, 0o600)
    _read(temporary)  # a copy that cannot be read back is an error now, not on the day it is needed
    temporary.replace(target)
    copies = sorted(p for p in folder.iterdir() if NAME.fullmatch(p.name))
    for old in copies[:-keep]:
        old.unlink()
    memories = sum(name.endswith("memory.sealed") for name in files)
    return f"Saved {target} with {memories} sealed memories; keeping {min(len(copies), keep)} copies."


def _read(path: Path) -> Dict[str, bytes]:
    """The files of a copy, each checked against its checksum. Raises ValueError if anything is wrong."""
    files: Dict[str, bytes] = {}
    try:
        with tarfile.open(path, "r:gz") as archive:
            for member in archive.getmembers():
                # Read by hand, by a fixed list of names: nothing in a copy chooses where it is written.
                if not member.isfile() or not (member.name == MANIFEST or MEMBER.fullmatch(member.name)):
                    raise ValueError(f"unexpected entry {member.name!r}")
                files[member.name] = archive.extractfile(member).read()
    except (OSError, EOFError, tarfile.TarError) as error:
        raise ValueError(f"cannot be read ({error})") from error
    if MANIFEST not in files:
        raise ValueError("it has no list of files")
    listed = json.loads(files.pop(MANIFEST))["files"]
    if set(listed) != set(files):
        raise ValueError("its files do not match its list")
    for name, data in files.items():
        if _checksum(data) != listed[name]:
            raise ValueError(f"{name} is damaged")
        if name.endswith("memory.sealed") and not data.startswith(SEALED):
            raise ValueError(f"{name} is not sealed")
        if name.endswith("vault.kdbx") and not data.startswith(KDBX_MAGIC):
            raise ValueError(f"{name} is not a KeePass file")
    return files


def verify(path: Path) -> str:
    try:
        files = _read(Path(path))
    except ValueError as error:
        raise SystemExit(f"{path} is NOT usable: {error}.") from error
    memories = sum(name.endswith("memory.sealed") for name in files)
    return f"{path} is whole: {memories} sealed memories."


def restore(path: Path, data_dir: Path) -> str:
    """Put a copy back. What was there moves aside to a dated folder, never deleted."""
    data_dir = Path(data_dir)
    try:
        files = _read(Path(path))
    except ValueError as error:
        raise SystemExit(f"Nothing was changed: {path} is not usable: {error}.") from error
    aside = data_dir / f"before-restore-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    moved: List[str] = []
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in KEY_FILES + ("grants.db", "grants.db-wal", "grants.db-shm", "accounts", "people"):
        if (data_dir / name).exists():
            aside.mkdir(mode=0o700, exist_ok=True)
            shutil.move(str(data_dir / name), str(aside / name))
            moved.append(name)
    for name, data in files.items():
        target = data_dir / name
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        target.write_bytes(data)
        os.chmod(target, 0o600)
    memories = sum(name.endswith("memory.sealed") for name in files)
    kept = f" What was there is in {aside}." if moved else ""
    return f"Restored {memories} sealed memories. Each app has to sign in again.{kept}"
