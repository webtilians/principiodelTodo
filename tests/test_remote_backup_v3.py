"""Copies of the sealed memories: nobody can read them, and they can really be restored."""
import json
import os
import tarfile

import pytest

from src.infinito3.remote import People, SealedMemories, main
from src.infinito3.remote_backup import backup, restore, verify

pytest.importorskip("cryptography")
PASSWORD = "mi casa tiene un limonero"


@pytest.fixture
def server(tmp_path):
    """A server's data folder with one account and one person with a private address, each with a memory."""
    from src.infinito3.remote_accounts import Accounts

    data = tmp_path / "server"
    people, accounts = People(data), Accounts(data)
    token = people.add("luis")
    SealedMemories(people).use("luis", people.key("luis", token), lambda m: m.remember("Vivo en Cádiz."), writes=True)
    key, code = accounts.create("ana", PASSWORD)
    SealedMemories(accounts).use("ana", key, lambda m: m.remember("Mi perra se llama Nena."), writes=True)
    return data, token, code


def memories_of_ana(data, password=PASSWORD):
    from src.infinito3.remote_accounts import Accounts

    accounts = Accounts(data)
    return SealedMemories(accounts).use("ana", accounts.open("ana", password), lambda m: m.list_memories(), writes=False)


def only_copy(folder):
    [copy] = list(folder.iterdir())
    return copy


def test_after_losing_everything_a_copy_brings_every_memory_back(server, tmp_path):
    data, token, code = server
    backup(data, tmp_path / "copies")
    restored = tmp_path / "another-computer"
    assert "Restored 2 sealed memories" in restore(only_copy(tmp_path / "copies"), restored)
    assert "Nena" in memories_of_ana(restored)  # with the same password
    people = People(restored)
    listed = SealedMemories(people).use("luis", people.key("luis", token), lambda m: m.list_memories(), writes=False)
    assert "Cádiz" in listed  # and the same address
    from src.infinito3.remote_accounts import Accounts

    accounts = Accounts(restored)
    assert accounts.recover("ana", code, "otra frase que sí recuerdo")  # and the same recovery code


def test_nothing_readable_goes_into_a_copy(server, tmp_path):
    data, _, code = server
    folder = data / "people" / "viejo"
    folder.mkdir(parents=True)
    (folder / "memory.db").write_text("Mi clave secreta en claro")  # a plain memory from before sealing
    (data / "grants.db").write_text("the apps' sign-ins")
    backup(data, tmp_path / "copies")
    with tarfile.open(only_copy(tmp_path / "copies")) as archive:
        names = archive.getnames()
        everything = b"".join(archive.extractfile(m).read() for m in archive.getmembers())
    assert "grants.db" not in names and not any(name.endswith("memory.db") for name in names)
    for secret in (b"Nena", "Cádiz".encode(), b"en claro", PASSWORD.encode(), code.encode()):
        assert secret not in everything


def test_a_restore_moves_what_was_there_aside_and_signs_apps_out(server, tmp_path):
    data, _, _ = server
    backup(data, tmp_path / "copies")
    from src.infinito3.remote_accounts import Accounts

    accounts = Accounts(data)
    SealedMemories(accounts).use("ana", accounts.open("ana", PASSWORD), lambda m: m.remember("Me mudé a Sevilla."),
                                 writes=True)
    (data / "grants.db").write_text("the apps' sign-ins")
    answer = restore(only_copy(tmp_path / "copies"), data)
    assert "Each app has to sign in again" in answer and not (data / "grants.db").exists()
    assert "Sevilla" not in memories_of_ana(data)  # back as it was
    [aside] = list(data.glob("before-restore-*"))
    assert "Sevilla" in memories_of_ana(aside)  # and nothing was lost


def test_a_damaged_copy_is_found_out_and_changes_nothing(server, tmp_path):
    data, _, _ = server
    backup(data, tmp_path / "copies")
    copy = only_copy(tmp_path / "copies")
    damaged = tmp_path / "damaged.tar.gz"
    with tarfile.open(copy) as source, tarfile.open(damaged, "w:gz") as target:
        for member in source.getmembers():
            content = source.extractfile(member)
            if member.name.endswith("ana/memory.sealed"):
                import io

                body = bytearray(content.read())
                body[-1] ^= 1
                member.size = len(body)
                content = io.BytesIO(bytes(body))
            target.addfile(member, content)
    with pytest.raises(SystemExit, match="damaged"):
        verify(damaged)
    with pytest.raises(SystemExit, match="Nothing was changed"):
        restore(damaged, data)
    assert "Nena" in memories_of_ana(data)
    truncated = tmp_path / "truncated.tar.gz"
    truncated.write_bytes(copy.read_bytes()[:40])
    with pytest.raises(SystemExit, match="not usable|NOT usable"):
        verify(truncated)
    assert "2 sealed memories" in verify(copy)


def test_a_copy_cannot_write_outside_the_data_folder(tmp_path):
    import io

    evil = tmp_path / "evil.tar.gz"
    with tarfile.open(evil, "w:gz") as archive:
        for name, body in (("MANIFEST.json", b'{"files": {}}'), ("../../outside.txt", b"x")):
            member = tarfile.TarInfo(name)
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    with pytest.raises(SystemExit, match="unexpected entry"):
        restore(evil, tmp_path / "data")
    assert not (tmp_path / "outside.txt").exists()


def test_only_the_newest_copies_are_kept(server, tmp_path, monkeypatch):
    import src.infinito3.remote_backup as module
    from datetime import datetime, timedelta, timezone

    data, _, _ = server
    folder = tmp_path / "copies"
    start = datetime(2026, 11, 1, tzinfo=timezone.utc)
    for day in range(5):
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return start + timedelta(days=day)

        monkeypatch.setattr(module, "datetime", Clock)
        backup(data, folder, keep=3)
    (folder / "my-notes.txt").write_text("not a copy")
    names = sorted(p.name for p in folder.iterdir())
    assert names == ["infinito-20261103-000000.tar.gz", "infinito-20261104-000000.tar.gz",
                     "infinito-20261105-000000.tar.gz", "my-notes.txt"]


def test_the_commands(server, tmp_path, monkeypatch, capsys):
    data, _, _ = server
    monkeypatch.setenv("INFINITO_REMOTE_DIR", str(data))
    main(["backup", str(tmp_path / "copies"), "--keep", "2"])
    assert "2 sealed memories" in capsys.readouterr().out
    copy = only_copy(tmp_path / "copies")
    main(["verify-backup", str(copy)])
    assert "is whole" in capsys.readouterr().out
    monkeypatch.setenv("INFINITO_REMOTE_DIR", str(tmp_path / "new"))
    main(["restore", str(copy)])
    assert "Restored 2" in capsys.readouterr().out
    assert "Nena" in memories_of_ana(tmp_path / "new")
    assert json.loads((tmp_path / "new" / "accounts.json").read_text())["ana"] and os.path.exists(tmp_path / "new" / "people.json")
