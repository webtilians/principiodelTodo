"""INFINITO over HTTP: private addresses, one memory per person, over a real streamable HTTP session."""
import json
import re
import socket
import threading
import time

import pytest

from src.infinito3.remote import People, main


def ids(text):
    return re.findall(r"\[([0-9a-f-]{36})\]", text)


def test_an_address_names_one_person_and_is_kept_only_as_a_hash(tmp_path):
    people = People(tmp_path)
    token = people.add("ana")
    assert people.who(token) == "ana" and people.who("x" * 43) is None
    assert token not in (tmp_path / "people.json").read_text()
    with pytest.raises(ValueError):
        people.add("ana")
    with pytest.raises(ValueError):
        people.add("Ana Pérez")


def test_removing_someone_revokes_the_address_and_deletes_the_memory(tmp_path):
    people = People(tmp_path)
    token = people.add("ana")
    folder = people.memory_folder("ana")
    folder.mkdir(parents=True)
    (folder / "memory.db").write_text("x")
    assert people.remove("ana")
    assert people.who(token) is None and not folder.exists()
    assert not people.remove("ana")


def test_the_command_prints_the_address_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INFINITO_REMOTE_DIR", str(tmp_path))
    monkeypatch.setenv("INFINITO_REMOTE_URL", "https://infinito.example.org/")
    main(["add", "ana"])
    address = capsys.readouterr().out.splitlines()[0]
    assert re.fullmatch(r"https://infinito\.example\.org/u/[A-Za-z0-9_-]{43}/mcp", address)
    main(["list"])
    assert capsys.readouterr().out.strip() == "ana"


def test_each_person_talks_to_their_own_memory_over_http(tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    anyio = pytest.importorskip("anyio")
    httpx2 = pytest.importorskip("httpx2")
    uvicorn = pytest.importorskip("uvicorn")
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    from src.infinito3.remote import build_app

    people = People(tmp_path)
    ana, luis = people.add("ana"), people.add("luis")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(build_app(people), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    base = f"http://127.0.0.1:{port}"

    async def call(token, *steps):
        async with streamable_http_client(f"{base}/u/{token}/mcp") as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = sorted(t.name for t in (await client.list_tools()).tools)
                answers = [(await client.call_tool(name, arguments)).content[0].text for name, arguments in steps]
                return tools, answers

    try:
        tools, _ = anyio.run(call, ana, ("remember", {"statement": "I live in Leeds."}),
                             ("remember", {"statement": "I work as a data engineer."}))
        _, [luis_sees] = anyio.run(call, luis, ("list_memories", {}))
        _, [listed] = anyio.run(call, ana, ("list_memories", {}))
        leeds = [i for i in ids(listed) if f"[{i}]" in listed.split("Leeds")[0].splitlines()[-1]]
        _, [forgot, recalled] = anyio.run(call, ana, ("forget", {"memory_ids": leeds}),
                                          ("recall", {"query": "where do I live"}))
        unknown = httpx2.post(f"{base}/u/{'x' * 43}/mcp", json={}).status_code
        health = httpx2.get(f"{base}/health").text
    finally:
        server.should_exit = True
        thread.join(timeout=10)

    assert tools == ["forget", "list_memories", "recall", "remember", "remember_many"]  # no vault from a server
    assert luis_sees == "No memories yet."
    assert forgot.startswith("Deleted 1 memories")
    assert "Leeds" not in recalled and "data engineer" in recalled
    assert unknown == 404 and health == "ok"
    # What is on disk is sealed: nothing a person said can be read there without their address.
    on_disk = b"".join(p.read_bytes() for p in (tmp_path / "people").rglob("*") if p.is_file())
    assert b"data engineer" not in on_disk and b"Leeds" not in on_disk


def _memory_of(people, name, token):
    from src.infinito3.remote import SealedMemories

    return SealedMemories(people), people.key(name, token)


def test_a_memory_is_sealed_and_only_its_address_opens_it(tmp_path):
    pytest.importorskip("cryptography")
    from cryptography.exceptions import InvalidTag

    people = People(tmp_path)
    token = people.add("ana")
    sealed, key = _memory_of(people, "ana", token)
    sealed.use("ana", key, lambda m: m.remember("Vivo en Leeds y trabajo de enfermera."), writes=True)
    files = [p.name for p in (tmp_path / "people" / "ana").iterdir()]
    assert files == ["memory.sealed"]
    assert b"Leeds" not in (tmp_path / "people" / "ana" / "memory.sealed").read_bytes()
    assert "Leeds" in sealed.use("ana", key, lambda m: m.list_memories(), writes=False)
    other = people.key("ana", people.add("luis"))  # the right person, the wrong address
    with pytest.raises(InvalidTag):
        sealed.use("ana", other, lambda m: m.list_memories(), writes=False)


def test_a_memory_written_before_sealing_is_sealed_and_wiped_at_first_use(tmp_path):
    pytest.importorskip("cryptography")
    from src.infinito3.mcp_server import AgentMemory

    people = People(tmp_path)
    token = people.add("ana")
    entries = json.loads((tmp_path / "people.json").read_text())
    del entries["ana"]["salt"]  # as written before sealing existed
    (tmp_path / "people.json").write_text(json.dumps(entries))
    folder = tmp_path / "people" / "ana"
    folder.mkdir(parents=True)
    plain = AgentMemory(str(folder / "memory.db"))
    plain.remember("Vivo en Leeds.")
    plain.close()
    sealed, key = _memory_of(people, "ana", token)
    listed = sealed.use("ana", key, lambda m: m.list_memories(), writes=False)
    assert '"Vivo en Leeds."' in listed
    assert sorted(p.name for p in folder.iterdir()) == ["memory.sealed"]


def test_this_computers_memory_can_be_copied_to_the_server(tmp_path, caplog):
    caplog.set_level("INFO")
    pytest.importorskip("mcp.server.mcpserver")
    uvicorn = pytest.importorskip("uvicorn")
    from src.infinito3.mcp_server import AgentMemory
    from src.infinito3.remote import build_app, copy_local

    local = AgentMemory(str(tmp_path / "local.db"))
    local.remember("Mi proyecto principal es INFINITO.")
    local.remember("Prefiero las explicaciones breves.")
    local.close()
    people = People(tmp_path / "server")
    token = people.add("enrique")
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(build_app(people), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        time.sleep(0.05)
    try:
        result = copy_local(f"http://127.0.0.1:{port}/u/{token}/mcp", str(tmp_path / "local.db"))
        again = copy_local(f"http://127.0.0.1:{port}/u/{token}/mcp", str(tmp_path / "local.db"))
    finally:
        server.should_exit = True
        thread.join(timeout=10)
    assert result == "Copied 2 of 2 memories."
    assert again == "Copied 0 of 2 memories; 2 were already there."  # copying twice duplicates nothing
    assert token not in caplog.text  # the address is never written to a log
    sealed, key = _memory_of(people, "enrique", token)
    listed = sealed.use("enrique", key, lambda m: m.list_memories(), writes=False)
    assert "INFINITO" in listed and "breves" in listed
