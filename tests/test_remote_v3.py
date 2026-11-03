"""INFINITO over HTTP: private addresses, one memory per person, over a real streamable HTTP session."""
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

    assert tools == ["forget", "list_memories", "recall", "remember"]  # no vault from a server
    assert luis_sees == "No memories yet."
    assert forgot.startswith("Deleted 1 memories")
    assert "Leeds" not in recalled and "data engineer" in recalled
    assert unknown == 404 and health == "ok"
