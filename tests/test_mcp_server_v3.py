"""INFINITO as an agent's memory: the four MCP tools, directly and over a real stdio MCP session."""
import json
import os
import re
import sys
from pathlib import Path

import pytest

from src.infinito3.mcp_server import INSTRUCTIONS, TOOL_DESCRIPTIONS, VAULT_INSTRUCTIONS, AgentMemory

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def memory(tmp_path):
    agent_memory = AgentMemory(str(tmp_path / "memory.db"))
    yield agent_memory
    agent_memory.close()


def ids(text):
    return re.findall(r"\[([0-9a-f-]{36})\]", text)


def test_remember_and_recall_keep_the_users_words_dated(memory):
    assert memory.remember("I live in Valencia.").startswith("Remembered as ")
    memory.remember("I work as a nurse.")
    recalled = memory.recall("where do I live")
    assert '"I live in Valencia."' in recalled and '"I work as a nurse."' in recalled
    assert "oldest first" in recalled


def test_secrets_questions_and_forget_requests_are_not_stored(memory):
    assert "a password" in memory.remember("My bank password is hunter2")
    assert "an identity document number" in memory.remember("Mi DNI es 12345678Z")
    assert "not memories" in memory.remember("Where do I live?")
    assert "asks to forget" in memory.remember("Forget where I live.")
    assert memory.list_memories() == "No memories yet."


def test_a_refused_secret_says_where_it_goes_and_is_never_repeated(memory):
    answer = memory.remember("La clave del wifi es limonero2024")
    assert "secrets never go in memory" in answer and "vault_add" in answer and "password manager" in answer
    assert "cannot be enabled from here" in answer  # ChatGPT asked to enable a tool that lives on the computer
    assert "limonero2024" not in answer
    assert memory.list_memories() == "No memories yet."


@pytest.mark.parametrize("statement", [
    "La contraseña del router está en la bóveda, en la entrada Router de casa.",
    "The router password is in the vault under Router de casa.",
    "The router key is in the vault as Router de casa.",
    "My bank password is stored in my password manager.",
])
def test_where_a_secret_is_kept_can_be_remembered(memory, statement):
    assert memory.remember(statement).startswith("Remembered")


@pytest.mark.parametrize("statement", [
    "The wifi password is invault2024", "My bank password is in2deep!", "The router key is keepass99",
    "The router password is in the vault: hunter2",
])
def test_a_value_is_still_refused_though_it_names_a_place(memory, statement):
    assert "secrets never go in memory" in memory.remember(statement)
    assert memory.list_memories() == "No memories yet."


def test_the_instructions_send_secrets_to_the_vault_and_stay_short():
    assert "refuses them" in INSTRUCTIONS and "vault_add" in INSTRUCTIONS and "password manager" in INSTRUCTIONS
    assert "use these tools" in VAULT_INSTRUCTIONS
    assert "vault_add" in TOOL_DESCRIPTIONS["remember"]
    assert len(INSTRUCTIONS) <= 2048  # every app loads this in every session


def test_a_statement_that_only_mentions_deleting_is_stored(memory):
    assert memory.remember("Decidimos eliminar duplicados sin tocar los originales.").startswith("Remembered")
    assert "asks to forget" in memory.remember("Olvida mi edad.")


def test_a_new_value_supersedes_the_old_one(memory):
    memory.remember("Me llamo Ana.")
    assert "replaces an earlier value" in memory.remember("Me llamo Lucía.")
    listed = memory.list_memories()
    assert '"Me llamo Lucía."' in listed and '"Me llamo Ana."' not in listed


def test_forget_deletes_for_good_and_keeps_what_was_not_asked(memory):
    memory.remember("Me llamo Ana, tengo 34 años y trabajo de arquitecta.")
    memory.remember("Mi hermano se llama Pablo.")
    target = [i for i, line in zip(ids(memory.list_memories()), memory.list_memories().splitlines()[1:])
              if "34 años" in line]
    result = memory.forget(target, keep=["Me llamo Ana y trabajo de arquitecta."])
    assert result.startswith("Deleted 1 memories")
    assert "anywhere else" in result  # the agent's own notes must forget it too
    listed = memory.list_memories()
    assert "34 años" not in listed  # ids are random hex, and one may contain 34
    assert '"Me llamo Ana y trabajo de arquitecta."' in listed and '"Mi hermano se llama Pablo."' in listed


def test_forget_reports_unknown_ids(memory):
    assert "Unknown ids" in memory.forget(["not-an-id"])


def test_the_server_works_over_mcp_stdio(tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    anyio = pytest.importorskip("anyio")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.infinito3.mcp_server"], cwd=str(REPO),
        env={**os.environ, "INFINITO_MEMORY_DB": str(tmp_path / "memory.db")})

    async def session():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                await client.initialize()
                tools = sorted(t.name for t in (await client.list_tools()).tools)
                await client.call_tool("remember", {"statement": "I live in Leeds."})
                await client.call_tool("remember", {"statement": "I work as a data engineer."})
                listed = (await client.call_tool("list_memories", {})).content[0].text
                leeds = [i for i in ids(listed) if f"[{i}]" in listed.split("Leeds")[0].splitlines()[-1]]
                forgot = (await client.call_tool("forget", {"memory_ids": leeds})).content[0].text
                recalled = (await client.call_tool("recall", {"query": "where do I live"})).content[0].text
                return tools, forgot, recalled

    tools, forgot, recalled = anyio.run(session)
    memory_tools = ["briefing", "forget", "list_memories", "overview", "recall", "remember", "remember_many"]
    try:
        import pykeepass  # noqa: F401
        vault_tools = ["vault_add", "vault_list", "vault_lock", "vault_reveal", "vault_status", "vault_unlock"]
    except ImportError:
        vault_tools = []
    assert tools == sorted(memory_tools + vault_tools)
    assert forgot.startswith("Deleted 1 memories")
    assert "Leeds" not in recalled and "data engineer" in recalled


def test_with_the_memory_on_a_server_this_computer_serves_only_the_vault(tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    pytest.importorskip("pykeepass")
    anyio = pytest.importorskip("anyio")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "src.infinito3.mcp_server"], cwd=str(REPO),
        env={**os.environ, "INFINITO_LOCAL_MEMORY": "off", "INFINITO_VAULT": str(tmp_path / "boveda.kdbx")})

    async def session():
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as client:
                info = await client.initialize()
                return info.server_info.name, sorted(t.name for t in (await client.list_tools()).tools)

    name, tools = anyio.run(session)
    assert name == "infinito-vault"
    assert tools == ["vault_add", "vault_list", "vault_lock", "vault_reveal", "vault_status", "vault_unlock"]


def test_recall_finds_a_memory_that_shares_no_word_with_the_question(memory):
    # A memory grown past a few dozen facts, as one did within a day of real use.
    for n in range(60):
        memory.remember(f"Mi proyecto número {n} usa la librería {n} para procesar datos.")
    memory.remember("Ahora mismo mis canciones favoritas son Iris de Goo Goo Dolls y Duvet de bôa.")
    assert "Iris de Goo Goo Dolls" in memory.recall("¿Qué música me gusta?")
