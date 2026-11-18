"""Post-v0.7 integration contracts for F10-S: genuine MCP processes, SQLite, and forgetting.

All values are synthetic. This intentionally does NOT claim atomicity between
separate HTTPS SealedMemories processes; that mechanism serializes within one
process, outside this SQLite-only fix.
"""
from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
import re
import sqlite3
import sys

import pytest

from src.infinito3.mcp_server import AgentMemory


def active_favorite_colors(db_path):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT content FROM memories WHERE status = 'active' "
            "AND fact_predicate = 'favorite_color'"
        ).fetchall()
    return [row[0] for row in rows]


def test_four_distinct_real_mcp_processes_share_atomic_sqlite(tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    db_path = tmp_path / "shared-across-four-mcp.db"
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "infinito3.mcp_server"],
        env={
            "INFINITO_MEMORY_DB": str(db_path),
            "INFINITO_LOCAL_MEMORY": "on",
        },
    )

    async def scenario():
        async with AsyncExitStack() as stack:
            sessions = []
            for i in range(4):
                read, write = await stack.enter_async_context(stdio_client(params))
                session = await stack.enter_async_context(ClientSession(read, write))
                await session.initialize()
                sessions.append(session)

            for round_index in range(12):
                values = [
                    f"Mi color favorito es turmalina{round_index}agente{agent}."
                    for agent in range(4)
                ]
                replies = await asyncio.gather(*[
                    session.call_tool("remember", {"statement": value})
                    for session, value in zip(sessions, values)
                ])
                assert all(
                    not r.is_error and r.content
                    and r.content[0].text.startswith("Remembered")
                    for r in replies
                ), (round_index, replies)

                current = active_favorite_colors(db_path)
                assert len(current) == 1, (round_index, current)
                assert current[0] in values, (round_index, current, values)
                listing = (await sessions[3].call_tool("list_memories", {})).content[0].text
                assert f'"{current[0]}"' in listing, (round_index, current, listing)
                assert not any(
                    f'"{v}"' in listing for v in values if v != current[0]
                ), (round_index, current, listing)

            # Distinct project memories, written by separate agents, survive
            # updates to the exclusive favorite color fact.
            texts = ["Acme: reunión aplazada al lunes.", "Beta: entrega aprobada hoy."]
            first = await sessions[0].call_tool("remember", {
                "statement": texts[0], "project": "Acme", "kind": "decision"})
            second = await sessions[1].call_tool("remember", {
                "statement": texts[1], "project": "Beta", "kind": "decision"})
            assert not first.is_error and not second.is_error
            a = (await sessions[2].call_tool(
                "recall", {"query": "reunión", "project": "Acme"})).content[0].text
            b = (await sessions[3].call_tool(
                "recall", {"query": "entrega", "project": "Beta"})).content[0].text
            assert texts[0] in a and texts[1] not in a
            assert texts[1] in b and texts[0] not in b

    asyncio.run(scenario())
    # Check persistence after all four independent server processes terminate.
    final = active_favorite_colors(db_path)
    assert len(final) == 1
    with sqlite3.connect(db_path) as conn:
        all_colors = conn.execute(
            "SELECT count(*), sum(status='active') "
            "FROM memories WHERE fact_predicate='favorite_color'"
        ).fetchone()
        assert all_colors[0] >= 1 and all_colors[1] == 1


def test_exclusive_updates_idempotence_forget_and_restart(tmp_path):
    path = tmp_path / "atomic-product.db"
    one = AgentMemory(str(path))
    two = AgentMemory(str(path))
    try:
        first = one.remember("Mi color favorito es azabache.")
        assert first.startswith("Remembered")
        assert two.remember("Mi color favorito es azabache.").startswith("Remembered")
        # Duplicate must not create two active rows.
        assert active_favorite_colors(path) == ["Mi color favorito es azabache."]
        two.remember("Mi color favorito es esmeralda.")
        assert active_favorite_colors(path) == ["Mi color favorito es esmeralda."]

        # Deleting the only active value must erase it (not restore prior one).
        memories = two.list_memories()
        matches = re.findall(r"\[([0-9a-f-]{36})\].*Mi color favorito es esmeralda", memories)
        assert len(matches) == 1, memories
        forgotten = one.forget(matches)
        # Forget includes the superseded historical value by design.
        assert "Deleted 2 memories" in forgotten
        assert active_favorite_colors(path) == []
        assert "esmeralda" not in two.list_memories()
        # SQLite must not resurrect old superseded values on a new process.
    finally:
        one.close()
        two.close()
    reopened = AgentMemory(str(path))
    try:
        assert active_favorite_colors(path) == []
        assert "azabache" not in reopened.list_memories()
        reopened.remember("Mi color favorito es coral.")
        assert active_favorite_colors(path) == ["Mi color favorito es coral."]
    finally:
        reopened.close()
