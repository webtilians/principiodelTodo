"""Regression for F10-S: two *different* authentic MCP stdio server processes
must never both expose exclusive user facts as active after concurrent writes.

This failed 16/16 scenarios before SQLite BEGIN IMMEDIATE (F10-S original),
and is now part of the product PR's ongoing regression suite. Synthetic only.
"""
from __future__ import annotations

import asyncio
import sqlite3
import sys

import pytest


def test_two_mcp_processes_serialize_exclusive_fact_replacements(tmp_path):
    pytest.importorskip("mcp.server.mcpserver")
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    path=tmp_path/"f10s-regression-shared.db"
    params=StdioServerParameters(
        command=sys.executable,
        args=["-m","infinito3.mcp_server"],
        env={"INFINITO_MEMORY_DB":str(path),
             "INFINITO_LOCAL_MEMORY":"on"},
    )

    async def scenario():
        async with stdio_client(params) as (read_a,write_a):
            async with ClientSession(read_a,write_a) as a:
                await a.initialize()
                async with stdio_client(params) as (read_b,write_b):
                    async with ClientSession(read_b,write_b) as b:
                        await b.initialize()
                        for i in range(12):
                            left=f"Mi color favorito es bronce{i}."
                            right=f"Mi color favorito es violeta{i}."
                            results=await asyncio.gather(
                                a.call_tool("remember",{"statement":left}),
                                b.call_tool("remember",{"statement":right}),
                            )
                            assert all(
                                not r.is_error and
                                r.content[0].text.startswith("Remembered")
                                for r in results
                            )
                            listing=(await b.call_tool(
                                "list_memories",{})).content[0].text
                            active_values=[
                                value for value in (left,right)
                                if '"'+value+'"' in listing
                            ]
                            assert len(active_values)==1,(
                                i,active_values,listing)
                            with sqlite3.connect(path) as db:
                                rows=db.execute(
                                    "SELECT content FROM memories "
                                    "WHERE status='active' AND "
                                    "fact_predicate='favorite_color'"
                                ).fetchall()
                            assert len(rows)==1, (i,rows)
                            assert rows[0][0]==active_values[0]

    asyncio.run(scenario())
