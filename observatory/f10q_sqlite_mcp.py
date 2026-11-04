"""Real MCP stdio tool endpoint for F10-Q. SQL execution is SQLite, not simulated operator responses.

Everything lives in the CI sandbox and uses an ephemeral SQLite file. Opening
the DB read-only and forbidding non-SELECTs are independent safeguards.
"""
from __future__ import annotations

import json
import os
import sqlite3

from mcp.server.mcpserver import MCPServer

server = MCPServer("f10q-sqlite-readonly", instructions="Read-only SQL on isolated synthetic fixtures.")

@server.tool(description="Run a parameterized read-only SELECT on a synthetic SQLite database.")
def sql_select(sql: str, params: list[int | str | None]) -> str:
    query = sql.strip()
    if not query.lower().startswith("select ") or ";" in query:
        return json.dumps({"error": "READ_ONLY_SELECT_REQUIRED"}, sort_keys=True)
    try:
        path = os.environ["F10Q_SQLITE_DB"]
        with sqlite3.connect("file:" + path + "?mode=ro", uri=True) as conn:
            conn.execute("PRAGMA query_only=ON")
            cur = conn.execute(query, params)
            result = {
                "columns": [c[0] for c in cur.description],
                "rows": [list(r) for r in cur.fetchall()],
            }
        return json.dumps(result, sort_keys=True, separators=(",",":"))
    except (sqlite3.Error, KeyError, ValueError) as exc:
        return json.dumps({"error": type(exc).__name__},sort_keys=True)

if __name__ == "__main__":
    server.run("stdio")
