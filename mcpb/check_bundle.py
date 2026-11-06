"""Start an unpacked bundle as Claude Desktop does (uv run, per its manifest) and call its tools.

Usage: python mcpb/check_bundle.py <uv> <unpacked bundle dir> <memory folder>
Exits non-zero if the server does not start, lists the wrong tools, keeps a secret or loses a memory.
"""
import json
import os
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

uv, bundle, folder = sys.argv[1:4]
config = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))["server"]["mcp_config"]
env = {k: v.replace("${user_config.memory_folder}", folder) for k, v in config.get("env", {}).items()}
env.update({k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "SYSTEMROOT", "LOCALAPPDATA",
                                                         "USERPROFILE", "APPDATA", "TEMP", "TMP")})
params = StdioServerParameters(command=uv, args=[a.replace("${__dirname}", bundle) for a in config["args"]], env=env)


async def check() -> None:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            tools = sorted(t.name for t in (await client.list_tools()).tools)
            assert tools == ["forget", "list_memories", "overview", "recall", "remember", "remember_many", "vault_add",
                             "vault_list", "vault_lock", "vault_reveal", "vault_status", "vault_unlock"], tools
            status = (await client.call_tool("vault_status", {})).content[0].text
            assert status.startswith("No vault yet"), status
            kept = (await client.call_tool("remember", {"statement": "I work at an advertising agency."})).content[0].text
            refused = (await client.call_tool("remember", {"statement": "My card is 4111 1111 1111 1111"})).content[0].text
            listed = (await client.call_tool("list_memories", {})).content[0].text
            assert kept.startswith("Remembered"), kept
            assert refused.startswith("Not stored"), refused
            assert "advertising agency" in listed and "4111" not in listed, listed
            print(listed)


anyio.run(check)
assert Path(env["INFINITO_MEMORY_DB"]).exists()
print("bundle OK")
