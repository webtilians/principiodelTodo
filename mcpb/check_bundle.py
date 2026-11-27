"""Start an unpacked bundle as Claude Desktop does (uv run, per its manifest) and call its tools.

Usage: python mcpb/check_bundle.py <uv> <unpacked bundle dir> <vault folder>
Exits non-zero if the server does not start, lists any tool but the vault's, offers a memory of its own or writes one:
the extension serves only the vault, and the memory comes from the person's INFINITO account through its connector.
"""
import json
import os
import re
import sys
from pathlib import Path

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

VAULT_TOOLS = ["vault_add", "vault_list", "vault_lock", "vault_reveal", "vault_status", "vault_unlock"]

uv, bundle, folder = sys.argv[1:4]
manifest = json.loads((Path(bundle) / "manifest.json").read_text(encoding="utf-8"))
config = manifest["server"]["mcp_config"]
env = {k: re.sub(r"\$\{user_config\.[a-z_]+\}", lambda _: folder, v) for k, v in config.get("env", {}).items()}
env.update({k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "SYSTEMROOT", "LOCALAPPDATA",
                                                         "USERPROFILE", "APPDATA", "TEMP", "TMP")})
params = StdioServerParameters(command=uv, args=[a.replace("${__dirname}", bundle) for a in config["args"]], env=env)
assert env.get("INFINITO_LOCAL_MEMORY") == "off" and "INFINITO_MEMORY_DB" not in env, env
assert sorted(tool["name"] for tool in manifest["tools"]) == VAULT_TOOLS, manifest["tools"]


async def check() -> None:
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as client:
            started = await client.initialize()
            assert started.server_info.name == "infinito-vault", started.server_info
            assert "only the user's vault" in (started.instructions or ""), started.instructions
            tools = sorted(t.name for t in (await client.list_tools()).tools)
            assert tools == VAULT_TOOLS, tools  # no remember, recall or forget: the memory is the account's
            status = (await client.call_tool("vault_status", {})).content[0].text
            assert status.startswith("No vault yet"), status
            print(status)


anyio.run(check)
assert not any(Path(folder).glob("memory.db*")), "the extension must not keep a memory of its own"
print("bundle OK: only the vault")
