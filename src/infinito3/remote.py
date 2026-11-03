"""INFINITO over HTTP, for agents that only take remote MCP servers, such as ChatGPT's connectors.

Each person gets a private address, https://<host>/u/<token>/mcp, and their own memory file:
the token in the address is the only key, so whoever has the address has that memory.
The server keeps only a hash of each token, so a lost address cannot be read back, only replaced.

The memory tools are the same four as the local server (mcp_server.py). There is no vault:
its windows open on the screen of the machine that runs the server, and that is not the user's.

    infinito-remote add ana        # prints Ana's address, once
    infinito-remote list
    infinito-remote remove ana     # the address stops working and Ana's memory is deleted
    infinito-remote serve --port 8500

Data lives in INFINITO_REMOTE_DIR (default ~/.infinito-remote): people.json and people/<name>/memory.db.
"""
# No `from __future__ import annotations`: the MCP SDK reads the tools' annotations, and Context is
# imported inside build_app so the core never needs the SDK.
import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from .mcp_server import INSTRUCTIONS, TOOL_DESCRIPTIONS, AgentMemory

DEFAULT_DIR = Path.home() / ".infinito-remote"
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
PERSONAL_PATH = re.compile(r"/u/([A-Za-z0-9_-]{32,64})/mcp/?")
# One statement is one fact; a long text is a document, not a memory.
MAX_STATEMENT = 2000


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class People:
    """Who may use the server: names and token hashes in people.json, a memory folder each."""

    def __init__(self, folder: Optional[Path] = None):
        self.folder = Path(folder or os.environ.get("INFINITO_REMOTE_DIR") or DEFAULT_DIR)
        self.file = self.folder / "people.json"
        self._lock = threading.Lock()
        self._seen = None
        self._by_hash: Dict[str, str] = {}

    def _read(self) -> dict:
        return json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}

    def _write(self, people: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.file.with_suffix(".tmp")
        temporary.write_text(json.dumps(people, indent=2, ensure_ascii=False), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.file)

    def add(self, name: str) -> str:
        if not NAME.fullmatch(name):
            raise ValueError("A name is lowercase letters, digits and dashes, up to 32 characters.")
        people = self._read()
        if name in people:
            raise ValueError(f"{name} already has an address; remove it first to give a new one.")
        token = secrets.token_urlsafe(32)
        people[name] = {"token_sha256": _hash(token), "since": datetime.now(timezone.utc).isoformat()[:19]}
        self._write(people)
        return token

    def remove(self, name: str) -> bool:
        people = self._read()
        if name not in people:
            return False
        del people[name]
        self._write(people)
        shutil.rmtree(self.memory_folder(name), ignore_errors=True)
        return True

    def names(self) -> List[str]:
        return sorted(self._read())

    def who(self, token: str) -> Optional[str]:
        """The name a token belongs to; people.json is read again whenever it changes on disk."""
        with self._lock:
            seen = self.file.stat().st_mtime_ns if self.file.exists() else None
            if seen != self._seen:
                self._by_hash = {entry["token_sha256"]: name for name, entry in self._read().items()}
                self._seen = seen
            return self._by_hash.get(_hash(token))

    def memory_folder(self, name: str) -> Path:
        return self.folder / "people" / name


class _PersonalAddress:
    """ASGI middleware: /u/<token>/mcp becomes /mcp for the MCP app, with the person's name in the scope."""

    def __init__(self, app, people: People):
        self.app = app
        self.people = people

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        if scope["path"] == "/health":
            return await _plain(send, 200, "ok")
        match = PERSONAL_PATH.fullmatch(scope["path"])
        name = self.people.who(match.group(1)) if match else None
        if not name:
            return await _plain(send, 404, "Not found")
        state = dict(scope.get("state") or {}, infinito_person=name)
        await self.app(dict(scope, path="/mcp", raw_path=b"/mcp", state=state), receive, send)


async def _plain(send, status: int, text: str) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")]})
    await send({"type": "http.response.body", "body": text.encode()})


def build_app(people: People):
    """The ASGI app: the four memory tools, each call answered from its caller's own memory."""
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.server.transport_security import TransportSecuritySettings

    memories: Dict[str, AgentMemory] = {}
    lock = threading.Lock()

    def memory_of(ctx: Context) -> AgentMemory:
        name = ctx.request_context.request.scope["state"]["infinito_person"]
        with lock:
            if name not in memories:
                folder = people.memory_folder(name)
                folder.mkdir(parents=True, exist_ok=True)
                memories[name] = AgentMemory(str(folder / "memory.db"))
            return memories[name]

    server = MCPServer("infinito-memory", instructions=INSTRUCTIONS)

    @server.tool(description=TOOL_DESCRIPTIONS["remember"])
    def remember(statement: str, ctx: Context) -> str:
        if len(statement or "") > MAX_STATEMENT:
            return f"Not stored: one memory is one fact, at most {MAX_STATEMENT} characters."
        return memory_of(ctx).remember(statement)

    @server.tool(description=TOOL_DESCRIPTIONS["recall"])
    def recall(query: str, ctx: Context) -> str:
        return memory_of(ctx).recall(query)

    @server.tool(description=TOOL_DESCRIPTIONS["list_memories"])
    def list_memories(ctx: Context) -> str:
        return memory_of(ctx).list_memories()

    @server.tool(description=TOOL_DESCRIPTIONS["forget"])
    def forget(memory_ids: List[str], ctx: Context, keep: Optional[List[str]] = None) -> str:
        return memory_of(ctx).forget(memory_ids, keep or [])

    # DNS rebinding protection checks Host and Origin, which behind a proxy and from ChatGPT's servers are
    # neither 127.0.0.1 nor known in advance. The secret address is what keeps a memory private.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = server.streamable_http_app(stateless_http=True, json_response=True, transport_security=security)
    return _PersonalAddress(app, people)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="infinito-remote", description="INFINITO's memory over HTTP, one per person.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="give someone a private address").add_argument("name")
    commands.add_parser("remove", help="revoke an address and delete that memory").add_argument("name")
    commands.add_parser("list", help="who has an address")
    serve = commands.add_parser("serve", help="run the server")
    serve.add_argument("--port", type=int, default=8500)
    serve.add_argument("--bind", default="127.0.0.1")
    args = parser.parse_args(argv)
    people = People()

    if args.command == "add":
        token = people.add(args.name)
        base = os.environ.get("INFINITO_REMOTE_URL", "https://<host>").rstrip("/")
        print(f"{base}/u/{token}/mcp")
        print("This address is shown only once. Whoever has it can read and change this memory.")
    elif args.command == "remove":
        print("Removed: the address no longer works and the memory is deleted." if people.remove(args.name)
              else f"Nobody called {args.name}.")
    elif args.command == "list":
        print("\n".join(people.names()) or "Nobody yet.")
    else:
        import uvicorn

        uvicorn.run(build_app(people), host=args.bind, port=args.port, log_level="warning",
                    access_log=False)


if __name__ == "__main__":
    main()
