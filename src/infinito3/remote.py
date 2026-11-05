"""INFINITO over HTTP: one memory per person, shared by every agent they use, readable only by its owner.

ChatGPT's connectors only take MCP servers on the internet, and an agent on another computer can only
reach a memory there, so this is where one person's memory can serve ChatGPT, Claude Code, Claude
Desktop and whatever agent comes next.

Most people connect with sign-in: every app is given the same public address, https://<host>/mcp, and
the person signs in once per app with a name and password (remote_accounts.py, OAuth). The older way
is a private address per person, https://<host>/u/<token>/mcp, whose token is the only key.

Nobody else can read a memory, whoever runs the server included. Each one is sealed on disk with
AES-256-GCM; for an address the key is derived from its token (HKDF-SHA256 with a salt of its own),
for an account it is random and kept only wrapped under the password, the recovery code and the apps'
tokens. The server keeps only hashes of tokens, so the files and their backups are unreadable without
them. A memory is opened, in this process only, while one of its owner's requests is served. What this
does not stop is someone who changes the server's code to capture passwords or tokens as they arrive;
for that the person has to trust whoever runs the server, as with any service.

The memory tools are the same four as the local server (mcp_server.py). There is no vault here: its
windows open on the screen of the machine that runs the server, and that is not the user's; the
local server serves it alone (INFINITO_LOCAL_MEMORY=off).

    infinito-remote serve --port 8500 --url https://memory.example.org   # sign-in needs the public URL
    infinito-remote list           # accounts and addresses
    infinito-remote remove ana     # Ana's memory is deleted, and her sign-ins or address stop working
    infinito-remote add ana        # the older way: prints a private address for Ana, once
    infinito-remote copy-local     # copies this computer's memory into an address's memory
    infinito-remote backup <folder>  # one more copy of the sealed memories (remote_backup.py)
    infinito-remote restore <file>   # puts a copy back, with the service stopped

Data lives in INFINITO_REMOTE_DIR (default ~/.infinito-remote): accounts.json, grants.db and
accounts/<name>/memory.sealed for sign-in; people.json and people/<name>/memory.sealed for addresses.
"""
# No `from __future__ import annotations`: the MCP SDK reads the tools' annotations, and Context is
# imported inside build_app so the core never needs the SDK.
import argparse
import base64
import hashlib
import json
import logging
import os
import re
import secrets
import shutil
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .mcp_server import INSTRUCTIONS, TOOL_DESCRIPTIONS, TOOL_HINTS, AgentMemory, MemoryItem
from .persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore

DEFAULT_DIR = Path.home() / ".infinito-remote"
NAME = re.compile(r"[a-z0-9][a-z0-9-]{0,31}")
PERSONAL_PATH = re.compile(r"/u/([A-Za-z0-9_-]{32,64})/mcp/?")
# One statement is one fact; a long text is a document, not a memory.
SEALED = b"INF1"  # format of a sealed memory: magic, 12-byte nonce, AES-GCM ciphertext and tag


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class People:
    """Who may use the server: names, token hashes and salts in people.json, a memory folder each."""

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
        people[name] = {"token_sha256": _hash(token), "since": datetime.now(timezone.utc).isoformat()[:19],
                        "salt": base64.b64encode(os.urandom(16)).decode()}
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

    def key(self, name: str, token: str) -> bytes:
        """The key that seals this person's memory, from their token; an address given before sealing gets a salt now."""
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF

        with self._lock:
            people = self._read()
            if "salt" not in people[name]:
                people[name]["salt"] = base64.b64encode(os.urandom(16)).decode()
                self._write(people)
            salt = base64.b64decode(people[name]["salt"])
        return HKDF(algorithm=hashes.SHA256(), length=32, salt=salt,
                    info=b"infinito-remote memory v1").derive(token.encode())

    def memory_folder(self, name: str) -> Path:
        return self.folder / "people" / name


def _wipe(path: Path) -> None:
    """Overwrite a file with zeros before deleting it."""
    if path.exists():
        with path.open("r+b") as handle:
            handle.write(b"\0" * path.stat().st_size)
            handle.flush()
            os.fsync(handle.fileno())
        path.unlink()


class SealedMemories:
    """Each person's memory as one encrypted file, opened only while one of their requests is served.

    owners keeps the folders: People for private addresses, Accounts (remote_accounts.py) for sign-in."""

    def __init__(self, owners, ranker=None):
        self.owners = owners
        self.ranker = ranker  # orders memories by meaning in recall (semantic.py), when the server has a model
        self._locks: Dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def _lock_of(self, name: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(name, threading.Lock())

    def _sealed(self, name: str) -> Path:
        return self.owners.memory_folder(name) / "memory.sealed"

    def read(self, name: str, key: bytes) -> Optional[bytes]:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        sealed = self._sealed(name)
        if sealed.exists():
            blob = sealed.read_bytes()
            if blob[:4] != SEALED:
                raise ValueError(f"{sealed} is not a sealed INFINITO memory.")
            return AESGCM(key).decrypt(blob[4:16], blob[16:], name.encode())
        plain = self.owners.memory_folder(name) / "memory.db"
        if not plain.exists():
            return None
        # A memory written before sealing: read it once; it is sealed and the plain files are wiped.
        source = sqlite3.connect(plain)
        try:
            source.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            return source.serialize()
        finally:
            source.close()

    def write(self, name: str, key: bytes, image: bytes) -> None:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM

        folder = self.owners.memory_folder(name)
        folder.mkdir(parents=True, exist_ok=True)
        nonce = os.urandom(12)
        temporary = folder / "memory.sealed.tmp"
        temporary.write_bytes(SEALED + nonce + AESGCM(key).encrypt(nonce, image, name.encode()))
        os.chmod(temporary, 0o600)
        temporary.replace(self._sealed(name))
        for leftover in ("memory.db", "memory.db-wal", "memory.db-shm"):
            _wipe(folder / leftover)

    def use(self, name: str, key: bytes, action: Callable[[AgentMemory], str], *, writes: bool) -> str:
        with self._lock_of(name):
            image = self.read(name, key)
            memory = AgentMemory(store=SQLiteCognitiveMemoryStore.from_bytes(image, HashEmbeddingProvider(256)),
                                 ranker=self.ranker)
            try:
                result = action(memory)
                plain_left = (self.owners.memory_folder(name) / "memory.db").exists()
                if writes or plain_left:
                    self.write(name, key, memory.store.to_bytes())
                return result
            finally:
                memory.close()


class _Router:
    """ASGI middleware in front of the MCP app. /mcp with a signed-in app's token, and /u/<token>/mcp (a private
    address), reach the MCP app with the person, their key and their kind of memory in the scope; the OAuth
    pages go to the sign-in app."""

    def __init__(self, app, people: People, auth=None):
        self.app = app
        self.people = people
        self.auth_app, self.auth_paths, self.bearer, self.metadata_url = auth or (None, set(), None, "")

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        path = scope["path"]
        if path == "/health":
            return await _plain(send, 200, "ok")
        if path in self.auth_paths:
            # The service log shows how far an app got signing in: method, path and status only, never a
            # query, a name, an address or a token.
            status = {}

            async def noted(message):
                if message["type"] == "http.response.start":
                    status["code"] = message["status"]
                await send(message)

            await self.auth_app(scope, receive, noted)
            if not (path == "/" and scope["method"] == "GET"):  # visits to the front page are not sign-in steps
                print(f"sign-in: {scope['method']} {path} {status.get('code')}", flush=True)
            return
        if path in ("/mcp", "/mcp/") and self.bearer:
            header = dict(scope["headers"]).get(b"authorization", b"").decode("latin-1")
            opened = self.bearer(header[7:].strip()) if header.lower().startswith("bearer ") else None
            if not opened:
                # The MCP way to say "sign in": the client finds the OAuth pages from this header.
                print(f"sign-in: {scope['method']} /mcp 401 ({'expired token' if header else 'no token'})", flush=True)
                challenge = f'Bearer error="invalid_token", resource_metadata="{self.metadata_url}"'
                return await _plain(send, 401, "Sign in to your INFINITO memory.",
                                    [(b"www-authenticate", challenge.encode())])
            (name, key), memories = opened, "accounts"
        else:
            match = PERSONAL_PATH.fullmatch(path)
            name = self.people.who(match.group(1)) if match else None
            if not name:
                # An app pointed at a wrong address shows up here; a private address is never written out.
                shown = "/u/…/mcp" if match else path[:80]
                print(f"sign-in: {scope['method']} {shown} 404 (no such address)", flush=True)
                return await _plain(send, 404, "Not found")
            key, memories = self.people.key(name, match.group(1)), "people"
        state = dict(scope.get("state") or {}, infinito_person=name, infinito_key=key, infinito_memories=memories)
        await self.app(dict(scope, path="/mcp", raw_path=b"/mcp", state=state), receive, send)


async def _plain(send, status: int, text: str, headers: Optional[list] = None) -> None:
    await send({"type": "http.response.start", "status": status,
                "headers": [(b"content-type", b"text/plain; charset=utf-8")] + (headers or [])})
    await send({"type": "http.response.body", "body": text.encode()})


def build_app(people: People, issuer: Optional[str] = None):
    """The ASGI app: the four memory tools, each call answered from its caller's own sealed memory.

    With issuer (the server's public URL), people can also sign in at issuer/mcp (remote_accounts.py)."""
    from mcp.server.mcpserver import Context, MCPServer
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp.types import ToolAnnotations

    from .semantic import ranker_from

    # With a model in INFINITO_EMBEDDINGS, recall finds memories by meaning; it loads in the background.
    ranker = ranker_from(os.environ.get("INFINITO_EMBEDDINGS"))
    if ranker is not None:
        threading.Thread(target=ranker.warm, daemon=True).start()
    sealed = {"people": SealedMemories(people, ranker)}
    auth = None
    grants = None
    if issuer:
        from .remote_accounts import Accounts, Grants, build_auth

        grants = Grants(people.folder)
        accounts = Accounts(people.folder)
        sealed["accounts"] = SealedMemories(accounts, ranker)

        def adopt(name: str, key: bytes, address: str) -> bool:
            """Someone who used a private address signs up with it: the memory moves and the address ends."""
            match = PERSONAL_PATH.search(address)
            person = people.who(match.group(1)) if match else None
            if not person:
                return False
            with sealed["people"]._lock_of(person):
                image = sealed["people"].read(person, people.key(person, match.group(1)))
                if image:
                    sealed["accounts"].write(name, key, image)
                people.remove(person)
            return True

        auth = build_auth(accounts, grants, issuer, adopt)

    def with_memory(ctx: Context, action: Callable[[AgentMemory], str], writes: bool) -> str:
        state = ctx.request_context.request.scope["state"]
        return sealed[state["infinito_memories"]].use(state["infinito_person"], state["infinito_key"], action,
                                                       writes=writes)

    server = MCPServer("infinito-memory", instructions=INSTRUCTIONS)
    # OpenAI's apps (ChatGPT, Codex) also read, per tool, that it needs the person signed in with OAuth.
    # Every tool does: without a token the server answers 401, as the MCP specification asks.
    signed_in = {"securitySchemes": [{"type": "oauth2", "scopes": ["memory"]}]} if issuer else None

    def memory_tool(name: str):
        return server.tool(description=TOOL_DESCRIPTIONS[name], annotations=ToolAnnotations(**TOOL_HINTS[name]),
                           meta=signed_in)

    @memory_tool("remember")
    def remember(statement: str, ctx: Context, kind: Optional[str] = None, source: Optional[str] = None,
                 project: Optional[str] = None, date: Optional[str] = None) -> str:
        return with_memory(ctx, lambda memory: memory.remember(statement, kind, source, project, date), writes=True)

    @memory_tool("remember_many")
    def remember_many(items: List[MemoryItem], ctx: Context, source: Optional[str] = None,
                      project: Optional[str] = None, date: Optional[str] = None) -> str:
        return with_memory(ctx, lambda memory: memory.remember_many(items, source, project, date), writes=True)

    @memory_tool("recall")
    def recall(query: str, ctx: Context, limit: Optional[int] = None, project: Optional[str] = None,
               kind: Optional[str] = None) -> str:
        return with_memory(ctx, lambda memory: memory.recall(query, limit, project, kind), writes=False)

    @memory_tool("list_memories")
    def list_memories(ctx: Context, project: Optional[str] = None, kind: Optional[str] = None) -> str:
        return with_memory(ctx, lambda memory: memory.list_memories(project, kind), writes=False)

    @memory_tool("forget")
    def forget(memory_ids: List[str], ctx: Context, keep: Optional[List[str]] = None) -> str:
        return with_memory(ctx, lambda memory: memory.forget(memory_ids, keep or []), writes=True)

    if issuer:
        from . import remote_vault

        def vault_tool(name: str):
            return server.tool(description=remote_vault.TOOL_DESCRIPTIONS[name],
                               annotations=ToolAnnotations(**remote_vault.TOOL_HINTS), meta=signed_in)

        def vault_link(ctx: Context, action: str, entry: str) -> str:
            state = ctx.request_context.request.scope["state"]
            if state["infinito_memories"] != "accounts":
                return remote_vault.NEEDS_ACCOUNT
            return remote_vault.answer_with_link(issuer, action, entry, lambda data: grants.issue(
                remote_vault.GRANT, client_id="", seconds=remote_vault.LINK_SECONDS, account=state["infinito_person"],
                key=state["infinito_key"], data=data))

        @vault_tool("vault_add")
        def vault_add(entry: str, ctx: Context) -> str:
            return vault_link(ctx, "add", entry)

        @vault_tool("vault_reveal")
        def vault_reveal(entry: str, ctx: Context) -> str:
            return vault_link(ctx, "reveal", entry)

        @vault_tool("vault_list")
        def vault_list(ctx: Context) -> str:
            state = ctx.request_context.request.scope["state"]
            return remote_vault.NOTHING_KEPT if state["infinito_memories"] == "accounts" else remote_vault.NEEDS_ACCOUNT

    # DNS rebinding protection checks Host and Origin, which behind a proxy and from ChatGPT's servers are
    # neither 127.0.0.1 nor known in advance. The secret address is what keeps a memory private.
    security = TransportSecuritySettings(enable_dns_rebinding_protection=False)
    app = server.streamable_http_app(stateless_http=True, json_response=True, transport_security=security)
    return _Router(app, people, auth)


async def _copy(address: str, statements: List[str]) -> Tuple[List[str], int]:
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client

    async with streamable_http_client(address) as (read, write):
        async with ClientSession(read, write) as client:
            await client.initialize()
            # What is already there is not sent again, so copying twice changes nothing.
            listed = (await client.call_tool("list_memories", {})).content[0].text
            new = [s for s in statements if f'"{s}"' not in listed]
            answers = [(await client.call_tool("remember", {"statement": s})).content[0].text for s in new]
            return answers, len(statements) - len(new)


def copy_local(address: str, local_db: Optional[str] = None) -> str:
    """Send this computer's memories, oldest first, to a personal address with remember."""
    import anyio

    for library in ("httpx", "httpx2"):  # their request lines would show the address
        logging.getLogger(library).setLevel(logging.WARNING)
    local = AgentMemory(local_db)
    try:
        statements = [record.content for record in local.store.all()]
    finally:
        local.close()
    if not statements:
        return "This computer has no memories to copy."
    answers, already = anyio.run(_copy, address, statements)
    kept = sum(answer.startswith("Remembered") for answer in answers)
    refused = len(answers) - kept
    return (f"Copied {kept} of {len(statements)} memories"
            + (f"; {already} were already there" if already else "")
            + (f"; {refused} were refused" if refused else "") + ".")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(prog="infinito-remote", description="INFINITO's memory over HTTP, one per person.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("add", help="give someone a private address (the older way)").add_argument("name")
    commands.add_parser("remove", help="delete someone's memory and end their sign-ins or address").add_argument("name")
    commands.add_parser("list", help="who has a memory: accounts and addresses")
    commands.add_parser("copy-local", help="copy this computer's memory into an address's memory")
    backup = commands.add_parser("backup", help="write one more copy of the sealed memories into a folder")
    backup.add_argument("folder")
    backup.add_argument("--keep", type=int, default=14, help="how many copies to keep in the folder")
    commands.add_parser("verify-backup", help="check that a copy is whole").add_argument("file")
    commands.add_parser("restore", help="put a copy back, with the service stopped").add_argument("file")
    # On a person's computer (remote_client.py): sign it in once, then agents' hooks bring the memory in.
    commands.add_parser("login", help="sign this computer in to an INFINITO server").add_argument("server")
    commands.add_parser("logout", help="delete this computer's session")
    commands.add_parser("context", help="hook for Claude Code and Codex: the memories related to a message")
    commands.add_parser("checkpoint", help="hook for Claude Code: after work, the agent saves where it stands")
    serve = commands.add_parser("serve", help="run the server")
    serve.add_argument("--port", type=int, default=8500)
    serve.add_argument("--bind", default="127.0.0.1")
    serve.add_argument("--url", default=os.environ.get("INFINITO_REMOTE_URL"),
                       help="the server's public URL; without it there is no sign-in, only addresses")
    args = parser.parse_args(argv)
    if args.command in ("login", "logout", "context", "checkpoint"):
        from . import remote_client

        if args.command == "context":
            return remote_client.context()
        if args.command == "checkpoint":
            return remote_client.checkpoint()
        return print(remote_client.login(args.server) if args.command == "login" else remote_client.logout())
    if args.command == "verify-backup":
        from . import remote_backup

        return print(remote_backup.verify(args.file))
    people = People()

    if args.command == "add":
        token = people.add(args.name)
        base = os.environ.get("INFINITO_REMOTE_URL", "https://<host>").rstrip("/")
        print(f"{base}/u/{token}/mcp")
        print("This address is shown only once. Whoever has it can read and change this memory.")
    elif args.command == "remove":
        from .remote_accounts import Accounts, Grants

        removed = people.remove(args.name)
        if Accounts(people.folder).remove(args.name):
            Grants(people.folder).drop_account(args.name)
            removed = True
        print("Removed: the memory is deleted and its sign-ins or address no longer work." if removed
              else f"Nobody called {args.name}.")
    elif args.command == "list":
        from .remote_accounts import Accounts

        names = people.names() + [f"{name} (sign-in)" for name in Accounts(people.folder).names()]
        print("\n".join(names) or "Nobody yet.")
    elif args.command in ("backup", "restore"):
        from . import remote_backup

        print(remote_backup.backup(people.folder, args.folder, args.keep) if args.command == "backup"
              else remote_backup.restore(args.file, people.folder))
    elif args.command == "copy-local":
        import getpass

        # Asked without echo, so the address stays out of the screen and the shell history.
        address = getpass.getpass("Your private INFINITO address (it will not be shown): ").strip()
        print(copy_local(address))
    else:
        import uvicorn

        # A restart waits at most 5 s for open requests; then the update goes on.
        uvicorn.run(build_app(people, args.url), host=args.bind, port=args.port, log_level="warning",
                    access_log=False, timeout_graceful_shutdown=5)


if __name__ == "__main__":
    main()
