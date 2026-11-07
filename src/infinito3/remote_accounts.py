"""Accounts for INFINITO's remote memory: one public address, and a password typed once per app.

Claude, ChatGPT and any other MCP client connect to https://<host>/mcp. The first time, the app opens
INFINITO's sign-in page; the person creates their memory there (a name, a password) or signs in, and
the app gets tokens through OAuth 2.1 (PKCE, dynamic client registration), as the MCP specification
asks. Nothing else to copy, install or keep secret.

Nobody who reads the server's disk can read a memory, whoever runs the server included:

- each memory is sealed with its own random key (SealedMemories in remote.py, AES-256-GCM);
- that key is stored only wrapped: under a key derived from the person's password (scrypt), under one
  from their recovery code, and under each token an app holds, of which only a hash is stored;
- so a memory opens only while a request brings the password, the recovery code or a live token.

The recovery code is shown once, when the memory is created. With it a forgotten password is replaced
and the memory kept; changing the password signs every app out. Losing both loses the memory.
"""
# No `from __future__ import annotations`: pydantic reads the field annotations of the token models.
import base64
import hashlib
import html
import json
import os
import re
import secrets
import shutil
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

NAME = re.compile(r"[a-z0-9][a-z0-9-]{1,31}")
MIN_PASSWORD = 10
RECOVERY_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"  # no 0/O, 1/I/L: it is copied by hand
ACCESS_SECONDS = 3600
REFRESH_SECONDS = 90 * 86400
CODE_SECONDS = 600
SIGN_IN_SECONDS = 1800  # the sign-in page, and the recovery code page before the app gets its tokens
SCOPE = "memory"
_SCRYPT = {"n": 2 ** 15, "r": 8, "p": 1}


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _seal(key: bytes, data: bytes, label: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    return nonce + AESGCM(key).encrypt(nonce, data, label.encode())


def _unseal(key: bytes, blob: bytes, label: str) -> Optional[bytes]:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    try:
        return AESGCM(key).decrypt(blob[:12], blob[12:], label.encode())
    except InvalidTag:
        return None


def _from_password(password: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(password.encode(), salt=salt, n=n, r=r, p=p, maxmem=128 * n * r * p + 2 ** 20, dklen=32)


def _from_token(token: str, purpose: bytes) -> bytes:
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    return HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=purpose).derive(token.encode())


def _recovery_code() -> str:
    letters = "".join(secrets.choice(RECOVERY_ALPHABET) for _ in range(20))
    return "-".join(letters[i:i + 4] for i in range(0, 20, 4))


def _normal_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (code or "").upper())


class Accounts:
    """Who has a memory: names, and the memory key wrapped under each person's password and recovery code."""

    def __init__(self, folder: Path):
        self.folder = Path(folder)
        self.file = self.folder / "accounts.json"
        self._lock = threading.Lock()
        self._failures: Dict[str, Tuple[int, float]] = {}

    def _read(self) -> dict:
        return json.loads(self.file.read_text(encoding="utf-8")) if self.file.exists() else {}

    def _write(self, accounts: dict) -> None:
        self.folder.mkdir(parents=True, exist_ok=True)
        temporary = self.file.with_suffix(".tmp")
        temporary.write_text(json.dumps(accounts, indent=2), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.file)

    @staticmethod
    def _wrap_with_password(key: bytes, password: str, name: str) -> dict:
        salt = os.urandom(16)
        return {"salt": _b64(salt), **_SCRYPT,
                "wrapped": _b64(_seal(_from_password(password, salt, **_SCRYPT), key, f"password:{name}"))}

    @staticmethod
    def _wrap_with_code(key: bytes, code: str, name: str) -> dict:
        return {"wrapped": _b64(_seal(_from_token(_normal_code(code), b"infinito recovery v1"), key, f"code:{name}"))}

    def names(self) -> List[str]:
        return sorted(self._read())

    def exists(self, name: str) -> bool:
        return name in self._read()

    def memory_folder(self, name: str) -> Path:
        return self.folder / "accounts" / name

    def create(self, name: str, password: str) -> Tuple[bytes, str]:
        """A new memory: its key, and the recovery code to show the person once."""
        key, code = os.urandom(32), _recovery_code()
        entry = {"since": datetime.now(timezone.utc).isoformat()[:19],
                 "password": self._wrap_with_password(key, password, name),
                 "recovery": self._wrap_with_code(key, code, name)}
        with self._lock:
            accounts = self._read()
            if name in accounts:
                raise ValueError(name)
            accounts[name] = entry
            self._write(accounts)
        return key, code

    def waiting(self, name: str) -> int:
        """Seconds before this name may try again: each failure after the third doubles the wait."""
        count, last = self._failures.get(name, (0, 0.0))
        wait = 0 if count < 3 else min(2 ** (count - 3) * 15, 900)
        return max(0, int(last + wait - time.time()))

    def _failed(self, name: str) -> None:
        count, _ = self._failures.get(name, (0, 0.0))
        self._failures[name] = (count + 1, time.time())

    def open(self, name: str, password: str) -> Optional[bytes]:
        entry = self._read().get(name)
        if entry is None:
            self._failed(name)
            return None
        stored = entry["password"]
        derived = _from_password(password, base64.b64decode(stored["salt"]), stored["n"], stored["r"], stored["p"])
        key = _unseal(derived, base64.b64decode(stored["wrapped"]), f"password:{name}")
        if key is None:
            self._failed(name)
        else:
            self._failures.pop(name, None)
        return key

    def recover(self, name: str, code: str, new_password: str) -> Optional[bytes]:
        """With the recovery code, a new password; the memory and the code stay the same."""
        entry = self._read().get(name)
        key = None
        if entry is not None:
            unwrap = _from_token(_normal_code(code), b"infinito recovery v1")
            key = _unseal(unwrap, base64.b64decode(entry["recovery"]["wrapped"]), f"code:{name}")
        if key is None:
            self._failed(name)
            return None
        self._failures.pop(name, None)
        wrapped = self._wrap_with_password(key, new_password, name)
        with self._lock:
            accounts = self._read()
            accounts[name]["password"] = wrapped
            self._write(accounts)
        return key

    def vault_allowed(self, name: str) -> bool:
        return bool(self._read().get(name, {}).get("vault"))

    def allow_vault(self, name: str, allowed: bool = True) -> bool:
        """The server's owner lets an account use the vault on the server (remote_vault.py), or stops it."""
        with self._lock:
            accounts = self._read()
            if name not in accounts:
                return False
            accounts[name]["vault"] = allowed
            self._write(accounts)
        return True

    def app_levels(self, name: str) -> Dict[str, str]:
        """What the person lets each of their apps do (remote_apps.py), by the app's client id."""
        return dict(self._read().get(name, {}).get("apps", {}))

    def app_level(self, name: str, client_id: str) -> str:
        from .remote_apps import DEFAULT

        return self.app_levels(name).get(client_id, DEFAULT)

    def set_app_level(self, name: str, client_id: str, level: Optional[str]) -> bool:
        """The person's choice for one app; None forgets it, and the app is back where a new one starts."""
        from .remote_apps import LEVELS

        if level is not None and level not in LEVELS:
            raise ValueError(level)
        with self._lock:
            accounts = self._read()
            if name not in accounts:
                return False
            apps = accounts[name].setdefault("apps", {})
            if level is None:
                apps.pop(client_id, None)
            else:
                apps[client_id] = level
            self._write(accounts)
        return True

    def remove(self, name: str) -> bool:
        with self._lock:
            accounts = self._read()
            if name not in accounts:
                return False
            del accounts[name]
            self._write(accounts)
        shutil.rmtree(self.memory_folder(name), ignore_errors=True)
        return True


class Grants:
    """What apps hold: registered clients, and sign-in requests, codes and tokens, stored only as hashes.

    A code or token also carries the memory key, wrapped under the token itself, so it is what opens
    the memory and the stored row alone does not."""

    def __init__(self, folder: Path):
        Path(folder).mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(Path(folder) / "grants.db"), check_same_thread=False)
        self._db.executescript(
            "CREATE TABLE IF NOT EXISTS clients (id TEXT PRIMARY KEY, info TEXT NOT NULL, since REAL NOT NULL);"
            "CREATE TABLE IF NOT EXISTS grants (hash TEXT PRIMARY KEY, kind TEXT NOT NULL, account TEXT NOT NULL,"
            " client_id TEXT NOT NULL, expires REAL NOT NULL, data TEXT NOT NULL, wrapped BLOB);")
        self._lock = threading.Lock()

    def add_client(self, client_id: str, info: str) -> None:
        with self._lock, self._db:
            self._db.execute("INSERT OR REPLACE INTO clients VALUES (?, ?, ?)", (client_id, info, time.time()))

    def client(self, client_id: str) -> Optional[str]:
        with self._lock:
            row = self._db.execute("SELECT info FROM clients WHERE id = ?", (client_id,)).fetchone()
        return row[0] if row else None

    def issue(self, kind: str, *, client_id: str, seconds: int, account: str = "", key: Optional[bytes] = None,
              data: Optional[dict] = None) -> str:
        token = secrets.token_urlsafe(32)
        digest = _hash(token)
        wrapped = _seal(_from_token(token, b"infinito grant v1"), key, digest) if key else None
        with self._lock, self._db:
            self._db.execute("DELETE FROM grants WHERE expires < ?", (time.time(),))
            self._db.execute("INSERT INTO grants VALUES (?, ?, ?, ?, ?, ?, ?)",
                             (digest, kind, account, client_id, time.time() + seconds, json.dumps(data or {}), wrapped))
        return token

    def find(self, kind: str, token: str) -> Optional[dict]:
        """The live grant of this kind, with its memory key unwrapped (key is None when it carries none)."""
        digest = _hash(token or "")
        with self._lock:
            row = self._db.execute("SELECT account, client_id, expires, data, wrapped FROM grants "
                                   "WHERE hash = ? AND kind = ?", (digest, kind)).fetchone()
        if row is None or row[2] < time.time():
            return None
        key = _unseal(_from_token(token, b"infinito grant v1"), row[4], digest) if row[4] else None
        if row[4] and key is None:
            return None
        return {"account": row[0], "client_id": row[1], "expires": row[2], "data": json.loads(row[3]), "key": key}

    def drop(self, token: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM grants WHERE hash = ?", (_hash(token or ""),))

    def drop_account(self, account: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM grants WHERE account = ?", (account,))

    def apps_of(self, account: str) -> List[dict]:
        """The apps that hold a live sign-in to this memory, first connected first: their client id, the name they
        registered with, the host they sign in from and the day they first connected."""
        with self._lock:
            rows = self._db.execute(
                "SELECT g.client_id, c.info, c.since FROM grants g JOIN clients c ON c.id = g.client_id "
                "WHERE g.account = ? AND g.kind IN ('access', 'refresh') AND g.expires > ? "
                "GROUP BY g.client_id ORDER BY c.since", (account, time.time())).fetchall()
        apps = []
        for client_id, info, since in rows:
            try:
                uris = json.loads(info).get("redirect_uris") or [""]
            except (ValueError, AttributeError):
                uris = [""]
            host = re.sub(r"^[a-z]+://([^/:]+).*$", r"\1", str(uris[0]))
            apps.append({"client_id": client_id, "name": _app_name(info) or host or "?", "host": host,
                         "since": datetime.fromtimestamp(since, timezone.utc).date().isoformat()})
        return apps

    def drop_client(self, account: str, client_id: str) -> None:
        """Signs one app out of one memory: its tokens stop working, and it has to sign in again."""
        with self._lock, self._db:
            self._db.execute("DELETE FROM grants WHERE account = ? AND client_id = ? AND kind IN ('access', 'refresh')",
                             (account, client_id))


def build_auth(accounts: Accounts, grants: Grants, issuer: str,
               adopt: Callable[[str, bytes, str], bool], memories=None):
    """The OAuth endpoints, the sign-in page, and a function that turns a bearer token into (name, key).

    adopt(name, key, old_address) moves a memory kept under a private address (remote.py's People) into
    the new account; it is how someone who used an address keeps their memory. memories (remote.py's
    SealedMemories) adds the page where a person sees and fixes their own memory."""
    import anyio
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import RedirectResponse
    from starlette.routing import Route

    from mcp.server.auth.provider import (AccessToken, AuthorizationCode, AuthorizationParams, RefreshToken,
                                          TokenError, construct_redirect_uri)
    from mcp.server.auth.handlers.metadata import MetadataHandler
    from mcp.server.auth.routes import (build_metadata, cors_middleware, create_auth_routes,
                                        create_protected_resource_routes)
    from mcp.server.auth.settings import AuthSettings, ClientRegistrationOptions, RevocationOptions
    from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

    class Code(AuthorizationCode):
        key: bytes

    class Refresh(RefreshToken):
        key: bytes

    issuer = issuer.rstrip("/")
    settings = AuthSettings(issuer_url=issuer, resource_server_url=f"{issuer}/mcp", validate_token_resource=False,
                            # Any scope an app asks for is accepted (openid, profile...): there is only one
                            # thing to grant, the person's memory, so refusing them would only lock apps out.
                            client_registration_options=ClientRegistrationOptions(
                                enabled=True, default_scopes=[SCOPE]),
                            revocation_options=RevocationOptions(enabled=True))

    def tokens(client_id: str, account: str, key: bytes, scopes: List[str], resource: Optional[str]) -> OAuthToken:
        data = {"scopes": scopes, "resource": resource}
        access = grants.issue("access", client_id=client_id, seconds=ACCESS_SECONDS, account=account, key=key, data=data)
        refresh = grants.issue("refresh", client_id=client_id, seconds=REFRESH_SECONDS, account=account, key=key,
                               data=data)
        return OAuthToken(access_token=access, expires_in=ACCESS_SECONDS, refresh_token=refresh, scope=" ".join(scopes))

    class Provider:
        async def get_client(self, client_id: str) -> Optional[OAuthClientInformationFull]:
            info = grants.client(client_id)
            return OAuthClientInformationFull.model_validate_json(info) if info else None

        async def register_client(self, client_info: OAuthClientInformationFull) -> None:
            grants.add_client(client_info.client_id, client_info.model_dump_json())

        async def authorize(self, client: OAuthClientInformationFull, params: AuthorizationParams) -> str:
            request = grants.issue("request", client_id=client.client_id, seconds=SIGN_IN_SECONDS, data={
                "state": params.state, "scopes": params.scopes or [SCOPE], "code_challenge": params.code_challenge,
                "redirect_uri": str(params.redirect_uri), "explicit": params.redirect_uri_provided_explicitly,
                "resource": params.resource})
            return f"{issuer}/login?r={request}"

        async def load_authorization_code(self, client, authorization_code: str) -> Optional[Code]:
            found = grants.find("code", authorization_code)
            if not found or found["client_id"] != client.client_id:
                return None
            data = found["data"]
            return Code(code=authorization_code, scopes=data["scopes"], expires_at=found["expires"],
                        client_id=found["client_id"], code_challenge=data["code_challenge"],
                        redirect_uri=data["redirect_uri"], redirect_uri_provided_explicitly=data["explicit"],
                        resource=data["resource"], subject=found["account"], key=found["key"])

        async def exchange_authorization_code(self, client, authorization_code: Code) -> OAuthToken:
            grants.drop(authorization_code.code)  # a code works once
            return tokens(client.client_id, authorization_code.subject, authorization_code.key,
                          authorization_code.scopes, authorization_code.resource)

        async def load_refresh_token(self, client, refresh_token: str) -> Optional[Refresh]:
            found = grants.find("refresh", refresh_token)
            if not found or found["client_id"] != client.client_id:
                return None
            return Refresh(token=refresh_token, client_id=found["client_id"], scopes=found["data"]["scopes"],
                           expires_at=int(found["expires"]), resource=found["data"]["resource"],
                           subject=found["account"], key=found["key"])

        async def exchange_refresh_token(self, client, refresh_token: Refresh, scopes: List[str]) -> OAuthToken:
            if not accounts.exists(refresh_token.subject):
                raise TokenError(error="invalid_grant", error_description="This memory no longer exists.")
            grants.drop(refresh_token.token)  # rotated: each refresh token works once
            return tokens(client.client_id, refresh_token.subject, refresh_token.key,
                          scopes or refresh_token.scopes, refresh_token.resource)

        async def load_access_token(self, token: str) -> Optional[AccessToken]:
            found = grants.find("access", token)
            if not found:
                return None
            return AccessToken(token=token, client_id=found["client_id"], scopes=found["data"]["scopes"],
                               expires_at=int(found["expires"]), resource=found["data"]["resource"],
                               subject=found["account"])

        async def revoke_token(self, token) -> None:
            grants.drop(token.token)

    routes = create_auth_routes(Provider(), settings.issuer_url,
                                client_registration_options=settings.client_registration_options,
                                revocation_options=settings.revocation_options)
    resource_routes = create_protected_resource_routes(settings.resource_server_url, [settings.issuer_url],
                                                       scopes_supported=[SCOPE], resource_name="INFINITO")
    # Some clients look for the metadata without the /mcp suffix.
    resource_routes.append(Route("/.well-known/oauth-protected-resource", resource_routes[0].endpoint,
                                 methods=["GET", "OPTIONS"]))
    # The server's own metadata also names the one scope, and says apps without a secret of their own may sign
    # in (PKCE protects them; Claude and ChatGPT register that way), so no app stops before registering.
    metadata = build_metadata(settings.issuer_url, None, settings.client_registration_options,
                              settings.revocation_options)
    metadata.scopes_supported = [SCOPE]
    metadata.token_endpoint_auth_methods_supported = ["none", "client_secret_post", "client_secret_basic"]
    server_metadata = cors_middleware(MetadataHandler(metadata).handle, ["GET", "OPTIONS"])
    routes = [r for r in routes if r.path != "/.well-known/oauth-authorization-server"]
    # Some clients look for it under these other names too.
    for path in ("/.well-known/oauth-authorization-server", "/.well-known/oauth-authorization-server/mcp",
                 "/.well-known/openid-configuration"):
        resource_routes.append(Route(path, server_metadata, methods=["GET", "OPTIONS"]))

    # Each password check takes about 32 MB and a tenth of a second on purpose (scrypt); one at a time,
    # so a flood of attempts waits in line instead of filling the server's memory.
    one_at_a_time = anyio.CapacityLimiter(1)

    async def sign_in_page(request: Request):
        t = _texts(request)
        pending = grants.find("request", request.query_params.get("r", ""))
        if pending is None:
            return _page(t, t["expired_title"], f"<p>{t['expired']}</p>", status=400)
        mode = request.query_params.get("m", "signin")
        return _form(t, request.query_params["r"], mode if mode in _FORMS else "signin", _asking(pending))

    async def sign_in(request: Request):
        t = _texts(request)
        form = await request.form()
        if form.get("action") == "continue":  # after the recovery code: the app gets its code
            found = grants.find("continue", str(form.get("c", "")))
            if found is None:
                return _page(t, t["expired_title"], f"<p>{t['expired']}</p>", status=400)
            grants.drop(str(form.get("c", "")))
            return _back_to_app(found["data"], found["account"], found["key"], found["client_id"])
        token = str(form.get("r", ""))
        pending = grants.find("request", token)
        if pending is None:
            return _page(t, t["expired_title"], f"<p>{t['expired']}</p>", status=400)
        mode = str(form.get("action", "signin"))
        mode = mode if mode in _FORMS else "signin"
        name = str(form.get("name", "")).strip().lower()
        password, repeat = str(form.get("password", "")), str(form.get("repeat", ""))

        def again(message: str):
            return _form(t, token, mode, _asking(pending), name=name, error=message, status=400)

        if not NAME.fullmatch(name):
            return again(t["bad_name"])
        if accounts.waiting(name):
            return again(t["wait"].format(n=accounts.waiting(name)))
        if mode in ("signup", "recover"):
            if len(password) < MIN_PASSWORD:
                return again(t["short"].format(n=MIN_PASSWORD))
            if password != repeat:
                return again(t["differ"])
        if mode == "signup":
            if accounts.exists(name):
                return again(t["taken"])
            key, code = await anyio.to_thread.run_sync(accounts.create, name, password, limiter=one_at_a_time)
            old = str(form.get("old_address", "")).strip()
            moved = old and await anyio.to_thread.run_sync(adopt, name, key, old)
            grants.drop(token)
            onward = grants.issue("continue", client_id=pending["client_id"], seconds=SIGN_IN_SECONDS,
                                  account=name, key=key, data=pending["data"])
            return _recovery_page(t, code, onward, moved=bool(moved), old_given=bool(old))
        if mode == "recover":
            key = await anyio.to_thread.run_sync(accounts.recover, name, str(form.get("code", "")), password,
                                                 limiter=one_at_a_time)
            if key is None:
                return again(t["bad_code"])
            grants.drop_account(name)  # a new password signs every app out
        elif not accounts.exists(name):
            # Most people who try to sign in to a name that does not exist have not created it yet:
            # the page offers to, with the name already filled in.
            return _form(t, token, "signup", _asking(pending), name=name, error=t["no_such"].format(name=name))
        else:
            key = await anyio.to_thread.run_sync(accounts.open, name, password, limiter=one_at_a_time)
            if key is None:
                return again(t["bad_password"])
        grants.drop(token)
        return _back_to_app(pending["data"], name, key, pending["client_id"])

    def _back_to_app(data: dict, name: str, key: bytes, client_id: str):
        code = grants.issue("code", client_id=client_id, seconds=CODE_SECONDS, account=name, key=key, data=data)
        target = construct_redirect_uri(data["redirect_uri"], code=code, state=data["state"])
        return RedirectResponse(target, status_code=303, headers=_HEADERS)

    def _asking(pending: dict) -> str:
        info = grants.client(pending["client_id"])
        client = json.loads(info) if info else {}
        host = re.sub(r"^[a-z]+://([^/:]+).*$", r"\1", pending["data"]["redirect_uri"])
        return f"{client.get('client_name') or host} ({host})"

    async def home(request: Request):
        return remote_home.home_page(_texts(request)["lang"], f"{issuer}/mcp", _STYLE, _HOME_HEADERS)

    from . import remote_home, remote_memory, remote_vault

    memory_page = [] if memories is None else remote_memory.routes(accounts, grants, memories, one_at_a_time,
                                                                   secure=issuer.startswith("https://"))
    app = Starlette(routes=routes + resource_routes + remote_vault.routes(accounts, grants, one_at_a_time) + memory_page
                    + [Route("/login", sign_in_page, methods=["GET"]), Route("/login", sign_in, methods=["POST"]),
                       Route("/", home, methods=["GET"])])

    def bearer(token: str) -> Optional[Tuple[str, bytes, Optional[str], str]]:
        """Whose memory a token opens: the account, its key, the app's name and the app's client id."""
        found = grants.find("access", token)
        if not found or not accounts.exists(found["account"]):
            return None
        return found["account"], found["key"], _app_name(grants.client(found["client_id"])), found["client_id"]

    paths = {route.path for route in app.routes}
    metadata_url = f"{issuer}/.well-known/oauth-protected-resource/mcp"
    return app, paths, bearer, metadata_url


# The sign-in page: plain HTML, no scripts, nothing loaded from elsewhere.

def _app_name(info: Optional[str]) -> Optional[str]:
    """The name an app registered with, as it appears next to what it saves. It is the app's own word for itself, the
    same the sign-in page shows, so it stays short and on one line."""
    try:
        name = str(json.loads(info).get("client_name") or "") if info else ""
    except (ValueError, AttributeError):
        return None
    return "".join(c for c in name if c.isprintable()).strip()[:60] or None


_HEADERS = {"Cache-Control": "no-store", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; frame-ancestors 'none'"}

# The front page runs one script, the copy button, and shows one picture.
_HOME_HEADERS = dict(_HEADERS, **{"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; "
                                  "script-src 'unsafe-inline'; frame-ancestors 'none'", "Cache-Control": "public, max-age=300"})

_TEXT = {
    "es": {
        "lang": "es", "brand": "INFINITO · tu memoria",
        "asking": "Te lo pide: <b>{who}</b>. Al entrar, esa app podrá usar tu memoria.",
        "signin": "Entrar en tu memoria", "signup": "Crea tu memoria", "recover": "Cambiar la contraseña",
        "name": "Tu nombre de usuario", "name_hint": "Letras minúsculas, números y guiones.",
        "password": "Contraseña", "new_password": "Contraseña nueva", "repeat": "Repítela",
        "password_hint": "Al menos {n} caracteres. Una frase de varias palabras es fácil de recordar y difícil de adivinar.",
        "code": "Código de recuperación",
        "old": "¿Ya usabas INFINITO con una dirección privada?",
        "old_field": "Pégala aquí y tu memoria vendrá contigo (opcional)",
        "go_signin": "Entrar", "go_signup": "Crear mi memoria", "go_recover": "Cambiar la contraseña y entrar",
        "to_signup": "¿Primera vez? Crea tu memoria", "to_signin": "¿Ya tienes memoria? Entra",
        "to_recover": "He olvidado la contraseña",
        "privacy": "Tu memoria se guarda cifrada con una llave que solo se abre con tu contraseña. "
                   "Ni quien administra este servidor puede leerla.",
        "bad_name": "Elige un nombre de usuario de 2 a 32 letras minúsculas, números o guiones.",
        "wait": "Demasiados intentos. Espera {n} segundos y vuelve a probar.",
        "short": "La contraseña necesita al menos {n} caracteres.",
        "differ": "Las dos contraseñas no coinciden.",
        "taken": "Ese nombre ya está en uso. Elige otro, o entra si es tuyo.",
        "bad_password": "Ese nombre y esa contraseña no abren ninguna memoria.",
        "no_such": "Todavía no hay ninguna memoria llamada «{name}». Créala ahora: escribe tu contraseña dos veces.",
        "bad_code": "Ese nombre y ese código de recuperación no abren ninguna memoria.",
        "expired_title": "Esta página ha caducado",
        "expired": "Vuelve a la app (Claude, ChatGPT…) y conecta INFINITO otra vez.",
        "code_title": "Guarda tu código de recuperación",
        "code_text": "Si olvidas la contraseña, este código es la única forma de recuperar tu memoria. "
                     "Apúntalo en papel o guárdalo en tu gestor de contraseñas. No se volverá a mostrar.",
        "moved": "Tu memoria anterior ya está en esta cuenta, y la dirección privada ha dejado de funcionar.",
        "not_moved": "Esa dirección privada no funciona, así que tu memoria empieza vacía.",
        "continue": "Ya lo he guardado, continuar",
    },
    "en": {
        "lang": "en", "brand": "INFINITO · your memory",
        "asking": "Asked by: <b>{who}</b>. Once you sign in, that app can use your memory.",
        "signin": "Sign in to your memory", "signup": "Create your memory", "recover": "Change your password",
        "name": "Your username", "name_hint": "Lowercase letters, digits and dashes.",
        "password": "Password", "new_password": "New password", "repeat": "Repeat it",
        "password_hint": "At least {n} characters. A sentence of a few words is easy to remember and hard to guess.",
        "code": "Recovery code",
        "old": "Did you use INFINITO with a private address?",
        "old_field": "Paste it here and your memory comes with you (optional)",
        "go_signin": "Sign in", "go_signup": "Create my memory", "go_recover": "Change the password and sign in",
        "to_signup": "First time? Create your memory", "to_signin": "Already have a memory? Sign in",
        "to_recover": "I forgot my password",
        "privacy": "Your memory is stored encrypted with a key that only your password opens. "
                   "Not even whoever runs this server can read it.",
        "bad_name": "Choose a username of 2 to 32 lowercase letters, digits or dashes.",
        "wait": "Too many attempts. Wait {n} seconds and try again.",
        "short": "The password needs at least {n} characters.",
        "differ": "The two passwords differ.",
        "taken": "That name is taken. Choose another, or sign in if it is yours.",
        "bad_password": "That name and password open no memory.",
        "no_such": "There is no memory called “{name}” yet. Create it now: type your password twice.",
        "bad_code": "That name and recovery code open no memory.",
        "expired_title": "This page has expired",
        "expired": "Go back to the app (Claude, ChatGPT…) and connect INFINITO again.",
        "code_title": "Keep your recovery code",
        "code_text": "If you forget your password, this code is the only way back into your memory. "
                     "Write it down or keep it in your password manager. It will not be shown again.",
        "moved": "Your earlier memory is now in this account, and the private address no longer works.",
        "not_moved": "That private address does not work, so your memory starts empty.",
        "continue": "I have kept it, continue",
    },
}

_FORMS = {
    "signin": ["name", "password"],
    "signup": ["name", "password", "repeat", "old"],
    "recover": ["name", "code", "password", "repeat"],
}


def _texts(request) -> dict:
    wanted = request.headers.get("accept-language", "").lower()
    return _TEXT["es" if wanted.startswith("es") or ",es" in wanted.replace(" ", "") else "en"]


_STYLE = """
:root { color-scheme: dark; --ink: #eaf0ff; --soft: #aab6d8; --line: rgba(130, 160, 255, .26); --page: #070b1c;
  --card: rgba(15, 21, 50, .82); --accent: #67e8f9; --accent-2: #a78bfa; --bad: #ff8a80; }
* { box-sizing: border-box; }
body { margin: 0; min-height: 100vh; color: var(--ink); font: 16px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif;
  background: radial-gradient(60rem 36rem at 85% -5%, rgba(124, 58, 237, .32), transparent 70%),
    radial-gradient(52rem 34rem at 5% 105%, rgba(14, 165, 233, .24), transparent 70%), var(--page);
  background-attachment: fixed; }
main { max-width: 26rem; margin: 2.5rem auto; padding: 0 1rem; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 1.5rem;
  box-shadow: 0 0 44px rgba(96, 120, 255, .16), inset 0 0 0 1px rgba(255, 255, 255, .03); }
.brand { color: var(--accent); font-size: .78rem; font-weight: 600; letter-spacing: .24em; text-transform: uppercase;
  margin: 0 0 .35rem; }
.brand::before { content: "∞"; margin-right: .5em; font-size: 1.15rem; letter-spacing: 0; vertical-align: -.08em; }
h1 { font-size: 1.35rem; margin: 0 0 1rem; }
label { display: block; font-weight: 600; margin: 1rem 0 .3rem; }
input[type=text], input[type=password] { width: 100%; padding: .65rem .75rem; font: inherit; color: inherit;
  background: transparent; border: 1px solid var(--line); border-radius: 9px; }
input:focus { outline: 2px solid var(--accent); outline-offset: 1px; }
.hint, .note { color: var(--soft); font-size: .88rem; margin: .3rem 0 0; }
.error { color: var(--bad); font-weight: 600; margin: 1rem 0 0; }
button { width: 100%; margin-top: 1.4rem; padding: .75rem; font: inherit; font-weight: 600; color: #071022;
  background: linear-gradient(90deg, var(--accent), var(--accent-2)); border: 0; border-radius: 9px; cursor: pointer; }
button:hover { filter: brightness(1.12); }
button:focus-visible, a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.links { margin-top: 1.1rem; display: flex; flex-direction: column; gap: .4rem; font-size: .92rem; }
a { color: var(--accent); }
details { margin-top: 1rem; } summary { cursor: pointer; color: var(--soft); font-size: .92rem; }
.code { font: 600 1.35rem/1.4 ui-monospace, Consolas, monospace; letter-spacing: .06em; text-align: center;
  padding: 1rem; margin: 1rem 0; border: 1px dashed var(--accent); border-radius: 10px; word-break: break-all; }
"""


def _page(t: dict, title: str, body: str, status: int = 200, style: str = ""):
    from starlette.responses import HTMLResponse

    return HTMLResponse(
        f'<!doctype html><html lang="{t["lang"]}"><head><meta charset="utf-8">'
        f'<meta name="viewport" content="width=device-width, initial-scale=1"><title>{html.escape(title)}</title>'
        f'<style>{_STYLE}{style}</style></head><body><main><div class="card"><p class="brand">{t["brand"]}</p>'
        f'<h1>{html.escape(title)}</h1>{body}</div></main></body></html>', status_code=status, headers=_HEADERS)


def _form(t: dict, request_token: str, mode: str, who: str, name: str = "", error: str = "", status: int = 200):
    e = html.escape
    fields = []
    for field in _FORMS[mode]:
        if field == "name":
            fields.append(f'<label for="name">{t["name"]}</label><input id="name" name="name" type="text" '
                          f'value="{e(name)}" autocomplete="username" autocapitalize="none" spellcheck="false" '
                          f'required{"" if name else " autofocus"}>'
                          + (f'<p class="hint">{t["name_hint"]}</p>' if mode == "signup" else ""))
        elif field == "password":
            label = t["new_password"] if mode == "recover" else t["password"]
            complete = "current-password" if mode == "signin" else "new-password"
            fields.append(f'<label for="password">{label}</label><input id="password" name="password" '
                          f'type="password" autocomplete="{complete}" required{" autofocus" if name else ""}>'
                          + ("" if mode == "signin" else
                             f'<p class="hint">{t["password_hint"].format(n=MIN_PASSWORD)}</p>'))
        elif field == "repeat":
            fields.append(f'<label for="repeat">{t["repeat"]}</label><input id="repeat" name="repeat" '
                          f'type="password" autocomplete="new-password" required>')
        elif field == "code":
            fields.append(f'<label for="code">{t["code"]}</label><input id="code" name="code" type="text" '
                          f'autocomplete="off" autocapitalize="characters" spellcheck="false" required>')
        elif field == "old":
            fields.append(f'<details><summary>{t["old"]}</summary><label for="old">{t["old_field"]}</label>'
                          f'<input id="old" name="old_address" type="password" autocomplete="off"></details>')
    r = e(request_token)
    links = {"signin": [("signup", "to_signup"), ("recover", "to_recover")],
             "signup": [("signin", "to_signin")], "recover": [("signin", "to_signin")]}[mode]
    body = (f'<p class="note">{t["asking"].format(who=e(who))}</p>'
            f'<form method="post" action="/login"><input type="hidden" name="r" value="{r}">'
            f'<input type="hidden" name="action" value="{mode}">{"".join(fields)}'
            + (f'<p class="error" role="alert">{e(error)}</p>' if error else "")
            + f'<button type="submit">{t["go_" + mode]}</button></form>'
            f'<div class="links">' + "".join(f'<a href="/login?r={r}&amp;m={m}">{t[k]}</a>' for m, k in links)
            + f'</div><p class="note" style="margin-top:1.2rem">{t["privacy"]}</p>')
    return _page(t, t[mode], body, status)


def _recovery_page(t: dict, code: str, onward: str, *, moved: bool, old_given: bool):
    e = html.escape
    note = (f'<p class="note">{t["moved"]}</p>' if moved
            else f'<p class="error">{t["not_moved"]}</p>' if old_given else "")
    body = (f'<p>{t["code_text"]}</p><div class="code">{e(code)}</div>{note}'
            f'<form method="post" action="/login"><input type="hidden" name="action" value="continue">'
            f'<input type="hidden" name="c" value="{e(onward)}"><button type="submit">{t["continue"]}</button></form>')
    return _page(t, t["code_title"], body)


# The server's front page: the link to send someone, with the steps for each app.
