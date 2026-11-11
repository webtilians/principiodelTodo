"""The engine's connection to the person's INFINITO: an app of its own, signed in once with the person's password.

The sign-in is the same OAuth flow as `infinito-remote login` (remote_client.py), under the engine's own name, so it
shows up on its own line on the memory page, under Apps, where the person gives it a level ("read and add" is all
it needs: it reads, and writes its reflections) or disconnects it. Its session is a refresh token kept in the
engine's folder, never on the server; the server only ever sees it when the engine calls.
"""
import base64
import hashlib
import http.server
import json
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from pathlib import Path
from typing import Callable, Optional

TIMEOUT = 60


class InfinitoError(Exception):
    """not_signed_in, refused (the server said no, with its words), transport_error."""


def _request(url: str, data: Optional[dict] = None, *, form: bool = False, token: Optional[str] = None,
             timeout: float = TIMEOUT) -> dict:
    headers = {"Accept": "application/json, text/event-stream"}
    body = None
    if data is not None:
        body = urllib.parse.urlencode(data).encode() if form else json.dumps(data).encode()
        headers["Content-Type"] = "application/x-www-form-urlencoded" if form else "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, body, headers), timeout=timeout) as response:
        raw = response.read().decode("utf-8")
    if raw.lstrip().startswith(("event:", "data:")):  # a streamed answer: its last data line is the message
        raw = [line[5:].strip() for line in raw.splitlines() if line.startswith("data:")][-1]
    return json.loads(raw or "{}")


class Infinito:
    def __init__(self, session_path: Path, request: Callable = _request):
        self.path, self.request = session_path, request
        self._next_id = 1

    # The session

    def session(self) -> Optional[dict]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def _save(self, session: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(session, indent=2), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self.path)

    def _keep(self, session: dict, tokens: dict) -> dict:
        session.update(access_token=tokens["access_token"], expires_at=time.time() + int(tokens.get("expires_in", 3600)),
                       refresh_token=tokens.get("refresh_token") or session.get("refresh_token"))
        self._save(session)
        return session

    def login(self, server: str, app_name: str, open_page: Callable[[str], object] = webbrowser.open,
              wait_seconds: float = 300) -> str:
        """The person signs the engine in on INFINITO's own page; the password never passes through here."""
        server = server.rstrip("/").removesuffix("/mcp")
        metadata = self.request(f"{server}/.well-known/oauth-authorization-server")
        arrived: dict = {}

        class Callback(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                arrived.update(urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query))
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write("<p style='font:18px system-ui;margin:3rem'>INFINITO: el motor cognitivo ya está "
                                 "conectado. Puedes cerrar esta pestaña.</p>".encode())

            def log_message(self, *args):
                pass

        listener = http.server.HTTPServer(("127.0.0.1", 0), Callback)
        redirect = f"http://127.0.0.1:{listener.server_address[1]}/callback"
        client = self.request(metadata["registration_endpoint"], {
            "client_name": app_name, "redirect_uris": [redirect], "token_endpoint_auth_method": "none",
            "grant_types": ["authorization_code", "refresh_token"], "response_types": ["code"], "scope": "memory"})
        verifier = secrets.token_urlsafe(48)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        page = metadata["authorization_endpoint"] + "?" + urllib.parse.urlencode({
            "response_type": "code", "client_id": client["client_id"], "redirect_uri": redirect, "scope": "memory",
            "code_challenge": challenge, "code_challenge_method": "S256", "state": state,
            "resource": f"{server}/mcp"})
        deadline = time.time() + wait_seconds

        def serve():
            while "code" not in arrived and "error" not in arrived and time.time() < deadline:
                listener.handle_request()

        listener.timeout = 1
        waiting = threading.Thread(target=serve, daemon=True)
        waiting.start()
        open_page(page)
        waiting.join(wait_seconds + 5)
        listener.server_close()
        if arrived.get("state", [None])[0] != state or "code" not in arrived:
            raise InfinitoError("not_signed_in")
        tokens = self.request(metadata["token_endpoint"], {
            "grant_type": "authorization_code", "code": arrived["code"][0], "redirect_uri": redirect,
            "client_id": client["client_id"], "code_verifier": verifier, "resource": f"{server}/mcp"}, form=True)
        self._keep({"server": server, "client_id": client["client_id"], "token_endpoint": metadata["token_endpoint"]},
                   tokens)
        return f"Signed in to {server} as «{app_name}». Session kept in {self.path}."

    def logout(self) -> bool:
        if self.path.exists():
            self.path.unlink()
            return True
        return False

    def _token(self, session: dict) -> str:
        if session.get("access_token") and session.get("expires_at", 0) > time.time() + 60:
            return session["access_token"]
        tokens = self.request(session["token_endpoint"], {"grant_type": "refresh_token",
                                                          "client_id": session["client_id"],
                                                          "refresh_token": session["refresh_token"]}, form=True)
        return self._keep(session, tokens)["access_token"]

    # The tools

    def call(self, tool: str, **arguments) -> str:
        """One MCP tool call, answered from the person's own memory. Arguments left as None are not sent."""
        session = self.session()
        if not session:
            raise InfinitoError("not_signed_in")
        try:
            token = self._token(session)
            mcp = f"{session['server']}/mcp"
            self.request(mcp, {"jsonrpc": "2.0", "id": self._bump(), "method": "initialize", "params": {
                "protocolVersion": "2025-06-18", "capabilities": {},
                "clientInfo": {"name": "infinito-cognition", "version": "1"}}}, token=token)
            answer = self.request(mcp, {"jsonrpc": "2.0", "id": self._bump(), "method": "tools/call", "params": {
                "name": tool, "arguments": {k: v for k, v in arguments.items() if v is not None}}}, token=token)
        except urllib.error.HTTPError as error:
            raise InfinitoError(f"http_{error.code}") from None
        except (urllib.error.URLError, TimeoutError, OSError, ValueError, KeyError):
            raise InfinitoError("transport_error") from None
        result = answer.get("result") or {}
        text = "".join(part.get("text", "") for part in result.get("content") or [] if part.get("type") == "text")
        if result.get("isError") or "error" in answer:
            raise InfinitoError(f"refused: {text or answer.get('error', {}).get('message', '')}"[:400])
        return text

    def _bump(self) -> int:
        self._next_id += 1
        return self._next_id
