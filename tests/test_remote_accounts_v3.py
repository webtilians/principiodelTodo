"""Signing in to INFINITO's remote memory: one public address, a password per person, OAuth for every app.

The apps here are the MCP SDK's own OAuth client, as Claude and ChatGPT use one: it finds the sign-in
from the server's 401, registers itself, sends the person to the page and gets its tokens."""
import json
import re
import socket
import threading
import time
from urllib.parse import parse_qs, urlparse

import pytest

from src.infinito3.remote import People, SealedMemories

pytest.importorskip("mcp.server.mcpserver")
pytest.importorskip("cryptography")
anyio = pytest.importorskip("anyio")
httpx2 = pytest.importorskip("httpx2")
uvicorn = pytest.importorskip("uvicorn")

PASSWORD = "mi casa tiene un limonero"


@pytest.fixture
def server(tmp_path):
    from src.infinito3.remote import build_app

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    people = People(tmp_path)
    running = uvicorn.Server(uvicorn.Config(build_app(people, base), host="127.0.0.1", port=port,
                                            log_level="warning"))
    thread = threading.Thread(target=running.run, daemon=True)
    thread.start()
    while not running.started:
        time.sleep(0.05)
    yield base, people, tmp_path
    running.should_exit = True
    thread.join(timeout=10)


class Storage:
    def __init__(self):
        self.tokens, self.client = None, None

    async def get_tokens(self):
        return self.tokens

    async def set_tokens(self, tokens):
        self.tokens = tokens

    async def get_client_info(self):
        return self.client

    async def set_client_info(self, client_info):
        self.client = client_info


def sign_in_page(base, authorize_url, *, mode, name, password=PASSWORD, extra=None):
    """What the person does in the browser: the app's link leads to INFINITO's page, they fill it in."""
    with httpx2.Client() as browser:
        login = browser.get(authorize_url).headers["location"]
        request = parse_qs(urlparse(login).query)["r"][0]
        page = browser.get(login, headers={"accept-language": "es-ES"})
        assert page.status_code == 200 and "Prueba" in page.text  # the page says which app asks
        form = {"r": request, "action": mode, "name": name, "password": password, "repeat": password, **(extra or {})}
        answer = browser.post(f"{base}/login", data=form, headers={"accept-language": "es-ES"})
        recovery = None
        if answer.status_code == 200 and "class=\"code\"" in answer.text:  # a new memory: its recovery code first
            recovery = re.search(r'class="code">([A-Z0-9-]+)<', answer.text).group(1)
            onward = re.search(r'name="c" value="([^"]+)"', answer.text).group(1)
            answer = browser.post(f"{base}/login", data={"action": "continue", "c": onward})
        return answer, recovery


def app_session(base, steps, *, mode="signin", name="ana", password=PASSWORD, extra=None, storage=None):
    """An app connecting to base/mcp: it is sent to sign in, then calls the tools."""
    from mcp import ClientSession
    from mcp.client.auth import AuthorizationCodeResult, OAuthClientProvider
    from mcp.client.streamable_http import streamable_http_client
    from mcp.shared.auth import OAuthClientMetadata

    seen = {}

    async def redirect(url):
        seen["authorize"] = url

    async def callback():
        answer, seen["recovery"] = await anyio.to_thread.run_sync(
            lambda: sign_in_page(base, seen["authorize"], mode=mode, name=name, password=password, extra=extra))
        assert answer.status_code == 303, answer.text
        query = parse_qs(urlparse(answer.headers["location"]).query)
        return AuthorizationCodeResult(code=query["code"][0], state=query.get("state", [None])[0])

    storage = storage or Storage()
    auth = OAuthClientProvider(f"{base}/mcp", OAuthClientMetadata(
        client_name="Prueba", redirect_uris=["http://localhost:3030/callback"]), storage, redirect, callback)

    async def run():
        async with httpx2.AsyncClient(auth=auth, timeout=60) as http:
            async with streamable_http_client(f"{base}/mcp", http_client=http) as (read, write):
                async with ClientSession(read, write) as client:
                    await client.initialize()
                    storage.tools = (await client.list_tools()).tools
                    storage.results = [await client.call_tool(tool, arguments) for tool, arguments in steps]
                    return [result.content[0].text for result in storage.results]

    return anyio.run(run), seen.get("recovery"), storage


def test_an_app_asks_to_sign_in_and_the_server_says_where(server):
    base, _, _ = server
    answer = httpx2.post(f"{base}/mcp", json={})
    assert answer.status_code == 401
    assert "resource_metadata" in answer.headers["www-authenticate"]
    metadata = httpx2.get(f"{base}/.well-known/oauth-protected-resource/mcp").json()
    assert metadata["resource"] == f"{base}/mcp"
    server = httpx2.get(f"{base}/.well-known/oauth-authorization-server").json()
    assert server["registration_endpoint"] == f"{base}/register" and server["scopes_supported"] == ["memory"]
    assert "none" in server["token_endpoint_auth_methods_supported"]  # apps without a secret, like ChatGPT


def test_one_password_opens_the_same_memory_from_every_app_and_nothing_on_disk_reads(server):
    base, _, folder = server
    [stored], recovery, first_app = app_session(
        base, [("remember", {"statement": "Vivo en Valencia y trabajo de enfermera."})], mode="signup")
    assert stored.startswith("Remembered") and re.fullmatch(r"([A-Z0-9]{4}-){4}[A-Z0-9]{4}", recovery)
    [listed], _, _ = app_session(base, [("list_memories", {})])  # another app, another sign-in
    assert "Valencia" in listed

    on_disk = b"".join(p.read_bytes() for p in folder.rglob("*") if p.is_file())
    assert b"Valencia" not in on_disk
    assert PASSWORD.encode() not in on_disk and recovery.encode() not in on_disk
    assert first_app.tokens.access_token.encode() not in on_disk  # tokens are kept only as hashes
    assert first_app.tokens.refresh_token.encode() not in on_disk


def test_a_wrong_password_opens_nothing(server):
    base, _, _ = server
    app_session(base, [("remember", {"statement": "Me llamo Ana."})], mode="signup")
    registered = httpx2.post(f"{base}/register", json={"redirect_uris": ["http://localhost:3030/callback"],
                                                        "client_name": "Prueba"}).json()
    authorize = httpx2.get(f"{base}/authorize", params={
        "client_id": registered["client_id"], "response_type": "code", "code_challenge": "x" * 43,
        "code_challenge_method": "S256", "redirect_uri": "http://localhost:3030/callback"})
    request = parse_qs(urlparse(authorize.headers["location"]).query)["r"][0]
    wrong = httpx2.post(f"{base}/login", data={"r": request, "action": "signin", "name": "ana",
                                                "password": "no es esta contraseña"},
                        headers={"accept-language": "es"})
    assert wrong.status_code == 400 and "no abren ninguna memoria" in wrong.text
    assert "Location" not in wrong.headers  # no code goes back to the app


def test_the_recovery_code_sets_a_new_password_keeps_the_memory_and_signs_apps_out(server):
    base, _, _ = server
    _, recovery, first_app = app_session(base, [("remember", {"statement": "Mi perra se llama Nala."})],
                                         mode="signup")
    new = "otra frase que sí recuerdo"
    [listed], _, _ = app_session(base, [("list_memories", {})], mode="recover", password=new,
                                 extra={"code": recovery.lower().replace("-", " ")})
    assert "Nala" in listed
    old_token = first_app.tokens.access_token
    assert httpx2.post(f"{base}/mcp", json={}, headers={"authorization": f"Bearer {old_token}"}).status_code == 401
    [again], _, _ = app_session(base, [("list_memories", {})], password=new)
    assert "Nala" in again


def test_a_private_address_memory_comes_along_when_its_owner_signs_up(server):
    base, people, _ = server
    token = people.add("luis")
    SealedMemories(people).use("luis", people.key("luis", token),
                               lambda m: m.remember("Prefiero las explicaciones breves."), writes=True)
    address = f"{base}/u/{token}/mcp"
    [listed], _, _ = app_session(base, [("list_memories", {})], mode="signup", name="luis",
                                 extra={"old_address": address})
    assert "breves" in listed
    assert people.who(token) is None and httpx2.post(address, json={}).status_code == 404


def test_when_the_access_runs_out_the_app_renews_it_without_asking_again(server):
    base, _, _ = server
    _, _, storage = app_session(base, [("remember", {"statement": "Mi color favorito es el verde."})], mode="signup")
    renew = {"grant_type": "refresh_token", "client_id": storage.client.client_id,
             "refresh_token": storage.tokens.refresh_token}
    if storage.client.client_secret:
        renew["client_secret"] = storage.client.client_secret
    renewed = httpx2.post(f"{base}/token", data=renew)
    assert renewed.status_code == 200, renewed.text
    fresh = renewed.json()
    assert httpx2.post(f"{base}/token", data=renew).status_code == 400  # each refresh token works once
    tools = httpx2.post(f"{base}/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                             "params": {"name": "list_memories", "arguments": {}}},
                        headers={"authorization": f"Bearer {fresh['access_token']}",
                                 "accept": "application/json, text/event-stream",
                                 "mcp-protocol-version": "2025-06-18"})
    assert tools.status_code == 200 and "verde" in tools.text


def test_signing_in_with_a_name_that_does_not_exist_offers_to_create_it(server):
    base, _, _ = server
    registered = httpx2.post(f"{base}/register", json={"redirect_uris": ["http://localhost:3030/callback"],
                                                        "client_name": "Prueba"}).json()
    authorize = httpx2.get(f"{base}/authorize", params={
        "client_id": registered["client_id"], "response_type": "code", "code_challenge": "x" * 43,
        "code_challenge_method": "S256", "redirect_uri": "http://localhost:3030/callback"})
    request = parse_qs(urlparse(authorize.headers["location"]).query)["r"][0]
    page = httpx2.post(f"{base}/login", data={"r": request, "action": "signin", "name": "nueva",
                                               "password": PASSWORD}, headers={"accept-language": "es"})
    assert page.status_code == 200 and "<h1>Crea tu memoria</h1>" in page.text
    assert 'value="nueva"' in page.text and "Todavía no hay ninguna memoria llamada «nueva»" in page.text


def test_the_front_page_gives_the_address_and_the_steps_for_each_app(server):
    base, _, _ = server
    page = httpx2.get(f"{base}/", headers={"accept-language": "es-ES,es;q=0.9"})
    assert page.status_code == 200
    assert f'value="{base}/mcp"' in page.text
    assert "En Claude" in page.text and "En ChatGPT" in page.text and "codex mcp login infinito" in page.text
    assert "plan de pago" in page.text and "@INFINITO" in page.text  # what ChatGPT needs, before someone starts
    assert "Guarda correos y teléfonos" in page.text  # and what the memory leaves out
    assert "Instrucciones personalizadas" in page.text and "remember_many" in page.text  # no @ in every message
    english = httpx2.get(f"{base}/", headers={"accept-language": "en-GB"}).text
    assert "One memory" in english and "paid plan" in english and "keeps emails and phone numbers" in english


def test_the_front_page_makes_you_want_it_and_says_nothing_it_cannot_back(server):
    base, _, _ = server
    for lang, slogan, grok_step, gemini_step in (
            ("es", "Esto es solo el principio de tu agente definitivo", "grok.com/connectors", "gemini mcp add"),
            ("en", "This is only the beginning of your ultimate agent", "grok.com/connectors", "gemini mcp add")):
        page = httpx2.get(f"{base}/", headers={"accept-language": lang})
        assert slogan in page.text and grok_step in page.text and gemini_step in page.text and "/mcp auth infinito" in page.text
        for app in ("Claude", "ChatGPT", "Grok", "Gemini", "Codex"):
            assert f'<div class="app">{app}' in page.text
        assert "45" in page.text and "A-MEM" in page.text and "arxiv.org/abs/2610.18052" in page.text
        assert page.text.count("<script") == 1 and "img-src 'self'" in page.headers["content-security-policy"]
        assert '/memory/brain.webp' in page.text and 'class="beta"' in page.text  # Gemini until it is proven end to end
        text = page.text.lower()
        for false in ("extremo a extremo", "end-to-end", "end to end", "ni quien administra", "whoever runs this server"):
            assert false not in text, false


def test_apps_are_told_which_tools_only_read_so_they_need_not_ask_first(server):
    base, _, _ = server
    _, _, app = app_session(base, [], mode="signup", name="pepa")
    reads = {tool.name: tool.annotations.read_only_hint for tool in app.tools}
    assert reads == {"recall": True, "list_memories": True, "remember": False, "remember_many": False,
                     "forget": False, "vault_add": True, "vault_reveal": True, "vault_list": True,
                     # These two only answer with a link: nothing changes until the person acts on the page.
                     "memory_page": True, "vault_delete": True,
                     "overview": True,
                     "photo_save": False, "photo_show": True}
    assert [t.annotations.destructive_hint for t in app.tools if t.name == "forget"] == [True]


def test_an_app_that_asks_for_other_scopes_can_still_register(server):
    base, _, _ = server
    answer = httpx2.post(f"{base}/register", json={"redirect_uris": ["https://chatgpt.com/connector_platform_oauth_redirect"],
                                                    "client_name": "ChatGPT", "scope": "openid profile email"})
    assert answer.status_code == 201, answer.text
    for alias in ("/.well-known/openid-configuration", "/.well-known/oauth-authorization-server/mcp"):
        assert httpx2.get(f"{base}{alias}").json()["registration_endpoint"] == f"{base}/register"


def test_every_tool_says_it_needs_sign_in_and_nothing_answers_without_it(server):
    base, _, _ = server
    _, _, app = app_session(base, [], mode="signup", name="lola")
    for tool in app.tools:
        assert tool.meta["securitySchemes"] == [{"type": "oauth2", "scopes": ["memory"]}], tool.name
    hello = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}}}
    for method in ("initialize", "tools/list"):
        answer = httpx2.post(f"{base}/mcp", json=dict(hello, method=method),
                             headers={"accept": "application/json, text/event-stream"})
        assert answer.status_code == 401, method  # clients learn they must sign in from this answer


def sign_this_computer_in(base, tmp_path, monkeypatch):
    from src.infinito3 import remote_client

    monkeypatch.setenv("INFINITO_REMOTE_SESSION", str(tmp_path / "session.json"))

    def person_signs_in(page):  # what the person does in the browser the command opens
        with httpx2.Client() as browser:
            login = browser.get(page).headers["location"]
            request = parse_qs(urlparse(login).query)["r"][0]
            back = browser.post(f"{base}/login", data={"r": request, "action": "signin", "name": "ana",
                                                        "password": PASSWORD})
            browser.get(back.headers["location"])  # the browser returns to the command's local page

    assert "is signed in" in remote_client.login(f"{base}/mcp", open_page=person_signs_in, wait_seconds=20)


def test_this_computer_signs_in_once_and_every_message_brings_the_memory(server, tmp_path, monkeypatch):
    import io

    from src.infinito3 import remote_client

    base, _, _ = server
    app_session(base, [("remember", {"statement": "Mi perra se llama Nena."})], mode="signup")
    sign_this_computer_in(base, tmp_path, monkeypatch)

    def hook(prompt):
        out = io.StringIO()
        event = json.dumps({"prompt": prompt, "hook_event_name": "UserPromptSubmit"}, ensure_ascii=False)
        remote_client.context(io.BytesIO(event.encode("utf-8")), out)  # as Claude Code sends it: UTF-8 bytes
        assert out.getvalue().isascii()  # readable whatever code page the console uses
        return json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"] if out.getvalue() else None

    assert "Nena" in hook("¿Cómo se llama mi perra?")
    assert "Mi perra se llama Nena." in hook("¿Cómo se llama mi perra?")  # accents and all
    session = json.loads((tmp_path / "session.json").read_text())
    session["expires_at"] = 0  # its hour is over: the hook renews the access by itself
    (tmp_path / "session.json").write_text(json.dumps(session))
    assert "Nena" in hook("y mi perra?")
    assert json.loads((tmp_path / "session.json").read_text())["refresh_token"] != session["refresh_token"]
    assert remote_client.logout().startswith("Signed out")
    assert hook("¿Cómo se llama mi perra?") is None  # no session: the message goes on without memory


def test_the_hook_never_stops_a_message(tmp_path, monkeypatch):
    import io

    from src.infinito3 import remote_client

    monkeypatch.setenv("INFINITO_REMOTE_SESSION", str(tmp_path / "session.json"))
    (tmp_path / "session.json").write_text(json.dumps({
        "server": "http://127.0.0.1:9", "client_id": "x", "token_endpoint": "http://127.0.0.1:9/token",
        "refresh_token": "x", "access_token": "x", "expires_at": 0}))
    out = io.StringIO()
    remote_client.context(io.StringIO('{"prompt": "hola"}'), out)  # nothing listens there
    assert out.getvalue() == ""


def test_a_sessions_first_message_also_brings_what_was_left_pending(server, tmp_path, monkeypatch):
    import io

    from src.infinito3 import remote_client

    base, _, _ = server
    app_session(base, [("remember", {"statement": "Mi perra se llama Nena."}),
                       ("remember", {"statement": "Abrir el PR de la rama nueva.", "kind": "pending", "project": "Acme"})],
                mode="signup")
    sign_this_computer_in(base, tmp_path, monkeypatch)
    transcript = tmp_path / "session.jsonl"

    def hook():
        out = io.StringIO()
        event = json.dumps({"prompt": "¿Cómo se llama mi perra?", "transcript_path": str(transcript),
                            "hook_event_name": "UserPromptSubmit"}, ensure_ascii=False)
        remote_client.context(io.BytesIO(event.encode("utf-8")), out)
        return json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]

    first = hook()  # the transcript is not even written yet
    assert "Nena" in first and remote_client.PENDING_HEADER in first
    assert "Abrir el PR" in first.split(remote_client.PENDING_HEADER)[1]
    transcript.write_text('{"type":"user"}\n', encoding="utf-8")
    assert remote_client.PENDING_HEADER in hook()  # the agent has not answered yet
    transcript.write_text('{"type":"user"}\n{"type":"assistant"}\n', encoding="utf-8")
    later = hook()
    assert "Nena" in later and remote_client.PENDING_HEADER not in later


def test_the_hook_never_outgrows_what_claude_code_accepts(monkeypatch):
    """Over 10,000 characters Claude Code keeps a 2 KB preview, so long memories must not cost all the others."""
    import io

    from src.infinito3 import remote_client

    short = '[abc] (2026-11-05, fact) "Mi perra se llama Nena."'
    long_ones = [f'[id{i}] (2026-11-0{i}, pending) "Memoria larga {i}: ' + "detalle " * 300 + '"' for i in range(1, 9)]
    monkeypatch.setattr(remote_client, "recall", lambda *_, **__: "\n".join(["20 memories:", short, *long_ones]))
    out = io.StringIO()
    remote_client.context(io.BytesIO(b'{"prompt": "hola"}'), out)
    told = json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]

    assert len(told) <= remote_client.HOOK_CHARS
    assert short in told  # a short memory is never what pays for the long ones
    assert all(f'[id{i}] (2026-11-0{i}, pending) "Memoria larga {i}' in told for i in range(1, 9))
    assert told.count(remote_client.LINE_CUT) == 9 and told.endswith(remote_client.LINES_CUT)  # 8 lines and the notice

    small = "\n".join(["2 memories:", short])
    assert remote_client._fit(small) == small  # what fits stays as it is


EDIT =("Edit", {"file_path": r"C:\repo\src\app.py"})
SAVE = ("mcp__abc__remember_many", {"items": []})
PUSH = ("Bash", {"command": "git -C repo push origin feat/x"})


def write_transcript(path, uses):
    """A session's transcript, from what its agent did: (minutes ago, tool, input)."""
    rows = ['{"type":"user","message":{"content":"hola"}}']
    for minutes, name, given in uses:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime(time.time() - minutes * 60))
        rows.append(json.dumps({"type": "assistant", "timestamp": stamp, "message": {"content": [
            {"type": "text", "text": "ok"}, {"type": "tool_use", "name": name, "input": given}]}}))
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def stopping(path, **event):
    import io

    from src.infinito3 import remote_client

    out = io.StringIO()
    remote_client.checkpoint(io.BytesIO(json.dumps({"transcript_path": str(path), **event}).encode()), out)
    return json.loads(out.getvalue()) if out.getvalue() else None


@pytest.mark.parametrize("uses, asks", [
    ([(1, "Read", {"file_path": "x"})], False),  # it only looked
    ([(1, *EDIT)], True),  # work, and nothing saved in this session
    ([(2, *EDIT), (1, *SAVE)], False),  # already saved
    ([(5, *SAVE), (1, *EDIT)], False),  # saved a moment ago: not after every small edit
    ([(31, *SAVE), (1, *EDIT)], True),  # saved long ago
    ([(5, *SAVE), (1, *PUSH)], True),  # shipping always asks: a session often ends there
    ([(1, "PowerShell", {"command": 'git commit -m "x"'})], True),
    ([(1, "Bash", {"command": "git log --oneline; echo push"})], False),  # looking at git is not shipping
    ([(1, "Write", {"file_path": r"C:\Users\x\.claude\projects\p\memory\a.md"}),
      (1, "Edit", {"file_path": r"C:\Users\x\AppData\Local\Temp\a.py"})], False),  # its notes and scratch files
])
def test_a_session_is_asked_to_save_where_it_stands_after_work_only(tmp_path, uses, asks):
    from src.infinito3 import remote_client

    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript, uses)
    answer = stopping(transcript)
    assert answer == ({"decision": "block", "reason": remote_client.SAVE_REASON} if asks else None)


def test_a_session_is_asked_only_once_and_never_kept_from_stopping(tmp_path):
    import io

    from src.infinito3 import remote_client

    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript, [(1, *EDIT)])
    assert stopping(transcript) is not None
    assert stopping(transcript, stop_hook_active=True) is None  # it was asked: this stop goes through
    assert stopping(tmp_path / "missing.jsonl") is None
    out = io.StringIO()
    remote_client.checkpoint(io.BytesIO(b"not json"), out)
    assert out.getvalue() == ""


def test_the_save_check_reads_only_the_end_of_a_long_transcript(tmp_path, monkeypatch):
    from src.infinito3 import remote_client

    transcript = tmp_path / "session.jsonl"
    write_transcript(transcript, [(90, *SAVE), (1, *EDIT)])
    size = transcript.stat().st_size
    last_row = transcript.read_text(encoding="utf-8").splitlines()[-1]
    monkeypatch.setattr(remote_client, "TRANSCRIPT_TAIL", len(last_row) + 40)  # cuts the row before it in two
    assert size > remote_client.TRANSCRIPT_TAIL
    assert stopping(transcript) is not None  # the cut row is skipped; the edit at the end is found


def test_an_app_saves_a_whole_meeting_and_asks_about_one_client(server):
    base, _, folder = server
    steps = [
        ("remember_many", {"items": [
            {"statement": "Acme aprueba el presupuesto de la web.", "kind": "decision"},
            {"statement": "Reclamar las fotos al fotógrafo.", "kind": "pending"},
            {"statement": "Quizá hacer también un vídeo.", "kind": "proposal"},
        ], "source": "reunión con Marta", "project": "Acme", "date": "2026-10-26"}),
        ("remember", {"statement": "Falta el contrato.", "kind": "pending", "project": "Globex"}),
        ("recall", {"query": "¿por dónde íbamos?", "project": "Acme"}),
        ("recall", {"query": "qué falta", "project": "Acme", "kind": "pending"}),
        ("list_memories", {"project": "Globex"}),
    ]
    [saved, _, where, pending, other], _, app = app_session(base, steps, mode="signup", name="marta")
    assert saved == "Remembered 3 of 3."
    assert "(2026-10-26, decision, Acme, source: reunión con Marta, by: Prueba)" in where and "Globex" not in where
    assert "fotógrafo" in pending and "presupuesto" not in pending
    assert "contrato" in other and "Acme" not in other
    schema = next(tool for tool in app.tools if tool.name == "remember_many").input_schema
    assert "statement" in str(schema) and "kind" in str(schema)  # the app is told what each item takes
    on_disk = b"".join(p.read_bytes() for p in folder.rglob("*") if p.is_file())
    assert "fotógrafo".encode() not in on_disk and b"presupuesto" not in on_disk  # sealed, like everything else
