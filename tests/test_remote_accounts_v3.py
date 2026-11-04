"""Signing in to INFINITO's remote memory: one public address, a password per person, OAuth for every app.

The apps here are the MCP SDK's own OAuth client, as Claude and ChatGPT use one: it finds the sign-in
from the server's 401, registers itself, sends the person to the page and gets its tokens."""
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
                    return [(await client.call_tool(tool, arguments)).content[0].text for tool, arguments in steps]

    return anyio.run(run), seen.get("recovery"), storage


def test_an_app_asks_to_sign_in_and_the_server_says_where(server):
    base, _, _ = server
    answer = httpx2.post(f"{base}/mcp", json={})
    assert answer.status_code == 401
    assert "resource_metadata" in answer.headers["www-authenticate"]
    metadata = httpx2.get(f"{base}/.well-known/oauth-protected-resource/mcp").json()
    assert metadata["resource"] == f"{base}/mcp"
    assert httpx2.get(f"{base}/.well-known/oauth-authorization-server").json()["registration_endpoint"]


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
    assert "One memory" in httpx2.get(f"{base}/", headers={"accept-language": "en-GB"}).text


def test_apps_are_told_which_tools_only_read_so_they_need_not_ask_first(server):
    base, _, _ = server
    _, _, app = app_session(base, [], mode="signup", name="pepa")
    reads = {tool.name: tool.annotations.read_only_hint for tool in app.tools}
    assert reads == {"recall": True, "list_memories": True, "remember": False, "forget": False}
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
        assert tool.meta == {"securitySchemes": [{"type": "oauth2", "scopes": ["memory"]}]}, tool.name
    hello = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "x", "version": "1"}}}
    for method in ("initialize", "tools/list"):
        answer = httpx2.post(f"{base}/mcp", json=dict(hello, method=method),
                             headers={"accept": "application/json, text/event-stream"})
        assert answer.status_code == 401, method  # clients learn they must sign in from this answer
