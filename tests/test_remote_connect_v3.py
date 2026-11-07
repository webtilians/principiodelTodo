"""Contract tests for a zero-secret OAuth/MCP onboarding flow."""
import base64
import json
from urllib.parse import parse_qs, urlparse

import pytest

from src.infinito3.remote_connect import CLIENTS, connect_page, connection_options, server_url
from src.infinito3.remote_installer import commands, install


def test_all_six_clients_are_supported():
    assert CLIENTS == ("chatgpt", "claude", "cursor", "codex", "claude-code", "other")


def test_cursor_one_click_is_the_official_base64_json_format():
    info = connection_options("https://memory.example.org/")
    uri = urlparse(info["cursor"])
    assert uri.scheme == "cursor"
    assert uri.netloc == "anysphere.cursor-deeplink"
    assert uri.path == "/mcp/install"
    args = parse_qs(uri.query)
    assert args["name"] == ["infinito"]
    assert json.loads(base64.b64decode(args["config"][0])) == {"url": "https://memory.example.org/mcp"}
    assert "password" not in info["cursor"] and "token" not in info["cursor"]


def test_public_config_is_the_same_url_across_all_clients():
    info = connection_options("https://memory.example.org/mcp")
    assert info["endpoint"] == "https://memory.example.org/mcp"
    assert json.loads(info["generic"])["mcpServers"]["infinito"]["url"] == info["endpoint"]
    assert info["endpoint"] in info["codex"] and info["endpoint"] in info["claude_code"]


@pytest.mark.parametrize("server", [
    "http://memory.example.org", "https://person:secret@memory.example.org",
    "https://memory.example.org/u/secret/mcp", "https://memory.example.org/?key=secret",
    "https://memory.example.org/#fragment", "https://memory.example.org:abc",
    "file:///etc/passwd", "https://",
])
def test_private_or_invalid_urls_cannot_be_used(server):
    with pytest.raises(ValueError):
        connection_options(server)


def test_http_allowed_only_for_local_development():
    assert server_url("http://127.0.0.1:8500/mcp") == "http://127.0.0.1:8500"
    assert server_url("http://localhost:8500") == "http://localhost:8500"


def test_page_is_localized_has_all_six_clients_and_no_inline_tokens():
    pytest.importorskip("starlette")
    for lang, title in (("es", "Conecta tus IAs"), ("en", "Connect your AIs")):
        page = connect_page(lang, "https://memory.example.org")
        content = page.body.decode("utf-8")
        assert title in content
        for client in CLIENTS:
            assert 'id="' + client + '"' in content
        assert 'https://memory.example.org/mcp' in content
        assert 'cursor://anysphere.cursor-deeplink/mcp/install' in content
        assert 'action="https://memory.example.org' not in content
        assert page.headers["cache-control"] == "no-store"
        assert page.headers["referrer-policy"] == "no-referrer"
        assert "default-src 'none'" in page.headers["content-security-policy"]


def test_install_plan_calls_only_vendor_cli_with_https_url():
    assert commands("codex", "https://memory.example.org") == [
        ["codex", "mcp", "add", "infinito", "--url", "https://memory.example.org/mcp"],
        ["codex", "mcp", "login", "infinito"],
    ]
    assert commands("claude-code", "https://memory.example.org")[0] == [
        "claude", "mcp", "add", "--transport", "http", "--scope", "user",
        "infinito", "https://memory.example.org/mcp",
    ]
    with pytest.raises(ValueError):
        commands("chatgpt", "https://memory.example.org")


def test_local_installer_requires_explicit_consent_and_never_uses_a_shell():
    calls = []

    def runner(argv, **kwargs):
        calls.append((argv, kwargs))

    with pytest.raises(PermissionError):
        install("codex", "https://memory.example.org", runner=runner)
    assert not calls
    preview = install("codex", "https://memory.example.org", dry_run=True, runner=runner)
    assert len(preview) == 2 and "codex mcp login infinito" in preview[1]
    assert not calls
    install("codex", "https://memory.example.org", approved=True, runner=runner, executable=lambda _: "/bin/codex")
    assert len(calls) == 2
    assert all(call[1] == {"check": True, "shell": False} for call in calls)


def test_missing_client_does_not_run_anything():
    calls = []
    with pytest.raises(FileNotFoundError):
        install("claude-code", "https://memory.example.org", approved=True,
                runner=lambda *a, **k: calls.append((a, k)), executable=lambda _: None)
    assert calls == []
