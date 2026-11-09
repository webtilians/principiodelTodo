"""One guide to connecting, with the fewest steps each app allows: a button where it can, the rest ready to copy."""
import base64
import json
import re
from urllib.parse import parse_qs, unquote, urlparse

import pytest

from src.infinito3 import remote_client
from src.infinito3.remote_home import APPS, HOOK, TEXT, connect_links, home_page

ADDRESS = "https://memoria.example/mcp"


def test_claude_opens_its_add_connector_dialog_with_infinito_already_filled_in():
    link = urlparse(connect_links(ADDRESS)["claude"])
    assert (link.scheme, link.netloc, link.path) == ("https", "claude.ai", "/customize/connectors")
    assert parse_qs(link.query) == {"modal": ["add-custom-connector"], "connectorName": ["INFINITO"],
                                    "connectorUrl": [ADDRESS]}


def test_cursor_and_vscode_install_it_with_one_click_and_only_the_address():
    links = connect_links(ADDRESS)
    cursor = urlparse(links["cursor"])
    assert links["cursor"].startswith("cursor://anysphere.cursor-deeplink/mcp/install?")
    query = parse_qs(cursor.query)
    assert query["name"] == ["infinito"] and json.loads(base64.b64decode(query["config"][0])) == {"url": ADDRESS}
    assert links["vscode"].startswith("vscode:mcp/install?")
    assert json.loads(unquote(links["vscode"].split("?", 1)[1])) == {"name": "infinito", "type": "http", "url": ADDRESS}


def test_the_commands_carry_the_address_and_no_secret():
    links = connect_links(ADDRESS)
    assert links["claude_code"] == f"claude mcp add --transport http --scope user infinito {ADDRESS}"
    assert links["codex"].splitlines() == [f"codex mcp add infinito --url {ADDRESS}", "codex mcp login infinito"]
    assert links["gemini"].endswith(ADDRESS)
    assert json.loads(links["other"]) == {"mcpServers": {"infinito": {"type": "http", "url": ADDRESS}}}
    for value in links.values():
        assert "token" not in value.lower() and "password" not in value.lower()


def test_the_hook_brings_the_memory_to_each_message_through_the_server_already_connected():
    (hook,) = HOOK["hooks"]["UserPromptSubmit"][0]["hooks"]
    assert hook == {"type": "mcp_tool", "server": "infinito", "tool": "recall",
                    "input": {"query": "${prompt}", "limit": 15, "max_chars": 9000}}
    assert json.loads(connect_links(ADDRESS)["hook"]) == HOOK


@pytest.mark.parametrize("lang", ["es", "en"])
def test_the_front_page_is_the_one_guide_with_every_app_and_a_copy_button_for_each_text(lang):
    page = home_page(lang, ADDRESS, "", {}).body.decode()
    t = TEXT[lang]
    for name in ("claude", "chatgpt", "grok", "gemini", "cursor", "vscode", "claude_code", "codex", "other"):
        assert f'id="guide-{name}"' in page and t[name] in page and t[f"{name}_steps"][0] in page
    for name, _ in APPS:
        assert f">{name}<" in page
    assert page.count("<script") == 1 and 'data-copy="address"' in page
    targets = re.findall(r'data-copy="([^"]+)"', page)
    assert len(targets) == 8 and all(f'id="{target}"' in page for target in targets)
    assert 'href="https://claude.ai/customize/connectors?modal=add-custom-connector&amp;' in page
    assert 'href="cursor://anysphere.cursor-deeplink/mcp/install?' in page and 'href="vscode:mcp/install?' in page
    assert t["claude_code_ask"] in page and '&quot;type&quot;: &quot;mcp_tool&quot;' in page
    assert "recall" in t["chatgpt_text"] and "<code>" not in t["chatgpt_text"]
    assert "pautas" in t["chatgpt_text"] or "guidelines" in t["chatgpt_text"]


@pytest.mark.parametrize("lang", ["es", "en"])
def test_the_security_research_comes_near_the_end_and_says_it_is_only_research(lang):
    page = home_page(lang, ADDRESS, "", {}).body.decode()
    t = TEXT[lang]
    assert page.index(t["proof_title"]) < page.index(t["research_title"]) < page.index('class="slogan"') < page.index('id="connect"')
    assert t["research_mark"] in page and t["research"] in page
    assert ("nunca el original" if lang == "es" else "never the original") in t["research"]
    assert ("Aún es investigación" if lang == "es" else "still research") in t["research"]


def test_a_hooks_answer_fits_and_every_line_stays():
    lines = [f"[{n}] (2026-11-07) \"{'palabra ' * 120}\"" for n in range(30)]
    fitted = remote_client.fit("\n".join(lines), 9000)
    assert len(fitted) <= 9000 and all(f"[{n}]" in fitted for n in range(30))


# Over the server

httpx2 = pytest.importorskip("httpx2")
from tests.test_remote_accounts_v3 import app_session, server  # noqa: E402,F401


def test_connect_is_a_short_address_for_the_one_guide(server):  # noqa: F811
    base, _, _ = server
    moved = httpx2.get(f"{base}/connect")
    assert moved.status_code == 302 and moved.headers["location"] == "/#connect"
    home = httpx2.get(f"{base}/", headers={"accept-language": "es-ES"})
    assert f'value="{base}/mcp"' in home.text and 'id="guide-claude_code"' in home.text


def test_recall_fits_what_it_says_into_a_hooks_room(server):  # noqa: F811
    base, _, _ = server
    long_one = "Mi proyecto tiene muchas partes: " + ", ".join(f"parte {n}" for n in range(180)) + "."
    steps = [("remember", {"statement": long_one}), ("remember", {"statement": "Vivo en Málaga."}),
             ("recall", {"query": "proyecto"}), ("recall", {"query": "proyecto", "max_chars": 1200})]
    answers, _, _ = app_session(base, steps, mode="signup", name="yago")
    whole, fitted = answers[2], answers[3]
    assert len(long_one) < 2000 and len(whole) > 1700 and len(fitted) <= 1200 and "Vivo en Málaga." in fitted and long_one not in fitted
