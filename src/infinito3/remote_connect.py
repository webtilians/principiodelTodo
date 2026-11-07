"""An installation page for existing INFINITO OAuth/MCP; no secrets and no client impersonation.

The user authorizes every client in the client's own UI. This module only produces
installation links, instructions and token-free configuration for that client.
"""
import base64
import html
import json
from urllib.parse import urlencode, urlsplit


CLIENTS = ("chatgpt", "claude", "cursor", "codex", "claude-code", "other")


def server_url(server: str) -> str:
    """Normalize a public INFINITO origin, never a user-specific bearer URL."""
    if not isinstance(server, str):
        raise ValueError("INFINITO server URL must be a string.")
    parsed = urlsplit(server.strip())
    if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("Expected a public INFINITO server URL without credentials or query.")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1")
    ):
        raise ValueError("INFINITO requires HTTPS outside local development.")
    if parsed.path.rstrip("/") not in ("", "/mcp"):
        raise ValueError("Use the public server origin, not a private memory URL.")
    # Detect malformed ports before emitting any link or command.
    _ = parsed.port
    return parsed._replace(path="", query="", fragment="").geturl().rstrip("/")


def connection_options(server: str) -> dict:
    """Only public URLs/configuration: never a password, access token, or private address."""
    origin = server_url(server)
    endpoint = origin + "/mcp"
    payload = base64.b64encode(json.dumps({"url": endpoint}, separators=(",", ":")).encode()).decode()
    return {
        "endpoint": endpoint,
        "cursor": "cursor://anysphere.cursor-deeplink/mcp/install?"
                  + urlencode({"name": "infinito", "config": payload}),
        "chatgpt": "https://chatgpt.com/plugins",
        "claude": "https://claude.ai/customize/connectors",
        "codex": 'codex mcp add infinito --url "' + endpoint + '"',
        "codex_login": "codex mcp login infinito",
        "claude_code": 'claude mcp add --transport http --scope user infinito "' + endpoint + '"',
        "local_codex": 'infinito-remote connect codex "' + origin + '"',
        "local_claude": 'infinito-remote connect claude-code "' + origin + '"',
        "generic": json.dumps({"mcpServers": {"infinito": {"url": endpoint}}}, indent=2),
    }


_TEXT = {
    "es": {
        "title": "Conecta tus IAs", "intro": "Una memoria para todos tus asistentes. Elige una aplicación y autoriza INFINITO en ella.",
        "choose": "1. Elige tu asistente", "setup": "2. Conéctalo", "address": "Dirección MCP compartida",
        "test": "3. Comprueba la conexión", "test_body": "En un asistente, guarda un recuerdo de prueba. En otro, pregunta por él. Borra después el recuerdo desde tu portal. Las escrituras y el uso automático dependen de los permisos de cada aplicación.",
        "safety": "No pegues contraseñas, claves ni URL privadas en configuraciones. INFINITO usa el mismo OAuth que ya existe; cada cliente debe pedirte permiso.",
        "chatgpt": "Abre Plugins en ChatGPT, añade un servidor MCP personalizado cuando tu plan y espacio de trabajo lo permitan, pega la dirección y autoriza. No existe instalación silenciosa para cuentas ajenas.",
        "claude": "Abre Personalizar → Conectores → Añadir conector personalizado; introduce INFINITO y la dirección MCP. Acepta la conexión e inicia sesión. Las opciones dependen de la cuenta.",
        "cursor": "Añadir a Cursor", "cursor_text": "Abre Cursor y confirma la instalación; Cursor realizará el inicio de sesión OAuth.",
        "codex": "En un terminal puedes usar el instalador de INFINITO, que te pide confirmación, o estos comandos oficiales:",
        "claude_code": "Usa el instalador de INFINITO (con confirmación) o el comando de Claude Code. Después autoriza en el propio cliente:",
        "other": "Utiliza esta configuración en un cliente compatible con MCP HTTP y OAuth. Debe poder descubrir el servidor y completar la autorización; no añadas tokens a este JSON.",
        "portal": "Ver mi memoria", "home": "Inicio", "open": "Abrir aplicación", "install": "Instalador local",
        "automatic": "Instalación guiada: la autorización sigue siendo del usuario.",
        "no_auto": "ChatGPT y Claude no permiten que una web externa instale un MCP y acepte permisos sin intervención del usuario.",
    },
    "en": {
        "title": "Connect your AIs", "intro": "One memory for every assistant. Pick an app, then authorize INFINITO there.",
        "choose": "1. Choose your assistant", "setup": "2. Connect it", "address": "Shared MCP endpoint",
        "test": "3. Verify the connection", "test_body": "Save a test memory in one assistant and ask about it from another. Delete the memory from your portal afterward. Writes and automatic recall depend on each host's permissions.",
        "safety": "Never put passwords, tokens or private addresses into client configs. INFINITO reuses its existing OAuth; each client must ask for your approval.",
        "chatgpt": "Open Plugins in ChatGPT, add a custom MCP server where your plan and workspace permit it, paste the endpoint and authorize. Third-party sites cannot silently install it.",
        "claude": "Open Customize → Connectors → Add custom connector, enter INFINITO and the MCP endpoint. Approve and sign in. Availability depends on your account.",
        "cursor": "Add to Cursor", "cursor_text": "Open Cursor and confirm the installation; Cursor handles OAuth sign-in.",
        "codex": "In your terminal use INFINITO's confirming installer, or the official commands below:",
        "claude_code": "Use INFINITO's confirming installer or Claude Code's command. Then authorize in the client:",
        "other": "Use this configuration in an HTTP MCP client with OAuth support. The client must discover the server and authorize; never add tokens to this JSON.",
        "portal": "My memory", "home": "Home", "open": "Open app", "install": "Local installer",
        "automatic": "Guided setup: the user still authorizes the connection.",
        "no_auto": "ChatGPT and Claude do not let third-party sites install MCP servers and silently grant permissions.",
    },
}


_STYLE = """
:root{color-scheme:dark;font-family:system-ui,-apple-system,Segoe UI,sans-serif;background:#080e1b;color:#edf5ff}
*{box-sizing:border-box}body{margin:0}a{color:#8deaff}
main{max-width:1030px;padding:24px 20px 76px;margin:auto}
header{display:flex;align-items:center;justify-content:space-between;gap:14px;margin-bottom:48px}
header a{color:#c5d7ed;text-decoration:none}header .brand{font-weight:800;letter-spacing:.1em;color:white}
h1{font-size:clamp(2rem,5vw,3.5rem);margin:10px 0}h2{margin:40px 0 18px;font-size:1.4rem}
h3{margin:4px 0 12px;font-size:1.15rem}p{line-height:1.6;color:#bac9db}
.lead{font-size:1.15rem;max-width:680px}.tag{color:#7ce5e9;font-size:.8rem;letter-spacing:.12em;text-transform:uppercase}
nav{display:grid;grid-template-columns:repeat(auto-fit,minmax(138px,1fr));gap:12px}
nav a,.panel{background:#111d30;border:1px solid #334a63;border-radius:14px;padding:19px}
nav a{text-decoration:none;color:#edf5ff;font-weight:650;text-align:center}
nav a:hover,nav a:focus-visible{border-color:#8deaff}
.panel{margin:14px 0;scroll-margin-top:18px}.panel p{margin:10px 0}
.action{display:inline-block;padding:10px 18px;margin:8px 12px 8px 0;border-radius:9px;background:#8deaff;color:#091323;font-weight:750;text-decoration:none}
.secondary{background:#253d57;color:#f2f8ff}.muted{font-size:.9rem;color:#aabfd8}
pre,.command{display:block;white-space:pre-wrap;overflow-wrap:anywhere;padding:13px;border:1px solid #374f65;border-radius:9px;background:#080e1b;font-size:.88rem;color:#bafcf7;user-select:all}
input{background:#081222;color:#dbf8fa;border:1px solid #374f65;border-radius:9px;padding:11px;width:100%;font:inherit}
.notice{border-left:3px solid #8deaff;padding:11px 16px;background:#152238;border-radius:6px;margin:23px 0}
code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace}
@media(prefers-reduced-motion:no-preference){nav a{transition:border-color .12s}}
"""


def connect_page(lang: str, server: str):
    """Render the public, zero-token connection wizard. OAuth is handled by the client."""
    from starlette.responses import HTMLResponse

    t = _TEXT["es" if lang == "es" else "en"]
    info = connection_options(server)
    e = lambda value: html.escape(value, quote=True)
    nav = "".join(
        '<a href="#' + key + '">' + e(label) + "</a>"
        for key, label in (
            ("chatgpt", "ChatGPT"), ("claude", "Claude"), ("cursor", "Cursor"),
            ("codex", "Codex"), ("claude-code", "Claude Code"), ("other", "Otros MCP" if lang == "es" else "Other MCP")
        )
    )

    def section(key, title, body):
        return '<section class="panel" id="' + key + '"><h3>' + e(title) + '</h3>' + body + "</section>"

    endpoint = '<input id="mcp" aria-label="' + e(t["address"]) + '" readonly value="' + e(info["endpoint"]) + '">'
    chatgpt = section("chatgpt", "ChatGPT",
                      '<p>' + e(t["chatgpt"]) + '</p><a class="action secondary" href="' +
                      e(info["chatgpt"]) + '" rel="noopener noreferrer">' + e(t["open"]) + '</a>')
    claude = section("claude", "Claude",
                     '<p>' + e(t["claude"]) + '</p><a class="action secondary" href="' +
                     e(info["claude"]) + '" rel="noopener noreferrer">' + e(t["open"]) + '</a>')
    cursor = section("cursor", "Cursor",
                     '<p>' + e(t["cursor_text"]) + '</p><a class="action" href="' +
                     e(info["cursor"]) + '">' + e(t["cursor"]) + '</a>')
    codex = section("codex", "Codex",
                    '<p>' + e(t["codex"]) + '</p><span class="muted">' + e(t["install"]) + '</span>' +
                    '<pre>' + e(info["local_codex"]) + '</pre><pre>' + e(info["codex"]) +
                    '\n' + e(info["codex_login"]) + '</pre>')
    code = section("claude-code", "Claude Code",
                   '<p>' + e(t["claude_code"]) + '</p><span class="muted">' + e(t["install"]) + '</span>' +
                   '<pre>' + e(info["local_claude"]) + '</pre><pre>' + e(info["claude_code"]) + '</pre>')
    generic = section("other", "Otros MCP" if lang == "es" else "Other MCP",
                      '<p>' + e(t["other"]) + '</p><pre>' + e(info["generic"]) + '</pre>')
    body = ('<!doctype html><html lang="' + e(lang) + '"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<meta name="robots" content="index,follow"><title>INFINITO · ' + e(t["title"]) +
            '</title><style>' + _STYLE + '</style></head><body><main><header>'
            '<a class="brand" href="/">∞ INFINITO</a><div><a href="/">' + e(t["home"]) +
            '</a> · <a href="/memory">' + e(t["portal"]) + '</a></div></header>'
            '<span class="tag">MCP · OAuth</span><h1>' + e(t["title"]) +
            '</h1><p class="lead">' + e(t["intro"]) + '</p><h2>' + e(t["choose"]) +
            '</h2><nav aria-label="' + e(t["choose"]) + '">' + nav +
            '</nav><h2>' + e(t["setup"]) + '</h2><label for="mcp">' + e(t["address"]) +
            '</label><div>' + endpoint + '</div><p class="muted">' + e(t["automatic"]) + '</p>' +
            cursor + chatgpt + claude + codex + code + generic +
            '<h2>' + e(t["test"]) + '</h2><p>' + e(t["test_body"]) +
            '</p><div class="notice">' + e(t["safety"]) + '</div><p class="muted">' +
            e(t["no_auto"]) + '</p></main></body></html>')
    return HTMLResponse(body, headers={
        "Cache-Control": "no-store",
        "Referrer-Policy": "no-referrer",
        "X-Content-Type-Options": "nosniff",
        "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'",
    })
