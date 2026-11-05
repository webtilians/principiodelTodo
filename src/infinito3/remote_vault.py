"""A vault that works from any app: the agent gets a link, the person types on the server's own page.

The vault on a computer (vault.py) opens windows on that computer's screen, which ChatGPT or a phone cannot
do. So on the server vault_add and vault_reveal answer with a link. Whatever is secret is typed and shown on
that page: it never passes through the agent, its model or the chat, and no tool ever returns a value.

This is the first step: the tools and the link, to learn whether apps accept them. Nothing is kept yet, and
the page says so and has nothing to fill in. A link stays valid after it is opened, because chat apps may
fetch a link before the person does; using it up belongs to the step that takes the secret.
"""
import html
from typing import Callable, Optional

from .safety import SensitiveInformationFilter
from .types import SafetyLevel

LINK_SECONDS = 600
MAX_NAME = 80
GRANT = "vault"

TOOL_DESCRIPTIONS = {
    "vault_add": "Keep a password, key or PIN in the user's vault. Give only a name for it, such as 'Router de casa', "
                 "never the value. It answers with a link: give it to the user, who types the value on that page, "
                 "so it never passes through you or the chat.",
    "vault_reveal": "Show the user a secret from their vault, by entry name. It answers with a link: give it to the "
                    "user, who sees the value on that page. You never receive it.",
    "vault_list": "The names of the entries in the user's vault. Values are never shown to you.",
}
# These tools change nothing by themselves and return no secret; only the person on the page can.
TOOL_HINTS = {"readOnlyHint": True, "openWorldHint": False}

NEEDS_ACCOUNT = "The vault needs an INFINITO account with sign-in; this memory uses a private address."
NOTHING_KEPT = "The vault has no entries yet: it is still being built and keeps nothing."


def refused_name(entry: str) -> Optional[str]:
    """Why this cannot be an entry's name: names are read by the agent, so a secret must not arrive as one."""
    name = " ".join((entry or "").split())
    if not name:
        return "Not done: give the entry a name, such as 'Router de casa'."
    if len(name) > MAX_NAME:
        return f"Not done: an entry's name has at most {MAX_NAME} characters."
    if SensitiveInformationFilter().inspect(name).level in (SafetyLevel.FORBIDDEN, SafetyLevel.SENSITIVE):
        return "Not done: that looks like the secret itself. Give only a name for it, and do not repeat the value."
    return None


def answer_with_link(issuer: str, action: str, entry: str, issue: Callable[[dict], str]) -> str:
    refused = refused_name(entry)
    if refused:
        return refused
    name = " ".join(entry.split())
    token = issue({"action": action, "entry": name})
    return (f"Give the user this link and ask them to open it: {issuer}/vault?r={token}\n"
            f"It opens their vault page, for '{name}'. Whatever is secret is typed or shown there, never here. It "
            f"lasts {LINK_SECONDS // 60} minutes. Nothing changes until the user finishes on that page, so do not "
            "say it is done.")


def page_body(t: dict, action: str, entry: str) -> str:
    """The page for a link. There is nothing to fill in yet, and it tells the person not to type a secret."""
    asked = t["vault_add"] if action == "add" else t["vault_reveal"]
    return f'<p>{asked.format(entry=html.escape(entry))}</p><p class="note">{t["vault_soon"]}</p>'
