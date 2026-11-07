"""Explicitly approved local client setup using installed vendor CLIs.

Only the public MCP URL is passed as an argument. We do not edit configuration
files, create access tokens, or bypass clients' own OAuth consent screens.
"""
import shutil
import subprocess
from typing import Callable, List, Optional, Tuple

from .remote_connect import server_url


CLIENTS = ("codex", "claude-code")


def commands(client: str, server: str) -> List[List[str]]:
    """A predictable, inspectable plan; subprocess is always invoked without a shell."""
    endpoint = server_url(server) + "/mcp"
    if client == "codex":
        return [
            ["codex", "mcp", "add", "infinito", "--url", endpoint],
            ["codex", "mcp", "login", "infinito"],
        ]
    if client == "claude-code":
        return [["claude", "mcp", "add", "--transport", "http", "--scope", "user",
                 "infinito", endpoint]]
    raise ValueError("Supported local clients: codex, claude-code.")


def install(client: str, server: str, *, approved: bool = False, dry_run: bool = False,
            runner: Callable[..., object] = subprocess.run,
            executable: Callable[[str], Optional[str]] = shutil.which) -> Tuple[str, ...]:
    """Install only after explicit approval. Never execute commands through a shell."""
    plan = commands(client, server)
    summary = tuple(" ".join(command) for command in plan)
    if dry_run:
        return summary
    if not approved:
        raise PermissionError("Confirm installation first. No configuration was changed.")
    cli = plan[0][0]
    if not executable(cli):
        raise FileNotFoundError(f"{cli} is not installed or not on PATH.")
    for command in plan:
        runner(command, check=True, shell=False)
    return summary
