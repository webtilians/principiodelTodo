"""Where INFINITO keeps its things on this PC: what the checks read, and where this app keeps its own."""
import os
from pathlib import Path

SERVER = "https://infinitomemory.com"


def infinito_home() -> Path:
    return Path(os.environ.get("INFINITO_HOME") or Path.home() / ".infinito")


def home() -> Path:
    """This app's own folder: its sealed session, its settings and what it last saw."""
    return Path(os.environ.get("INFINITO_DESKTOP_HOME") or infinito_home() / "desktop")


def cognition_home() -> Path:
    return Path(os.environ.get("INFINITO_COGNITION_HOME") or infinito_home() / "cognition")


def backups() -> Path:
    return infinito_home() / "copias-servidor"


def hooks_session() -> Path:
    """The Claude Code hooks' session (src/infinito3/remote_client.py): only whether it is there is ever looked at."""
    return Path(os.environ.get("INFINITO_REMOTE_SESSION") or infinito_home() / "remote-session.json")


def claude_settings() -> Path:
    return Path.home() / ".claude" / "settings.json"
