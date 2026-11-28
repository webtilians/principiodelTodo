"""INFINITO Desktop: INFINITO's nervous system on a Windows PC (docs/INFINITO_DESKTOP_DOSSIER_2026-11-27.md).

Phase 1 is a tray icon that says at a glance whether INFINITO works on this PC and warns when something stops working.
Phase 2 is the quick capture: Ctrl+Alt+I from anywhere, and what the person types or dictates goes to their memory as a
note of theirs. The app signs in as a device (src/infinito3/remote_device_api.py), with a scope that only lets it add
notes: it opens no part of the memory, not even reading it. It reads only INFINITO's own files on this PC.

Run from a checkout, it finds the core (src/infinito3) and the cognitive engine (cognition/) next to it; the built
executable carries both.
"""
import sys
from pathlib import Path

VERSION = "0.2.0"
NAME = "INFINITO Desktop"

_ROOT = Path(__file__).resolve().parents[2]
for _place, _marker in ((_ROOT / "src", _ROOT / "src" / "infinito3"), (_ROOT, _ROOT / "cognition")):
    if _marker.is_dir() and str(_place) not in sys.path:
        sys.path.append(str(_place))
