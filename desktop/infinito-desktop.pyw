"""Starts INFINITO Desktop from a checkout with pythonw (no console window), and is the built executable's entry."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from infinito_desktop.__main__ import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
