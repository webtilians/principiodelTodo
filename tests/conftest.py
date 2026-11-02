"""Shared pytest configuration.

Tests import the core as ``src.infinito3``; make the repository root importable
even when pytest is launched without ``python -m``.
"""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
