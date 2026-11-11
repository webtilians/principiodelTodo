"""Deterministic P0 cognitive diary. Never connects to production or an LLM."""
from .fixture import fictional_day
from .journal import AuditError, Journal

__all__ = ["AuditError", "Journal", "fictional_day"]
