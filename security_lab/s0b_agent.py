"""S0-B bounded adversarial policy agent for real MCP calls against a disposable clone.

This is intentionally NOT an LLM and is not a vulnerability-exploitation framework.
The transport is supplied by tests and may only contact their loopback MCP server.
No tools, networks, targets, secrets, shell access or models are configured by the agent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

ALLOWED = frozenset({
    "remember", "recall", "list_memories", "forget",
    "vault_list", "vault_reveal", "vault_delete",
})


@dataclass(frozen=True)
class Observation:
    tool: str
    # Redacted boolean observations, never whole MCP answers or tokenized URLs.
    matched_canary: bool
    was_denied: bool


class BoundedMCPAgent:
    """Adapt next action to the previous tool feedback, with a hard call budget."""

    def __init__(self, invoke: Callable[[str, dict], str], *, max_calls: int = 3):
        if not 1 <= max_calls <= 8:
            raise ValueError("S0-B call budget must be 1..8")
        self._invoke = invoke
        self.max_calls = max_calls
        self.calls = 0
        self.audit: list[Observation] = []

    def call(self, tool: str, arguments: Mapping[str, object], *, watch: str = "") -> str:
        if tool not in ALLOWED:
            raise PermissionError("Tool outside S0-B allowlist")
        if self.calls >= self.max_calls:
            raise RuntimeError("S0-B agent exceeded call budget")
        if not isinstance(arguments, Mapping):
            raise ValueError("Only typed MCP tool arguments are allowed")
        self.calls += 1
        answer = self._invoke(tool, dict(arguments))
        if not isinstance(answer, str):
            raise TypeError("MCP tool response must be a text result")
        self.audit.append(Observation(
            tool=tool,
            matched_canary=bool(watch and watch in answer),
            was_denied=("Not allowed" in answer or "by invitation" in answer),
        ))
        return answer

    def probe_foreign_memory(self, marker: str, foreign_id: str) -> dict[str, bool]:
        """Attempt disclosure, then deletion, only if prior reads do not disclose."""
        listed = self.call("list_memories", {}, watch=marker)
        if marker in listed:
            return {"disclosed": True, "delete_denied": False, "short_circuit": True}
        recalled = self.call("recall", {"query": marker}, watch=marker)
        if marker in recalled:
            return {"disclosed": True, "delete_denied": False, "short_circuit": True}
        deletion = self.call("forget", {"memory_ids": [foreign_id]})
        return {
            "disclosed": False,
            "delete_denied": "Unknown ids" in deletion or "Not allowed" in deletion,
            "short_circuit": False,
        }

    def probe_vault(self, marker: str, entry: str) -> tuple[str, str]:
        """Try an authorized list and reveal; do not fetch links or enter secrets."""
        listed = self.call("vault_list", {}, watch=marker)
        shown = self.call("vault_reveal", {"entry": entry}, watch=marker)
        return listed, shown


def sanitize_summary(agent: BoundedMCPAgent) -> dict:
    """Report only aggregate booleans, never bearer tokens, links or memory text."""
    return {
        "calls": agent.calls,
        "max_calls": agent.max_calls,
        "tools": [event.tool for event in agent.audit],
        "canary_exposed": any(event.matched_canary for event in agent.audit),
        "denied_count": sum(event.was_denied for event in agent.audit),
    }
