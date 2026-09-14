import re
from typing import List, Pattern, Tuple

from .types import SafetyDecision, SafetyLevel


class SensitiveInformationFilter:
    """Fail-closed filter for secrets that must never enter long-term memory.

    The filter is deliberately conservative. It does not attempt to identify
    every possible form of PII; it blocks the classes of secrets for which a
    false negative would be especially damaging.
    """

    _FORBIDDEN_PATTERNS: List[Tuple[Pattern[str], str]] = [
        (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I), "private_key"),
        (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "api_key"),
        (re.compile(r"\b(?:password|contrase(?:ñ|n)a|passwd)\s*(?:es|:|=)?\s*\S+", re.I), "password"),
        (re.compile(r"\b(?:pin)\s*(?:es|:|=)?\s*\d{3,8}\b", re.I), "pin"),
    ]

    _SENSITIVE_PATTERNS: List[Tuple[Pattern[str], str]] = [
        (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "email"),
        (re.compile(r"\b(?:\+?34[ -]?)?[6789]\d{8}\b"), "phone"),
    ]

    def inspect(self, text: str) -> SafetyDecision:
        for pattern, reason in self._FORBIDDEN_PATTERNS:
            if pattern.search(text):
                return SafetyDecision(
                    level=SafetyLevel.FORBIDDEN,
                    reason=reason,
                    redacted_text=self._redact(text, pattern),
                )

        for pattern, reason in self._SENSITIVE_PATTERNS:
            if pattern.search(text):
                return SafetyDecision(
                    level=SafetyLevel.SENSITIVE,
                    reason=reason,
                    redacted_text=self._redact(text, pattern),
                )

        return SafetyDecision(level=SafetyLevel.SAFE)

    @staticmethod
    def _redact(text: str, pattern: Pattern[str]) -> str:
        return pattern.sub("[REDACTED]", text)
