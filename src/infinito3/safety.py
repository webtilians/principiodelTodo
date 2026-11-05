import re
from typing import Callable, List, Pattern, Tuple

from .types import SafetyDecision, SafetyLevel


def _plain(text: str) -> str:
    return re.sub(r"[\s.-]", "", text).upper()


def _iban_ok(found: str, text: str) -> bool:
    """IBAN with a valid check sum, or anything IBAN-shaped after the word IBAN."""
    s = _plain(found)
    if not 15 <= len(s) <= 34:
        return False
    if re.search(r"\biban\b", text, re.I):
        return True
    return int("".join(str(int(c, 36)) for c in s[4:] + s[:4])) % 97 == 1


def _card_ok(found: str, text: str) -> bool:
    """Card number passing the Luhn check, or any long number next to the word card."""
    digits = [int(c) for c in _plain(found)]
    if not 13 <= len(digits) <= 19:
        return False
    if re.search(r"\b(?:tarjeta|card|visa|mastercard|amex)\b", text, re.I):
        return True
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d > 4 else d * 2) for i, d in enumerate(reversed(digits)))
    return total % 10 == 0


_GUARDED = re.compile(r"\b(?:wi-?fi|router|banco|bank|correo|e-?mail|cuenta|account|alarma|alarm|puerta|door|"
                      r"caja fuerte|safe|m[oó]vil|phone|ordenador|computer|port[aá]til|laptop|acceso|login)\b", re.I)


# "The router password is in the vault" says where a secret is kept, which a memory may hold; "is" gives no value there.
_NOT_WHERE_IT_IS_KEPT = r"(?!\s+(?:kept\s+|stored\s+|saved\s+)?in\s+(?:the\s+|my\s+|a\s+)?(?:vault|keepass|password\s+manager)\b)"


def _given_secret(found: str, text: str) -> bool:
    """A key or code actually given: it looks like one, or it guards something ("la clave del wifi es ...")."""
    if re.search(r"c[oó]digo postal|postal code|zip code", found, re.I):
        return False
    value = re.split(r"\b(?:es|is|era|was)\b|[:=]", found, maxsplit=1)[-1].strip().strip("\"'«»“”.,;")
    looks_like_one = len(value) >= 4 and bool(re.search(r"\d|[^\w\s]", value))
    return looks_like_one or bool(_GUARDED.search(found))


class SensitiveInformationFilter:
    """Fail-closed filter for secrets that must never enter long-term memory.

    The filter is deliberately conservative. It does not attempt to identify
    every possible form of PII; it blocks the classes of secrets for which a
    false negative would be especially damaging.
    """

    _FORBIDDEN_PATTERNS: List[Tuple[Pattern[str], str]] = [
        (re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.I), "private_key"),
        (re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"), "api_key"),
        # A password given ("mi contraseña del banco es X", "password: X"), not one mentioned
        # ("a password-protected PDF", "un gestor de contraseñas").
        (re.compile(r"\b(?:password|passwd|contrase(?:ñ|n)a)\b(?:\s+[\w.-]+){0,4}?\s*(?:\b(?:es|is|was|era)\b"
                    + _NOT_WHERE_IT_IS_KEPT + r"|[:=])\s*\S+", re.I), "password"),
        (re.compile(r"\b(?:pin)\s*(?:es|:|=)?\s*\d{3,8}\b", re.I), "pin"),
        # Spanish DNI (12345678Z) and NIE (X1234567L), whether or not the control letter is
        # right: a typo in the letter does not make it less of an identity number.
        # After a space the letter must be a capital: "el pedido 12345678 y llega" is not one.
        (re.compile(r"\b\d{8}(?:-?[A-Za-z]| [A-Z])\b"), "national_id"),
        (re.compile(r"\b[XYZxyz][ -]?\d{7}(?:-?[A-Za-z]| [A-Z])\b"), "national_id"),
        (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "national_id"),  # US Social Security number
    ]

    # Shapes that ordinary text can also take, so a check sum or a nearby word confirms them.
    _CHECKED_FORBIDDEN: List[Tuple[Pattern[str], str, Callable[[str, str], bool]]] = [
        (re.compile(r"\b[A-Za-z]{2}\d{2}(?:[ ]?[A-Za-z0-9]){11,30}\b"), "bank_account", _iban_ok),
        (re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), "card_number", _card_ok),
        # "Clave" and "code" also mean a key to a problem ("la clave es la constancia"): a given
        # value that looks like a secret, or a word for what it guards, confirms it.
        # Within one sentence: "the QR code instead. Is it best..." gives nothing.
        (re.compile(r"\b(?:clave|c[oó]digo|key|code|passcode|passphrase)\b(?:[ \t]+[\w-]+){0,4}?[ \t]*"
                    r"(?:\b(?:es|is|era|was)\b" + _NOT_WHERE_IT_IS_KEPT + r"|[:=])[ \t]*\S+", re.I),
         "password", _given_secret),
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

        for pattern, reason, confirmed in self._CHECKED_FORBIDDEN:
            if any(confirmed(match.group(0), text) for match in pattern.finditer(text)):
                return SafetyDecision(
                    level=SafetyLevel.FORBIDDEN,
                    reason=reason,
                    redacted_text=pattern.sub(
                        lambda m: "[REDACTED]" if confirmed(m.group(0), text) else m.group(0), text),
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
