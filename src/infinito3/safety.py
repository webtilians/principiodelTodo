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


_PASSWORD_WORD = r"\b(?:password|passwd|contrase(?:ñ|n)a)\b"
# What a sentence about a password starts with ("contraseña: no la compartas", "password: change it every month"),
# where a password itself would not: an article, a negation, a pronoun or a common verb.
_SPEECH = re.compile(
    r"(?:la|el|los|las|lo|un|una|unos|unas|no|ni|nunca|siempre|ya|se|que|y|o|en|de|del|al|con|por|para|sin|mi|mis|tu|"
    r"tus|su|sus|nuestra|nuestro|esta|este|esa|ese|hay|es|está|tengo|tiene|tenemos|hemos|he|ha|cambia|cambio|cambié|"
    r"cámbiala|pide|pídela|pregunta|mínimo|máximo|nueva|antigua|caducada|olvidada|the|a|an|it|its|is|not|never|"
    r"always|my|your|our|their|to|in|on|for|with|without|we|i|you|ask|change|changed|use|same|minimum|maximum|new|"
    r"old|expired|forgotten)$", re.I)


# A password given after "es", "is", ":" or "=", up to the end of its clause. "The router password is in the vault"
# says where it is kept; "...is in the vault: hunter2" still gives one, after the colon.
_PASSWORD_GIVEN = re.compile(_PASSWORD_WORD + r"(?:\s+[\w.-]+){0,4}?\s*(?:\b(?:es|is|was|era)\b" + _NOT_WHERE_IT_IS_KEPT
                             + r"|[:=])\s*(?P<value>[^\s.;\n][^.;\n]*)", re.I)


def _password_given(found: str, text: str) -> bool:
    """A password actually given, not a sentence about one. One word, or something with a digit or a symbol, is a
    password; several words that start like ordinary speech are not. Fail-closed otherwise: "contraseña: caballo
    batería grapa" is a passphrase. The value is where the rule found it, not after the first "is" in the words."""
    given = _PASSWORD_GIVEN.match(found)
    value = given.group("value").strip() if given else found
    words = [word.strip("\"'«»“”,") for word in value.split()]
    words = [word for word in words if word]
    if len(words) <= 1:
        return True
    first = words[0]
    if re.search(r"\d|[^\w\s]", first):
        return True
    return not _SPEECH.match(first)


def _password_right_after(found: str, text: str) -> bool:
    """"con la contraseña limonero77": the password word followed at once by something shaped like one, with no "es"
    or ":" in between. Letters with digits, or six digits or more; a year or a count ("3 veces") is not one."""
    token = re.split(r"\s+", found.strip(), maxsplit=1)[-1].strip("\"'«»“”.,;:")
    if len(token) < 6:
        return False
    if token.isdigit():
        return not re.fullmatch(r"(?:19|20)\d{2}", token)
    return bool(re.search(r"\d", token) and re.search(r"[A-Za-zÀ-ÿ]", token)) or bool(
        re.search(r"\d", token) and re.search(r"[^\w]", token))


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
        # A password given ("mi contraseña del banco es X", "password: X"), not one mentioned ("a password-protected
        # PDF", "un gestor de contraseñas") nor a sentence about one ("contraseña: no la compartas"). The match runs to
        # the end of the clause, so the check sees how the value starts.
        (_PASSWORD_GIVEN, "password", _password_given),
        # ...and one given with nothing in between: "entra con la contraseña limonero77".
        (re.compile(_PASSWORD_WORD + r"\s+[\"'«“]?\S+", re.I), "password", _password_right_after),
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
