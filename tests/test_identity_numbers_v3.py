"""Identity documents, bank accounts and cards are secrets, and requests for information are not memories."""
import pytest

from src.infinito3.engine import CognitiveEngine
from src.infinito3.memory import is_question
from src.infinito3.safety import SensitiveInformationFilter
from src.infinito3.types import SafetyLevel

# Synthetic numbers only: valid check letters and sums, belonging to nobody.
SECRETS = [
    ("hola soy Marta, mi DNI es 12345678Z", "national_id"),
    ("mi dni es 12345678a", "national_id"),  # wrong control letter: still an identity number
    ("Mi NIE: X-1234567-L", "national_id"),
    ("My SSN is 123-45-6789", "national_id"),
    ("Mi cuenta es ES91 2100 0418 4502 0005 1332", "bank_account"),
    ("IBAN ES12 3456 7890 1234 5678 9012", "bank_account"),
    ("pago con la 4111 1111 1111 1111", "card_number"),
    ("mi tarjeta es 1234 5678 9012 3456", "card_number"),
]

ORDINARY = [
    "Pagué 1.150 € el 12/03/2024 por el alquiler.",
    "Mi pedido es el 12345678 y llega el martes.",
    "Vivo en el código postal 46001.",
    "The invoice total was 1234567890123 cents, a typo in the system.",
    "Mi bici es una Orbea de 2019.",
    "Should the limited PDF be password-protected before I upload it?",
    "Uso un gestor de contraseñas para todo.",
    "I keep the recovery key in a password manager as a secure note.",
]


@pytest.mark.parametrize("text", ["Mi contraseña del banco es hunter2", "My bank password is hunter2",
                                  "password: hunter2", "La contraseña=hunter2"])
def test_a_given_password_is_still_blocked(text):
    decision = SensitiveInformationFilter().inspect(text)
    assert (decision.level, decision.reason) == (SafetyLevel.FORBIDDEN, "password")


@pytest.mark.parametrize("text, reason", SECRETS)
def test_identity_and_payment_numbers_are_blocked(text, reason):
    decision = SensitiveInformationFilter().inspect(text)
    assert (decision.level, decision.reason) == (SafetyLevel.FORBIDDEN, reason)
    assert "[REDACTED]" in decision.redacted_text


@pytest.mark.parametrize("text", ORDINARY)
def test_ordinary_numbers_are_not_secrets(text):
    assert SensitiveInformationFilter().inspect(text).level == SafetyLevel.SAFE


def test_a_blocked_identity_number_is_never_stored():
    engine = CognitiveEngine()
    decision = engine.process("hola soy Marta, mi DNI es 12345678Z")
    assert decision.safety.level == SafetyLevel.FORBIDDEN
    assert engine.memory_store.search("DNI Marta", top_k=5) == []


@pytest.mark.parametrize("text", [
    "Haz una lista de todo lo que sabes de mí.",
    "dime donde vivo",
    "cuéntame qué recuerdas",
    "List everything you know about me.",
    "tell me my dentist's name",
])
def test_requests_for_information_are_not_stored(text):
    assert is_question(text)
    engine = CognitiveEngine()
    engine.process(text)
    assert engine.memory_store.search(text, top_k=5) == []


def test_a_request_that_also_states_something_is_kept():
    assert not is_question("Dime algo bonito, que hoy cumplo 40 años.")
