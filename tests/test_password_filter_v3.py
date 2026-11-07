"""A password is blocked when one is given, however it is written; a sentence about a password is not one."""
import pytest

from src.infinito3.safety import SensitiveInformationFilter
from src.infinito3.types import SafetyLevel

GIVEN = [
    # With "es", "is", ":" or "=", as before.
    "Mi contraseña del banco es hunter2", "My bank password is hunter2", "password: hunter2", "La contraseña=hunter2",
    "La contraseña del router es R0uter-Casa!2026", "Mi contraseña es limonero77", "contraseña: perro",
    "La contraseña es Perro Azul Feliz", "contraseña: caballo batería grapa", "password: correct horse battery staple",
    "Mi contraseña es «limonero77».", "contraseña: la123", "The router password is in the vault: hunter2",
    # With nothing in between, which used to go through.
    "Entra con la contraseña limonero77", "con la contraseña limonero77", "Use the password Hunter2026 to log in",
    "la contraseña 482913 del portal", "la contraseña limonero-77",
]

ABOUT_A_PASSWORD = [
    # What was blocked before: a sentence after the colon.
    "Ojo con la contraseña: no la compartas con nadie.", "Sobre la contraseña: la cambio cada mes.",
    "Contraseña: mínimo 12 caracteres y un símbolo.", "Password: change it every three months.",
    "La contraseña del wifi es la de siempre.", "Mi contraseña es la misma que la del correo.",
    # What already passed, and must keep passing.
    "Uso un gestor de contraseñas para todo.", "Should the limited PDF be password-protected before I upload it?",
    "La contraseña del router está en la bóveda, en la entrada Router de casa.",
    "The router password is in the vault.", "Cambié la contraseña 3 veces en 2024.",
    "La contraseña de 2025 caducó.", "Olvidé la contraseña del portal otra vez.",
]


@pytest.mark.parametrize("text", GIVEN)
def test_a_password_given_is_blocked_however_it_is_written(text):
    decision = SensitiveInformationFilter().inspect(text)
    assert (decision.level, decision.reason) == (SafetyLevel.FORBIDDEN, "password"), text
    assert "[REDACTED]" in decision.redacted_text


@pytest.mark.parametrize("text", ABOUT_A_PASSWORD)
def test_a_sentence_about_a_password_is_not_one(text):
    assert SensitiveInformationFilter().inspect(text).level != SafetyLevel.FORBIDDEN, text
