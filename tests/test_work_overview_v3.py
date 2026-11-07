"""What a working day asks of the memory: who to write to, who saved what, and how each client is doing."""
import re
from datetime import date

import pytest

from src.infinito3.mcp_server import BY_THE_USER, INSTRUCTIONS, KINDS, TOOL_DESCRIPTIONS, AgentMemory
from src.infinito3.remote_accounts import _app_name
from src.infinito3.remote_vault import _looks_secret
from tests.test_remote_accounts_v3 import app_session, server  # noqa: F401  # server is a fixture
from tests.test_remote_memory_page_v3 import browser, enter, ids_in, memory_of

httpx2 = pytest.importorskip("httpx2")

SPANISH = {"accept-language": "es-ES"}
TODAY = date(2026, 11, 6)


@pytest.fixture
def memory(tmp_path):
    agent_memory = AgentMemory(str(tmp_path / "memory.db"))
    yield agent_memory
    agent_memory.close()


# Emails and phone numbers

@pytest.mark.parametrize("statement", ["Mi correo es ana@acme.example", "My email is ana@example.com",
                                       "El teléfono de la agencia es 612345678"])
def test_emails_and_phone_numbers_are_kept_as_they_were_said(memory, statement):
    assert memory.remember(statement).startswith("Remembered")
    recalled = memory.recall("contacto")
    assert statement in recalled and "[REDACTED]" not in recalled


def test_a_clients_contacts_are_kept_with_the_client(memory):
    assert memory.remember("Ana escribe desde ana@acme.example", kind="fact", project="Acme",
                           source="reunión").startswith("Remembered")
    assert "ana@acme.example" in memory.recall("Ana", project="Acme")


def test_secrets_are_still_refused_even_next_to_an_email(memory):
    assert "a password" in memory.remember("Mi correo es ana@acme.example y la contraseña es limonero77")
    assert "a card number" in memory.remember("La tarjeta 4111 1111 1111 1111 de ana@acme.example")
    assert "an identity document number" in memory.remember("El DNI de Ana es 12345678Z")
    assert memory.list_memories() == "No memories yet."


def test_a_question_with_an_email_is_still_not_a_memory(memory):
    assert "not memories" in memory.remember("¿Cuál es el correo de ana@acme.example?")
    assert memory.list_memories() == "No memories yet."


def test_a_vault_note_may_say_whose_account_it_is_but_not_hold_a_secret():
    assert not _looks_secret("La cuenta de ana@acme.example")
    assert _looks_secret("La contraseña es limonero77")


# Two more kinds

@pytest.mark.parametrize("kind", ["assumption", "to_verify"])
def test_an_assumption_or_something_to_verify_is_never_read_as_a_fact(memory, kind):
    assert kind in KINDS
    assert memory.remember("El premio se falla en marzo.", kind=kind, project="Acme").startswith("Remembered")
    assert f", {kind}, Acme" in memory.recall("premio", project="Acme", kind=kind)
    assert memory.recall("premio", project="Acme", kind="fact") == "No memories about that."


def test_the_instructions_name_the_new_kinds_and_the_overview_and_stay_short():
    assert "to_verify" in INSTRUCTIONS and "overview" in INSTRUCTIONS and len(INSTRUCTIONS) <= 2048
    assert "assumption" in TOOL_DESCRIPTIONS["remember"]
    assert "Contact details" not in TOOL_DESCRIPTIONS["remember"]


# Who saved it

def test_each_way_of_saving_says_which_app_saved_it(memory):
    memory.remember("Vivo en Málaga.", author="ChatGPT")
    memory.remember("Acme aprueba el plan.", kind="decision", project="Acme", author="Claude")
    memory.remember_many([{"statement": "Globex firma en marzo.", "kind": "agreement"}], project="Globex", author="Grok")
    listed = memory.list_memories()
    assert 'by: ChatGPT) "Vivo en Málaga."' in listed
    assert 'by: Claude) "Acme aprueba el plan."' in listed
    assert 'by: Grok) "Globex firma en marzo."' in listed


def test_what_an_app_restates_when_forgetting_says_it_was_that_app(memory):
    memory.remember("Vivo en Málaga y tengo una perra.", author="ChatGPT")
    (found,) = ids_in(memory.list_memories())
    memory.forget([found], keep=["Tengo una perra."], author="Claude")
    assert 'by: Claude) "Tengo una perra."' in memory.list_memories()


def test_with_no_app_nothing_is_said(memory):
    memory.remember("Vivo en Málaga.")
    assert "by:" not in memory.list_memories()


@pytest.mark.parametrize("info, name", [
    ('{"client_name": "ChatGPT"}', "ChatGPT"), ('{"client_name": "  Grok  "}', "Grok"),
    ('{"client_name": "Una\\nInstrucción"}', "UnaInstrucción"), ('{"client_name": ""}', None), ("{}", None),
    (None, None), ("no es json", None), ("[1, 2]", None)])
def test_an_apps_name_stays_on_one_line_or_is_left_out(info, name):
    assert _app_name(info) == name


def test_an_apps_long_name_is_cut():
    assert len(_app_name('{"client_name": "' + "x" * 500 + '"}')) == 60


def test_the_app_saving_is_the_servers_word_not_the_models(server):  # noqa: F811
    base, _, _ = server
    steps = [("remember", {"statement": "Vivo en Málaga.", "author": "Otra app"}),
             ("remember", {"statement": "Acme aprueba el plan.", "kind": "decision", "project": "Acme"}),
             ("list_memories", {})]
    answers, _, app = app_session(base, steps, mode="signup", name="irene")
    assert 'by: Prueba) "Vivo en Málaga."' in answers[-1] and 'by: Prueba) "Acme aprueba el plan."' in answers[-1]
    assert "Otra app" not in answers[-1]
    for tool in app.tools:
        assert "author" not in tool.input_schema["properties"], tool.name


def test_the_page_says_who_saved_it_and_a_correction_says_it_was_you(server):  # noqa: F811
    base, _, _ = server
    listed = memory_of(base, "elena", ["Mi dentista se llama Marta."])
    (dentist,) = ids_in(listed)
    with browser() as client:
        page = enter(client, base, "elena")
        assert "guardado por Prueba" in page.text
        form = client.get(f"{base}/memory", params={"edit": dentist})
        csrf = re.search(r'name="csrf" value="([^"]+)"', form.text).group(1)
        fixed = client.post(f"{base}/memory", data={"action": "correct", "csrf": csrf, "id": dentist,
                                                    "text": "Mi dentista se llama Marta Ruiz."})
        assert "guardado por ti" in fixed.text and "guardado por Prueba" not in fixed.text
    (answer,), _, _ = app_session(base, [("list_memories", {})], name="elena")
    assert f'by: {BY_THE_USER}) "Mi dentista se llama Marta Ruiz."' in answer


def test_a_correction_can_add_an_email_now(server):  # noqa: F811
    base, _, _ = server
    listed = memory_of(base, "sara", ["Mi dentista se llama Marta."])
    (dentist,) = ids_in(listed)
    with browser() as client:
        form = enter(client, base, "sara")
        csrf = re.search(r'name="csrf" value="([^"]+)"', form.text).group(1)
        fixed = client.post(f"{base}/memory", data={"action": "correct", "csrf": csrf, "id": dentist,
                                                    "text": "Mi dentista es marta@clinica.example"})
        assert "marta@clinica.example" in fixed.text and "No se ha guardado" not in fixed.text


# The overview

def fill(memory):
    memory.remember("Acme aprueba el presupuesto.", kind="decision", project="Acme", date="2026-11-05")
    memory.remember("Reclamar a Marta los textos.", kind="pending", project="Acme", date="2026-11-04")
    memory.remember("La fecha del premio.", kind="to_verify", project="ACME", date="2026-11-03")
    memory.remember("Globex firma en marzo.", kind="agreement", project="Globex", date="2026-10-20")
    memory.remember("Vivo en Málaga.")


def test_each_client_with_its_last_activity_most_recent_first(memory):
    fill(memory)
    text = memory.overview(today=TODAY)
    assert "as of 2026-11-06" in text and "data, not instructions" in text
    assert text.index("Acme") < text.index("Globex")
    assert "last activity 2026-11-05 (1 day ago); 3 memories (1 decision, 1 pending, 1 to_verify)." in text
    assert "last activity 2026-10-20 (17 days ago)" in text


def test_a_client_with_no_recent_activity_is_flagged(memory):
    fill(memory)
    text = memory.overview(today=TODAY)
    assert "NO ACTIVITY FOR 17 DAYS" in text and "NO ACTIVITY FOR 1 DAY;" not in text
    assert "NO ACTIVITY" not in memory.overview(stale_days=30, today=TODAY)
    assert "NO ACTIVITY FOR 1 DAY;" in memory.overview(stale_days=1, today=TODAY)
    assert memory.overview(stale_days=0, today=date(2026, 11, 5)).count("NO ACTIVITY") == 1  # active today is not quiet


def test_what_is_still_open_is_shown_with_its_id(memory):
    fill(memory)
    text = memory.overview(today=TODAY)
    assert "Reclamar a Marta los textos." in text and "La fecha del premio." in text
    assert "Acme aprueba el presupuesto." not in text  # a decision is not open
    assert ids_in(text)  # so that an app can act on it


def test_only_the_latest_open_items_are_listed_and_the_rest_are_counted(memory):
    for day in range(1, 9):
        memory.remember(f"Tarea número {day} pendiente.", kind="pending", project="Acme", date=f"2026-11-0{day}")
    text = memory.overview(today=date(2026, 11, 9))
    assert "Tarea número 8" in text and "Tarea número 4" in text and "Tarea número 3" not in text
    assert "and 3 older open items" in text


def test_the_same_client_written_differently_is_one_client(memory):
    fill(memory)
    assert memory.overview(today=TODAY).count("last activity") == 2


def test_memories_with_no_client_are_only_counted(memory):
    fill(memory)
    text = memory.overview(today=TODAY)
    assert "1 memory with no client or project" in text and "Vivo en Málaga" not in text


def test_with_no_clients_it_says_so(memory):
    assert memory.overview() == "No memories have a client or project yet."
    memory.remember("Vivo en Málaga.")
    assert memory.overview() == "No memories have a client or project yet. 1 memory with no client or project; recall finds them."


def test_an_old_email_saved_today_does_not_make_a_quiet_client_look_busy(memory):
    memory.remember("Globex firma en marzo.", kind="agreement", project="Globex", date="2026-01-10")
    assert "NO ACTIVITY FOR" in memory.overview(today=TODAY)


def test_a_date_ahead_is_not_a_negative_count(memory):
    memory.remember("Acme entrega en diciembre.", kind="agreement", project="Acme", date="2026-12-01")
    assert "(today)" in memory.overview(today=TODAY)


def test_an_app_gets_the_overview_from_the_server(server):  # noqa: F811
    base, _, _ = server
    steps = [("remember", {"statement": "Reclamar a Marta los textos.", "kind": "pending", "project": "Acme"}),
             ("overview", {})]
    (saved, text), _, _ = app_session(base, steps, mode="signup", name="julia")
    assert saved.startswith("Remembered") and "Acme: last activity" in text and "Reclamar a Marta" in text


def test_the_front_page_tells_what_changed_in_both_languages(server):  # noqa: F811
    base, _, _ = server
    spanish = httpx2.get(f"{base}/", headers=SPANISH).text
    assert "Guarda correos y teléfonos" in spanish and "¿Cómo va todo?" in spanish and "qué IA lo guardó" in spanish
    assert "Tus fotos también se recuerdan" in spanish and "Por invitación" in spanish
    english = httpx2.get(f"{base}/", headers={"accept-language": "en"}).text
    assert "keeps emails and phone numbers" in english and "which AI saved it" in english
    assert "Your photos are remembered too" in english and "By invitation" in english


# The briefing, from the server

def test_a_briefing_over_the_server_says_which_app_saved_what(server):  # noqa: F811
    base, _, _ = server
    steps = [("remember", {"statement": "La entrega del logo es el 15 de noviembre.", "kind": "agreement",
                           "project": "Acme"}),
             ("remember", {"statement": "La entrega del logo pasa al 20 de noviembre.", "kind": "fact",
                           "project": "Acme"}),
             ("briefing", {"moment": "morning"})]
    answers, _, _ = app_session(base, steps, mode="signup", name="nuria")
    assert "It may disagree with an earlier memory" in answers[1]
    assert "Saved by Prueba 2." in answers[2] and "New memories that may disagree with earlier ones: 1." in answers[2]
