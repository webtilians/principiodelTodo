"""A memory for work: what each thing is, where it came from and which client it belongs to."""
import re

import pytest

from src.infinito3.mcp_server import AgentMemory


@pytest.fixture
def memory(tmp_path):
    agent_memory = AgentMemory(str(tmp_path / "memory.db"))
    yield agent_memory
    agent_memory.close()


def lines(text):
    return [line for line in text.splitlines() if line.startswith("[")]


def test_a_memory_says_what_it_is_where_it_came_from_and_whose_it_is(memory):
    assert memory.remember("Acme aprueba el presupuesto de 4.000 euros.", "decision", "correo de Ana", "Acme",
                           "2026-10-12").startswith("Remembered")
    [line] = lines(memory.recall("presupuesto"))
    assert "(2026-10-12, decision, Acme, source: correo de Ana)" in line


def test_something_pending_is_kept_though_it_reads_like_a_request(memory):
    assert memory.remember("Reclamar a Marta los textos de Acme antes del viernes.", kind="pending",
                           project="Acme").startswith("Remembered")
    assert "Reclamar a Marta" in memory.recall("textos", kind="pending")


def test_a_question_secrets_and_contact_details_are_refused_for_work_too(memory):
    assert "not memories" in memory.remember("¿Qué falta en Acme?", kind="pending", project="Acme")
    assert "a password" in memory.remember("La contraseña de Acme es hunter2secret", kind="fact", project="Acme")
    assert "an email address" in memory.remember("Ana escribe desde ana@acme.example", kind="fact", project="Acme")
    assert memory.list_memories() == "No memories yet."


def test_a_wrong_kind_or_date_stores_nothing_and_says_why(memory):
    assert "kind must be one of" in memory.remember("Acme firma el viernes.", kind="rumor")
    assert "date must look like" in memory.remember("Acme firma el viernes.", kind="fact", date="el viernes")
    assert memory.list_memories() == "No memories yet."


def test_the_same_sentence_about_two_clients_stays_two_memories(memory):
    memory.remember("Falta enviar el presupuesto.", kind="pending", project="Acme")
    memory.remember("Falta enviar el presupuesto.", kind="pending", project="Globex")
    assert len(lines(memory.list_memories())) == 2
    assert "Acme" in memory.recall("presupuesto", project="Acme")
    assert "Globex" not in memory.recall("presupuesto", project="Acme")


def test_one_clients_fact_never_replaces_anothers_or_the_users(memory):
    memory.remember("Vivo en Málaga.")
    memory.remember("El responsable vive en Cádiz. Vivo en Cádiz, dice en su firma.", kind="fact", project="Acme")
    memory.remember("Vivo en Sevilla, escribe el proveedor.", kind="fact", project="Globex")
    listed = memory.list_memories()
    assert "Vivo en Málaga." in listed and "Cádiz" in listed and "Sevilla" in listed


def test_a_clients_memories_are_found_even_if_they_were_saved_without_its_tag(memory):
    memory.remember("Trabajo con Acme desde 2024.")
    memory.remember("Acme ganó un premio en octubre.", kind="fact", project="acme")
    memory.remember("Globex cambió de logo.", kind="fact", project="Globex")
    found = memory.recall("qué sabemos", project=" ACME ")
    assert "desde 2024" in found and "premio" in found and "Globex" not in found
    assert "desde 2024" in memory.list_memories(project="acme")


def test_another_clients_memory_stays_out_though_it_names_this_client(memory):
    memory.remember("Acme aprueba la campaña.", kind="decision", project="Acme")
    memory.remember("Globex compite con Acme en Navidad.", kind="fact", project="Globex")
    assert "aprueba" in memory.recall("qué sabemos", project="Acme")
    assert "compite" not in memory.recall("qué sabemos", project="Acme")
    assert "compite" not in memory.list_memories(project="Acme")
    assert "compite" in memory.recall("Globex compite con Acme")  # without a client, the search is by topic


def test_a_clients_name_is_a_whole_word_in_memories_without_a_tag(memory):
    memory.remember("Mañana tengo dentista.")
    memory.remember("Hablé con Ana sobre el logo.")
    found = memory.recall("qué sabemos", project="Ana")
    assert "logo" in found and "dentista" not in found


def test_project_scope_rejects_overlapping_names_and_foreign_text(memory):
    memory.remember("ProyectoNube: Entidad lima estado cobre.", kind="fact", project="ProyectoNube")
    memory.remember("TENANT_B_ProyectoNube: Entidad lima estado violeta.",
                    kind="fact", project="TENANT_B_ProyectoNube")
    memory.remember("Menciono ProyectoNube de pasada.", kind="fact", project="OtraCuenta")
    requested = memory.recall("estado entidad lima", project="ProyectoNube")
    assert "cobre" in requested
    assert "violeta" not in requested
    assert "de pasada" not in requested
    requested_list = memory.list_memories(project="proyectonube")
    assert "cobre" in requested_list and "violeta" not in requested_list
    foreign = memory.recall("estado entidad lima", project="TENANT_B_ProyectoNube")
    assert "violeta" in foreign and "cobre" not in foreign


def test_blank_project_does_not_match_projectless_memories(memory):
    memory.remember("Recuerdo personal sin etiqueta.")
    memory.remember("ProyectoNube tiene un objetivo.", kind="fact", project="ProyectoNube")
    assert memory.recall("recuerdo", project="   ") == "No memories about that."
    assert memory.list_memories(project="   ") == "No memories about that."


def test_recall_narrows_to_what_is_pending_with_one_client(memory):
    memory.remember("Acme quiere la web en mayo.", kind="fact", project="Acme")
    memory.remember("Enviar a Acme el calendario.", kind="pending", project="Acme")
    memory.remember("Enviar a Globex la factura.", kind="pending", project="Globex")
    pending = lines(memory.recall("pendiente", project="Acme", kind="pending"))
    assert len(pending) == 1 and "calendario" in pending[0]
    assert memory.recall("nada", project="Initech") == "No memories about that."


def test_an_old_email_saved_today_takes_its_place_among_the_rest(memory):
    memory.remember("Acme pide cambiar el logo.", kind="fact", project="Acme", date="2026-10-20")
    memory.remember("Acme aprobó el logo azul.", kind="decision", project="Acme", date="2026-03-02")
    found = lines(memory.list_memories(project="Acme"))
    assert "logo azul" in found[0] and "cambiar el logo" in found[1]  # oldest first: the later one is current


def test_a_whole_meeting_goes_in_with_one_call(memory):
    answer = memory.remember_many([
        {"statement": "Acme aprueba el presupuesto.", "kind": "decision"},
        {"statement": "Marta entrega los textos el viernes.", "kind": "agreement"},
        {"statement": "Reclamar las fotos al fotógrafo.", "kind": "pending"},
        {"statement": "Quizá hacer también un vídeo.", "kind": "proposal"},
        {"statement": "La clave del gestor es Zx9-hunter2-Qq", "kind": "fact"},
    ], source="reunión con Marta", project="Acme", date="2026-10-26")
    assert answer.startswith("Remembered 4 of 5.")
    assert "Item 5:" in answer and "hunter2" not in answer  # a refused secret is never repeated
    found = lines(memory.list_memories(project="Acme"))
    assert len(found) == 4 and all("2026-10-26" in line and "source: reunión con Marta" in line for line in found)
    assert "proposal" in memory.recall("vídeo", kind="proposal")


def test_an_item_can_say_its_own_source_and_too_many_store_nothing(memory):
    memory.remember_many([{"statement": "Acme cambia de sede.", "kind": "fact", "source": "nota de prensa"}],
                         source="reunión", project="Acme")
    assert "source: nota de prensa" in memory.list_memories()
    assert "at most" in memory.remember_many([{"statement": f"Dato {n} de Globex."} for n in range(101)], project="Globex")
    assert "Globex" not in memory.list_memories()


def test_plain_memories_read_as_before(memory):
    memory.remember("Me llamo Ana.")
    assert re.fullmatch(r'\[[0-9a-f-]{36}\] \(\d{4}-\d{2}-\d{2}\) "Me llamo Ana\."', lines(memory.list_memories())[0])
