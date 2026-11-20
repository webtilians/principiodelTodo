"""INFINITO Teams, the core (remote_teams.py): who reads which space, and that the rest cannot even be opened.

The organization is the dossier's Acme: its owner (ceo), an org admin, Ana (development, project Alfa), Bea (design,
Alfa) and Carlos (marketing, Beta); and Globex, another company that also has a "Proyecto Alfa". All of it made up."""
import json
from types import SimpleNamespace

import pytest

from src.infinito3 import remote_backup
from src.infinito3.remote_accounts import Accounts
from src.infinito3.remote_teams import ADMIN, NOT_FOUND, PENDING, ACTIVE, Refused, Teams

pytest.importorskip("cryptography")

PASSWORD = "una contraseña bastante larga"
DECISION = "En Alfa usamos el pago en una sola pantalla."


def offer_from(teams, org, inviter, key):
    """An invitation as Grants.find gives it back, with the general space's key if the inviter had it."""
    kept = {}

    def issue(space_key, data):
        kept.update(key=space_key, data=data, account=inviter)
        return "token"

    teams.invite(org, inviter, key, issue)
    return kept


def build(folder):
    accounts = Accounts(folder)
    keys = {name: accounts.create(name, PASSWORD)[0] for name in ("ceo", "admin", "ana", "bea", "carlos", "eva")}
    accounts.allow_teams("ceo")
    accounts.allow_teams("eva")
    teams = Teams(accounts)
    acme = teams.create_org("ceo", keys["ceo"], "Acme")
    for name in ("admin", "ana", "bea", "carlos"):
        teams.accept(offer_from(teams, acme, "ceo", keys["ceo"]), name, keys[name])
    teams.set_role(acme, "ceo", "admin", ADMIN)
    alfa = teams.create_space(acme, "ceo", keys["ceo"], "Proyecto Alfa")
    beta = teams.create_space(acme, "ceo", keys["ceo"], "Proyecto Beta")
    assert teams.add(acme, alfa, "ceo", keys["ceo"], "ana") == ACTIVE
    assert teams.add(acme, alfa, "ceo", keys["ceo"], "bea") == ACTIVE
    assert teams.add(acme, beta, "ceo", keys["ceo"], "carlos") == ACTIVE
    globex = teams.create_org("eva", keys["eva"], "Globex")
    globex_alfa = teams.create_space(globex, "eva", keys["eva"], "Proyecto Alfa")
    return SimpleNamespace(folder=folder, accounts=accounts, keys=keys, teams=teams, acme=acme, alfa=alfa, beta=beta,
                           globex=globex, globex_alfa=globex_alfa)


@pytest.fixture
def w(tmp_path):
    return build(tmp_path)


def texts(found):
    return [memory["text"] for space in found for memory in space["memories"]]


def recall(w, who, spaces=None, org=None, query="pago"):
    return w.teams.recall(org or w.acme, who, w.keys[who], query, spaces)


def remember(w, who, space, statement=DECISION, **extra):
    return w.teams.remember(w.acme, space, who, w.keys[who], statement, kind=extra.pop("kind", "fact"), **extra)


# Who reads what (A01-A03)

def test_a_project_memory_reaches_its_members_and_the_owner_with_who_saved_it(w):
    assert remember(w, "ana", w.alfa, kind="decision").startswith("Remembered as")
    for reader in ("bea", "ceo"):
        [memory] = [m for space in recall(w, reader, [w.alfa]) for m in space["memories"]]
        assert (memory["text"], memory["by"], memory["kind"]) == (DECISION, "ana", "decision")


def test_someone_outside_a_project_cannot_read_it_list_it_or_tell_it_from_one_that_does_not_exist(w):
    remember(w, "ana", w.alfa)
    with pytest.raises(Refused) as not_theirs:
        recall(w, "carlos", [w.alfa])
    with pytest.raises(Refused) as made_up:
        recall(w, "carlos", ["0123456789abcdef"])
    assert str(not_theirs.value) == str(made_up.value) == NOT_FOUND
    assert DECISION not in texts(recall(w, "carlos"))  # every space he may read: the general one and Beta
    assert {s["name"] for s in w.teams.spaces(w.acme, "carlos")} == {"General", "Proyecto Beta"}


def test_a_refused_read_never_opens_a_sealed_space(w, monkeypatch):
    remember(w, "ana", w.alfa)
    opened = []
    real = w.teams.sealed.read
    monkeypatch.setattr(w.teams.sealed, "read", lambda name, key: opened.append(name) or real(name, key))
    with pytest.raises(Refused):
        recall(w, "carlos", [w.beta, w.alfa])  # one of his and one that is not: refused as a whole
    with pytest.raises(Refused):
        w.teams.memories(w.acme, w.alfa, "carlos", w.keys["carlos"])
    assert opened == []


def test_being_a_manager_somewhere_gives_nothing_in_another_project(w):
    teams, k = w.teams, w.keys
    teams.add(w.acme, w.beta, "ceo", k["ceo"], "bea", role="manager")
    remember(w, "ana", w.alfa)
    assert DECISION not in texts(recall(w, "carlos"))
    with pytest.raises(Refused):
        teams.add(w.acme, w.alfa, "bea", k["bea"], "carlos")  # Bea manages Beta, not Alfa


# Organizations apart (A04, A14)

def test_another_companys_ids_open_nothing_even_with_the_same_project_name(w):
    remember(w, "ana", w.alfa)
    w.teams.remember(w.globex, w.globex_alfa, "eva", w.keys["eva"], "Globex usa tres pantallas de pago.", kind="fact")
    for org, spaces in ((w.globex, None), (w.globex, [w.globex_alfa]), (w.acme, [w.globex_alfa])):
        with pytest.raises(Refused, match=NOT_FOUND):
            w.teams.recall(org, "ana", w.keys["ana"], "pago", spaces)
    assert texts(w.teams.recall(w.globex, "eva", w.keys["eva"], "pago")) == ["Globex usa tres pantallas de pago."]
    assert [o["name"] for o in w.teams.orgs_of("ana")] == ["Acme"]


def test_a_made_up_organization_id_is_not_found(w):
    for made_up in ("", "../acme", "0" * 16, None):
        with pytest.raises(Refused, match=NOT_FOUND):
            w.teams.recall(made_up, "ana", w.keys["ana"], "pago")


# The owner reads the organization, never a person (A05)

def test_the_owner_reads_every_space_without_a_seat_and_nothing_touches_a_persons_own_memory(w):
    remember(w, "carlos", w.beta, "Beta lanza la campaña en marzo.")
    assert "Beta lanza la campaña en marzo." in texts(recall(w, "ceo", query="campaña"))
    written = {p.relative_to(w.folder).parts[0] for p in w.folder.rglob("*") if p.is_file()}
    assert written == {"accounts.json", "orgs"}  # no person's folder was even created


# Nobody gives themselves anything (A06, A07, ADR D3)

def test_members_cannot_raise_themselves_or_reach_further(w):
    teams, k = w.teams, w.keys
    with pytest.raises(Refused):
        teams.set_role(w.acme, "ana", "ana", ADMIN)
    with pytest.raises(Refused):
        teams.add(w.acme, w.beta, "ana", k["ana"], "ana")
    with pytest.raises(Refused):
        teams.create_space(w.acme, "ana", k["ana"], "Mi proyecto")
    with pytest.raises(Refused):
        teams.invite(w.acme, "ana", k["ana"], lambda key, data: "token")
    with pytest.raises(Refused):
        teams.create_org("ana", k["ana"], "Mi empresa")  # the server's owner did not allow her
    assert teams.org(w.acme, "ana")["members"] == {} and teams.org(w.acme, "ana")["audit"] == []


def test_an_org_admin_runs_spaces_without_reading_them_and_cannot_add_themselves(w):
    teams, k = w.teams, w.keys
    remember(w, "ana", w.alfa)
    with pytest.raises(Refused, match=NOT_FOUND):
        recall(w, "admin", [w.alfa])
    with pytest.raises(Refused):
        teams.add(w.acme, w.alfa, "admin", k["admin"], "admin")
    gamma = teams.create_space(w.acme, "admin", k["admin"], "Proyecto Gamma")
    policy = json.loads((w.folder / "orgs" / w.acme / "policy.json").read_text(encoding="utf-8"))
    assert set(policy["spaces"][gamma]["keys"]) == {"ceo"}  # the admin made it, only the owner can open it
    assert teams.add(w.acme, w.alfa, "admin", k["admin"], "carlos") == PENDING
    assert DECISION not in texts(recall(w, "carlos"))
    teams.confirm(w.acme, w.alfa, "ceo", k["ceo"], "carlos")
    assert DECISION in texts(recall(w, "carlos", [w.alfa]))


def test_someone_invited_by_an_admin_waits_for_the_general_space_until_it_is_confirmed(w):
    teams, k = w.teams, w.keys
    remember(w, "ana", teams.spaces(w.acme, "ana")[0]["id"], "La oficina abre a las nueve.")
    teams.accept(offer_from(teams, w.acme, "admin", k["admin"]), "eva", k["eva"])
    assert texts(recall(w, "eva", query="oficina")) == []
    general = teams.spaces(w.acme, "eva")[0]
    assert (general["name"], general["state"]) == ("General", PENDING)
    teams.confirm(w.acme, general["id"], "ceo", k["ceo"], "eva")
    assert texts(recall(w, "eva", query="oficina")) == ["La oficina abre a las nueve."]


# Leaving, and a new key (A08)

def test_someone_removed_from_a_project_is_refused_at_once_and_the_project_gets_a_new_key(w):
    teams, k = w.teams, w.keys
    remember(w, "ana", w.alfa)
    old_folder = w.folder / "orgs" / w.acme / "spaces" / w.alfa / "v1"
    teams.remove(w.acme, w.alfa, "ceo", "bea")
    with pytest.raises(Refused, match=NOT_FOUND):
        recall(w, "bea", [w.alfa])
    assert DECISION in texts(recall(w, "ana", [w.alfa]))  # Ana's read gives the project its new key
    policy = json.loads((w.folder / "orgs" / w.acme / "policy.json").read_text(encoding="utf-8"))
    space = policy["spaces"][w.alfa]
    assert space["version"] == 2 and set(space["keys"]) == {"ceo", "ana"} and not space["rotate"]
    assert (old_folder.parent / "v2" / "memory.sealed").exists() and old_folder.exists()  # kept one turn for copies
    teams.remove(w.acme, w.alfa, "ana", "ana")  # Ana leaves too
    recall(w, "ceo", [w.alfa])
    assert not old_folder.exists()


def test_leaving_the_organization_takes_every_space_with_it(w):
    teams, k = w.teams, w.keys
    teams.remove_member(w.acme, "admin", "carlos")
    assert teams.orgs_of("carlos") == []
    with pytest.raises(Refused, match=NOT_FOUND):
        recall(w, "carlos")
    with pytest.raises(Refused):
        teams.remove_member(w.acme, "admin", "ceo")
    with pytest.raises(Refused):
        teams.remove_member(w.acme, "ana", "bea")


# Writing (A11, ADR D7)

def test_the_same_idempotency_key_stores_once(w):
    first = remember(w, "ana", w.alfa, idempotency_key="k-1")
    again = remember(w, "ana", w.alfa, idempotency_key="k-1")
    assert first.startswith("Remembered as") and again.startswith("Already remembered as")
    assert texts(recall(w, "ana", [w.alfa])) == [DECISION]


def test_a_decision_sent_by_an_app_waits_as_a_proposal(w):
    answer = remember(w, "ana", w.alfa, kind="decision", app="Claude")
    assert "proposal" in answer
    [memory] = w.teams.memories(w.acme, w.alfa, "bea", w.keys["bea"])
    assert (memory["kind"], memory["by"]) == ("proposal", "ana · Claude")


def test_only_the_author_a_manager_or_the_owner_takes_a_memory_back(w):
    remember(w, "ana", w.alfa)
    [memory] = w.teams.memories(w.acme, w.alfa, "bea", w.keys["bea"])
    with pytest.raises(Refused):
        w.teams.forget(w.acme, w.alfa, "bea", w.keys["bea"], memory["id"])
    assert w.teams.forget(w.acme, w.alfa, "ana", w.keys["ana"], memory["id"])
    assert w.teams.memories(w.acme, w.alfa, "ceo", w.keys["ceo"]) == []


def test_a_reader_only_member_cannot_add(w):
    w.teams.add(w.acme, w.beta, "ceo", w.keys["ceo"], "bea", write=False)
    with pytest.raises(Refused, match="not add"):
        remember(w, "bea", w.beta)


# The disk and the copies (A12)

def test_no_words_of_a_memory_reach_the_disk_in_the_clear(w):
    remember(w, "ana", w.alfa, idempotency_key="k-1")
    on_disk = b"".join(p.read_bytes() for p in w.folder.rglob("*") if p.is_file())
    assert DECISION.encode() not in on_disk and "una sola pantalla".encode() not in on_disk


def test_a_copy_restored_elsewhere_keeps_who_reads_what(w, tmp_path):
    remember(w, "ana", w.alfa)
    w.teams.remove(w.acme, w.alfa, "ceo", "bea")
    recall(w, "ana", [w.alfa])  # a new key, with the old version still on disk
    copies = tmp_path / "copies"
    remote_backup.backup(w.folder, copies)
    elsewhere = tmp_path / "elsewhere"
    remote_backup.restore(next(copies.iterdir()), elsewhere)
    teams = Teams(Accounts(elsewhere))
    assert texts(teams.recall(w.acme, "ana", w.keys["ana"], "pago", [w.alfa])) == [DECISION]
    with pytest.raises(Refused, match=NOT_FOUND):
        teams.recall(w.acme, "bea", w.keys["bea"], "pago", [w.alfa])


# Order by meaning, one cache per space (ADR D9)

class Ranker:
    def __init__(self):
        self.caches = []

    def rank(self, query, texts, cache=None):
        self.caches.append(id(cache))
        return list(range(len(texts)))


def test_each_space_ranks_with_a_cache_of_its_own(tmp_path):
    w = build(tmp_path)
    ranker = w.teams.ranker = Ranker()
    for n in range(12):
        remember(w, "ana", w.alfa, f"Nota {n} de Alfa sobre el pago.")
        remember(w, "carlos", w.beta, f"Nota {n} de Beta sobre el pago.")
    w.teams.recall(w.acme, "ceo", w.keys["ceo"], "pago", [w.alfa, w.beta], limit=3)
    w.teams.recall(w.acme, "ana", w.keys["ana"], "pago", [w.alfa], limit=3)
    assert len(set(ranker.caches)) == 2 and ranker.caches[0] == ranker.caches[2]


# What the review of the four pull requests found (20/11/2026)

def test_a_deleted_name_taken_again_by_someone_else_is_nobody_in_the_organization(w):
    teams = w.teams
    teams.set_role(w.acme, "ceo", "bea", ADMIN)
    w.accounts.remove("bea")
    impostor = w.accounts.create("bea", "otra contraseña bastante larga")[0]
    w.accounts.key_pair("bea", impostor)
    assert teams.orgs_of("bea") == []
    with pytest.raises(Refused, match=NOT_FOUND):
        teams.recall(w.acme, "bea", impostor, "pago")
    with pytest.raises(Refused):
        teams.invite(w.acme, "bea", impostor, lambda key, data: "token")
    # Invited on her own, she starts from nothing: no seat in Alfa from the account that had the name before.
    teams.accept(offer_from(teams, w.acme, "ceo", w.keys["ceo"]), "bea", impostor)
    assert {s["name"] for s in teams.spaces(w.acme, "bea")} == {"General"}
    assert teams.org(w.acme, "ceo")["members"]["bea"] == "member"


def test_a_long_idempotency_key_still_stores_once(w):
    long_key = "tarea-" + "x" * 150
    remember(w, "ana", w.alfa, idempotency_key=long_key)
    assert remember(w, "ana", w.alfa, idempotency_key=long_key).startswith("Already remembered as")
    assert texts(recall(w, "ana", [w.alfa])) == [DECISION]


def test_refused_reads_never_push_the_record_of_changes_out(w):
    for _ in range(600):
        with pytest.raises(Refused):
            recall(w, "carlos", [w.alfa])
    actions = [item["action"] for item in w.teams.org(w.acme, "ceo")["audit"]]
    assert "create_org" in actions and "add_member:member" in actions


def test_changing_the_role_of_someone_already_in_a_space_keeps_their_access(w):
    teams, k = w.teams, w.keys
    remember(w, "ana", w.alfa)
    with pytest.raises(Refused, match="already"):
        teams.add(w.acme, w.alfa, "admin", k["admin"], "bea", role="manager")
    assert teams.add(w.acme, w.alfa, "ceo", k["ceo"], "bea", role="manager") == ACTIVE
    assert DECISION in texts(recall(w, "bea", [w.alfa]))
    assert teams.add(w.acme, w.alfa, "bea", k["bea"], "carlos") == ACTIVE  # now she manages Alfa


def test_a_project_whose_manager_cannot_be_added_is_not_created(w):
    before = {s["id"] for s in w.teams.spaces(w.acme, "ceo")}
    with pytest.raises(Refused, match="invite them first"):
        w.teams.create_space(w.acme, "ceo", w.keys["ceo"], "Proyecto Gamma", manager="beaa")
    assert {s["id"] for s in w.teams.spaces(w.acme, "ceo")} == before


def test_the_owner_naming_themselves_manager_of_a_new_project_just_gets_the_project(w):
    """What happened on the real server: the owner wrote their own name as manager and got "not found"."""
    alfa2 = w.teams.create_space(w.acme, "ceo", w.keys["ceo"], "Proyecto Delta", manager="ceo")
    assert next(s for s in w.teams.spaces(w.acme, "ceo") if s["id"] == alfa2)["manage"]


def test_whoever_runs_a_space_is_told_why_someone_cannot_be_added(w):
    from src.infinito3.remote_teams import NOBODY, OWNER_READS, SELF

    teams, k = w.teams, w.keys
    for target, said in (("admin", SELF), ("ceo", OWNER_READS), ("eva", NOBODY), ("nadie", NOBODY)):
        with pytest.raises(Refused) as refused:
            teams.add(w.acme, w.alfa, "admin", k["admin"], target)
        assert str(refused.value) == said, target
    with pytest.raises(Refused, match=NOT_FOUND):
        teams.add(w.acme, w.alfa, "carlos", k["carlos"], "eva")  # outside Alfa: still nothing told
