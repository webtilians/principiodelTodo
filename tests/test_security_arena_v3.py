"""The security arena: real attacks on a throwaway clone of the server, recorded for a page to replay.

One recording serves every test here: it starts the real server six times, so it takes a while."""
import json
import re

import pytest

pytest.importorskip("mcp.server.mcpserver")
pytest.importorskip("cryptography")
pytest.importorskip("httpx2")
pytest.importorskip("uvicorn")
pytest.importorskip("pykeepass")

from security_lab.arena import recorder  # noqa: E402


@pytest.fixture(scope="module")
def run():
    return recorder.record()


def the_round(run, name):
    return next(entry for entry in run["rounds"] if entry["id"] == name)


def test_the_server_as_it_is_stops_every_attack(run):
    current = the_round(run, "current")
    assert current["attacks"] >= 15 and current["through"] == []


@pytest.mark.parametrize("fault, gets_through", [("levels-off", {"A10", "A11", "A13", "A14"}),
                                                 ("disconnect-off", {"A15", "A15b"})])
def test_a_planted_fault_is_noticed_and_the_original_code_stops_it_again(run, fault, gets_through):
    entry = the_round(run, fault)
    assert gets_through <= set(entry["through"])
    assert entry["outcome"] == "caught" and entry["replay"] and all(event["held"] for event in entry["replay"])


def test_a_token_that_opens_the_wrong_account_reads_nothing(run):
    entry = the_round(run, "wrong-account")
    assert entry["outcome"] == "contained" and entry["through"] == []
    # What stops it is the sealing: another account's key does not open the file, so the tool fails.
    assert any(event["reply"].startswith("Error executing tool") for event in entry["events"])
    assert "contained" in entry["fault"]


def test_an_app_that_reaches_the_vault_gets_a_link_and_never_the_value(run):
    reveal = next(event for event in the_round(run, "levels-off")["events"] if event["id"] == "A14b")
    assert reveal["held"] and "vault?r=…" in reveal["reply"]
    assert recorder.VAULT_VALUE not in json.dumps(run)


def test_every_step_says_what_reached_the_attacker_and_what_it_decided(run):
    for entry in run["rounds"]:
        moved = set()
        for event in entry["events"]:
            assert event["decided"]["en"] and event["decided"]["es"]
            assert event["rule"] in run["rules"] and event["door"] in run["doors"]
            assert (event["arrived"] is None) == (event["who"] not in moved)  # nothing arrives before a first move
            moved.add(event["who"])


def test_a_move_made_only_after_a_success_is_absent_when_nothing_got_through(run):
    assert not any(event["id"].endswith("b") for event in the_round(run, "current")["events"])
    assert {"A15", "A15b"} <= {event["id"] for event in the_round(run, "disconnect-off")["events"]}


def test_the_recording_holds_no_token_and_no_address_of_the_clone(run):
    text = json.dumps(run, ensure_ascii=False)
    assert not re.search(r"[A-Za-z0-9_-]{40,}", text)  # a token, a code or a key would be this long
    assert "127.0.0.1" not in text
