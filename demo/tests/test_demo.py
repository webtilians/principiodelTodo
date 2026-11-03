"""The demo's API, offline: no key, the stand-in model, hash embeddings."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient  # noqa: E402

from demo.app import create_app  # noqa: E402
from demo.service import Demo, OfflineLLM, Settings  # noqa: E402


class Spy:
    """The stand-in model, keeping every request it was sent."""

    def __init__(self):
        self.inner = OfflineLLM()
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self.inner.generate(request)

    def sent(self) -> str:
        return "\n".join(m.content for r in self.requests for m in r.messages)


def client(**settings):
    spy = Spy()
    demo = Demo(Settings(**settings), llm=spy)
    return TestClient(create_app(demo, sweep=False)), demo, spy


def say(api, session, text, ip="10.0.0.1"):
    return api.post("/api/chat", json={"session": session, "message": text}, headers={"X-Forwarded-For": ip})


def texts(rows):
    return [row["text"] for row in rows]


def test_the_page_and_status_are_served():
    api, _, _ = client()
    assert "INFINITO" in api.get("/").text
    assert api.get("/healthz").json() == {"ok": True}
    status = api.get("/api/status").json()
    assert status["available"] is True and status["messages_per_visitor"] == 30


def test_infinito_deletes_on_request_and_the_plain_memory_keeps_everything():
    api, _, _ = client()
    sid = api.post("/api/session").json()["session"]
    say(api, sid, "Vivo en Valencia.")
    say(api, sid, "Trabajo de enfermera en un hospital.")
    reply = say(api, sid, "Olvida dónde vivo.").json()
    assert reply["infinito"]["forgotten"] == 1
    assert texts(reply["infinito"]["memories"]) == ["Trabajo de enfermera en un hospital."]
    assert texts(reply["plain"]["memories"]) == ["Vivo en Valencia.", "Trabajo de enfermera en un hospital.",
                                                 "Olvida dónde vivo."]
    asked = say(api, sid, "¿Dónde vivo?").json()
    assert "Valencia" not in asked["infinito"]["answer"]
    assert texts(asked["infinito"]["memories"]) == ["Trabajo de enfermera en un hospital."]


def test_a_secret_is_blocked_and_never_reaches_the_model():
    api, _, spy = client()
    sid = api.post("/api/session").json()["session"]
    say(api, sid, "Vivo en Valencia.")
    reply = say(api, sid, "Mi contraseña del banco es hunter2").json()
    assert reply["infinito"]["blocked"] == "password"
    assert texts(reply["infinito"]["memories"]) == ["Vivo en Valencia."]
    assert reply["plain"]["memories"][-1] == {**reply["plain"]["memories"][-1], "text": "Mi contraseña del banco es hunter2",
                                              "secret": True}
    say(api, sid, "¿Cuál es mi contraseña?")
    assert "hunter2" not in spy.sent()


def test_visitors_do_not_share_memories():
    api, _, _ = client()
    first = api.post("/api/session").json()["session"]
    second = api.post("/api/session").json()["session"]
    say(api, first, "Vivo en Valencia.")
    reply = say(api, second, "Trabajo en una panadería.").json()
    assert texts(reply["infinito"]["memories"]) == ["Trabajo en una panadería."]
    assert texts(reply["plain"]["memories"]) == ["Trabajo en una panadería."]


def test_forget_everything_drops_the_session():
    api, demo, _ = client()
    sid = api.post("/api/session").json()["session"]
    say(api, sid, "Vivo en Valencia.")
    assert api.post("/api/forget-everything", json={"session": sid}).json() == {"ok": True}
    assert sid not in demo.sessions
    assert say(api, sid, "¿Dónde vivo?").json() == {"error": "unknown_session"}


@pytest.mark.parametrize("settings, text, code, status", [
    ({}, "x" * 401, "too_long", 400),
    ({}, "   ", "empty", 400),
    ({"daily_budget": 0.0}, "Hola.", "budget", 429),
])
def test_refusals(settings, text, code, status):
    api, _, _ = client(**settings)
    sid = api.post("/api/session").json()["session"]
    response = say(api, sid, text)
    assert (response.status_code, response.json()) == (status, {"error": code})


def test_limits_per_visitor_and_per_connection():
    api, _, _ = client(messages_per_visitor=2, messages_per_ip_hour=3)
    sid = api.post("/api/session").json()["session"]
    assert say(api, sid, "Uno.").json()["messages_left"] == 1
    assert say(api, sid, "Dos.").json()["messages_left"] == 0
    assert say(api, sid, "Tres.").json() == {"error": "visitor_limit"}
    other = api.post("/api/session").json()["session"]
    assert say(api, other, "Cuatro.").status_code == 200
    assert say(api, other, "Cinco.").json() == {"error": "ip_limit"}
    assert say(api, other, "Cinco.", ip="10.0.0.2").status_code == 200


def test_without_a_key_the_public_demo_says_it_is_not_ready():
    demo = Demo(Settings(require_key=True))
    api = TestClient(create_app(demo, sweep=False))
    assert api.get("/api/status").json()["available"] is False
    sid = api.post("/api/session").json()["session"]
    assert say(api, sid, "Hola.").json() == {"error": "unavailable"}


def test_idle_sessions_are_swept():
    api, demo, _ = client()
    sid = api.post("/api/session").json()["session"]
    demo.sessions[sid].used -= 61 * 60
    demo.sweep()
    assert sid not in demo.sessions
