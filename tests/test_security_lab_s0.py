"""S0 contracts: real sealed memory storage, entirely synthetic and temporary."""
import socket

import pytest

pytest.importorskip("cryptography")

from security_lab.s0 import DANGEROUS_ENV, GATES, no_python_connections, observe, preflight, run, verify


def test_s0_gates_pass_against_current_real_memory_classes(monkeypatch):
    for name in DANGEROUS_ENV:
        monkeypatch.delenv(name, raising=False)
    checks = run()
    assert {c.name for c in checks} == set(GATES)
    assert all(c.passed for c in checks), [c.name for c in checks if not c.passed]


def test_environment_with_deployment_path_aborts_before_creating_test_data(monkeypatch, tmp_path):
    for name in DANGEROUS_ENV:
        monkeypatch.delenv(name, raising=False)
    live = tmp_path / "must-not-write"
    monkeypatch.setenv("INFINITO_REMOTE_DIR", str(live))
    with pytest.raises(RuntimeError, match="Refusing S0"):
        preflight()
    with pytest.raises(RuntimeError, match="Refusing S0"):
        run()
    assert not live.exists()


def test_outbound_socket_attempt_is_rejected_without_contacting_network():
    with no_python_connections(), socket.socket() as sock:
        with pytest.raises(RuntimeError, match="forbidden"):
            sock.connect(("127.0.0.1", 1))


def test_verifier_detects_tampering_without_trusting_test_agent(monkeypatch, tmp_path):
    for name in DANGEROUS_ENV:
        monkeypatch.delenv(name, raising=False)
    with no_python_connections():
        observations = observe(tmp_path)
    observations["ana_after"] = ""
    checks = {c.name: c.passed for c in verify(observations)}
    assert checks["cross_owner_forget"] is False
    assert checks["tenant_isolation"] is True


def test_s0_never_uses_environment_memory_file(monkeypatch, tmp_path):
    for name in DANGEROUS_ENV:
        monkeypatch.delenv(name, raising=False)
    marker = tmp_path / "outside-lab.db"
    marker.write_text("must-not-touch", encoding="utf-8")
    assert all(c.passed for c in run())
    assert marker.read_text(encoding="utf-8") == "must-not-touch"
