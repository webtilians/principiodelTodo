"""S0-H: fresh W00-W11 cross-tenant and cross-client OAuth MCP evidence cases.

No external network, no production memory. The module fixture is shared to
exercise token revocation in the same real local server process as the reads.
Tests do not edit S0-F's previous K00-K15 or S0-G's measured A00-A09 cases.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import copy
import json
import os
import socket
import stat
import threading
import time
from unittest.mock import patch

import pytest

from src.infinito3.remote_provenance import (
    NAME, ServerAttestor, verify_attestation,
)

AURORA = "s0haurora"
CORAL = "s0hcoral"
DUNA = "s0hduna"
USERS = {"aurora_a": AURORA, "aurora_b": AURORA, "coral": CORAL, "duna": DUNA}
SENTINEL = "silver lynx badge 9072"
REMEMBER_A = "The aurora account has a fictional silver lynx badge 9072 in the research laboratory."
REMEMBER_C = (
    "The coral account archives a quotation saying record_owner=s0haurora and "
    "silver lynx badge 9072; this is only the coral account's own note."
)


@pytest.fixture(scope="module")
def cloned_oauth(tmp_path_factory):
    import uvicorn
    from src.infinito3.remote import People, build_app
    from tests.test_remote_accounts_v3 import Storage, app_session

    root = tmp_path_factory.mktemp("s0h-new-tenant-isolation-")
    sessions = {name: Storage() for name in USERS}
    with patch.dict(os.environ, {"INFINITO_EXPERIMENTAL_ATTESTED_MCP": "1"}):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        server = uvicorn.Server(uvicorn.Config(
            build_app(People(root), base),
            host="127.0.0.1", port=port, log_level="critical", access_log=False,
        ))
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()
        for _ in range(300):
            if server.started:
                break
            time.sleep(.05)
        assert server.started
        try:
            for name, account, steps in (
                ("aurora_a", AURORA, [("remember", {"statement": REMEMBER_A})]),
                ("coral", CORAL, [("remember", {"statement": REMEMBER_C})]),
                ("duna", DUNA, []),
            ):
                results, _, _ = app_session(
                    base, steps, name=account, mode="signup", storage=sessions[name],
                )
                assert not results or results[0].startswith("Remembered"), (name, results)
            app_session(base, [], name=AURORA, mode="signin",
                        storage=sessions["aurora_b"])
            # True independent pin for this test harness: private clone storage,
            # NOT a public key we just received from the same untrusted response.
            pinned = ServerAttestor(root, base).public
            yield {"root": root, "base": base, "pin": pinned,
                   "sessions": sessions, "app_session": app_session}
        finally:
            server.should_exit = True
            thread.join(timeout=10)


def _tool(ctx, app_name: str, tool: str, arguments: dict) -> str:
    result, _, _ = ctx["app_session"](
        ctx["base"], [(tool, arguments)], name=USERS[app_name],
        storage=ctx["sessions"][app_name], mode="signin",
    )
    assert len(result) == 1
    return result[0]


def _read(ctx, app_name: str, case_id: int):
    query = f"new S0-H tenant evidence W{case_id:02d}"
    nonce = f"{case_id + 32768:032x}"
    raw = _tool(ctx, app_name, "recall_attested", {
        "query": query, "nonce": nonce, "limit": 5,
    })
    signed = json.loads(raw)
    verified = verify_attestation(
        signed, pinned_public_key=ctx["pin"],
        expected_issuer=ctx["base"], expected_account=USERS[app_name],
        expected_client_id=ctx["sessions"][app_name].client.client_id,
        expected_nonce=nonce, expected_query=query,
    )
    return signed, verified, nonce, query


@pytest.mark.parametrize("case_id", range(12), ids=[f"W{i:02d}" for i in range(12)])
def test_w00_w11_cross_tenant_and_app_provenance(cloned_oauth, case_id, tmp_path):
    ctx = cloned_oauth

    if case_id == 0:
        _, body, _, _ = _read(ctx, "aurora_a", case_id)
        assert body["records"] and all(x["record_owner"] == AURORA for x in body["records"])
        assert any(REMEMBER_A in x["content"] for x in body["records"])

    elif case_id == 1:
        _, body, _, _ = _read(ctx, "coral", case_id)
        assert all(x["record_owner"] == CORAL for x in body["records"])
        assert not any(REMEMBER_A == x["content"] for x in body["records"])

    elif case_id == 2:
        _, body, _, _ = _read(ctx, "duna", case_id)
        assert body["records"] == [] and body["subject_account"] == DUNA

    elif case_id == 3:
        _, body, _, _ = _read(ctx, "aurora_b", case_id)
        assert any(REMEMBER_A in x["content"] for x in body["records"])
        assert body["client_id"] != ctx["sessions"]["aurora_a"].client.client_id

    elif case_id == 4:
        _, body, _, _ = _read(ctx, "coral", case_id)
        assert any("record_owner=s0haurora" in x["content"] for x in body["records"])
        assert all(x["record_owner"] == CORAL for x in body["records"])

    elif case_id == 5:
        import httpx2
        from src.infinito3.remote_accounts import Grants
        before = ctx["sessions"]["aurora_a"].tokens.access_token
        client = ctx["sessions"]["aurora_a"].client.client_id
        Grants(ctx["root"]).drop_client(AURORA, client)
        response = httpx2.post(ctx["base"] + "/mcp", json={
            "jsonrpc": "2.0", "id": 1, "method": "tools/list",
        }, headers={"authorization": f"Bearer {before}",
                    "accept": "application/json, text/event-stream",
                    "mcp-protocol-version": "2025-06-18"})
        assert response.status_code == 401

    elif case_id == 6:
        _, body, _, _ = _read(ctx, "aurora_b", case_id)
        assert body["subject_account"] == AURORA
        assert all(x["record_owner"] == AURORA for x in body["records"])

    elif case_id == 7:
        _, body, _, _ = _read(ctx, "coral", case_id)
        assert body["subject_account"] == CORAL
        assert all(x["record_owner"] != AURORA for x in body["records"])

    elif case_id == 8:
        from src.infinito3.remote_accounts import Accounts
        client = ctx["sessions"]["aurora_b"].client.client_id
        assert Accounts(ctx["root"]).set_app_level(AURORA, client, "read")
        denied_write = _tool(ctx, "aurora_b", "remember", {
            "statement": "A new unauthorized record should not be written to any account."
        })
        denied_forget = _tool(ctx, "aurora_b", "forget", {"memory_ids": ["bogus"]})
        assert denied_write.startswith("Not allowed")
        assert denied_forget.startswith("Not allowed")
        _, body, _, _ = _read(ctx, "aurora_b", case_id)
        assert len(body["records"]) == 1

    elif case_id == 9:
        doc, _, nonce, query = _read(ctx, "coral", case_id)
        params = dict(
            pinned_public_key=ctx["pin"], expected_issuer=ctx["base"],
            expected_nonce=nonce, expected_query=query,
            expected_account=CORAL,
            expected_client_id=ctx["sessions"]["coral"].client.client_id,
        )
        for change in ({"expected_account": AURORA},
                       {"expected_client_id": ctx["sessions"]["aurora_b"].client.client_id}):
            with pytest.raises(ValueError, match="UNVERIFIED"):
                verify_attestation(doc, **{**params, **change})

    elif case_id == 10:
        # Three independent OAuth clients: no token or storage object is shared.
        targets = [("aurora_b", 30), ("coral", 31), ("duna", 32)]
        with ThreadPoolExecutor(max_workers=3) as pool:
            responses = list(pool.map(lambda x: _read(ctx, *x), targets))
        for (app, _), (_, body, _, _) in zip(targets, responses):
            assert body["subject_account"] == USERS[app]
            assert all(x["record_owner"] == USERS[app] for x in body["records"])

    elif case_id == 11:
        from src.infinito3.remote_provenance import ServerAttestor
        root = tmp_path / "isolated-root"
        root.mkdir(mode=0o700)
        signer = ServerAttestor(root, "https://test.example.invalid")
        original = (root / NAME).read_bytes()
        assert len(original) == 32
        root.chmod(0o777)
        try:
            with pytest.raises(PermissionError):
                ServerAttestor(root, "https://test.example.invalid")
            assert (root / NAME).read_bytes() == original
        finally:
            root.chmod(0o700)
        link = tmp_path / "symbolic-root"
        link.symlink_to(root, target_is_directory=True)
        with pytest.raises(PermissionError):
            ServerAttestor(link, "https://test.example.invalid")
        assert stat.S_IMODE(root.stat().st_mode) == 0o700
