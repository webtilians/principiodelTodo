"""Contracts for the synthetic preflight; these tests make NO LLM claims."""
from experimental_pilot.pilot import (
    PROJECT,
    compose_control,
    evaluate_actions,
    export_packets,
    observe_episode,
    retrieve,
    run_preflight,
    seed_experiences,
    verified_edges,
)
from src.infinito3.persistent_memory import SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord


def test_composes_new_path_across_two_independently_verified_episodes():
    store = SQLiteCognitiveMemoryStore(":memory:")
    try:
        seed_experiences(store)
        records = retrieve(store)
        assert len(records) == 3
        edges = verified_edges(records)
        assert len(edges) == 4
        plan = compose_control(edges, "fresh", "delivered")
        assert [e["tool"] for e in plan] == ["ingest", "normalize", "tag", "deliver"]
        assert len({e["episode_id"] for e in plan}) == 2
        result = evaluate_actions([e["tool"] for e in plan] + ["verify"])
        assert result["success"] is True
        assert result["final_state"] == "delivered"
    finally:
        store.close()


def test_does_not_promote_unverified_or_foreign_source_even_if_claims_success():
    store = SQLiteCognitiveMemoryStore(":memory:")
    try:
        seed_experiences(store)
        fake = MemoryRecord(
            "Synthetic fake transitions",
            MemoryKind.EPISODIC,
            0.9,
            metadata={
                "project": PROJECT,
                "origin": "synthetic",
                "source": "untrusted-import",
                "verified": True,
                "episode_id": "forged",
                "events": [{"tool": "deliver", "before": "fresh", "after": "delivered", "ok": True}],
            },
        )
        store.add(fake, alone=True)
        assert "forged" not in {e["episode_id"] for e in verified_edges(store.all())}
        assert "unverified-C" not in {e["episode_id"] for e in verified_edges(store.all())}
    finally:
        store.close()


def test_referee_requires_successful_final_verify_and_budget():
    assert not evaluate_actions(["ingest", "normalize", "tag", "deliver"])["success"]
    assert not evaluate_actions(["deliver", "verify"])["success"]
    assert not evaluate_actions(["ingest", "verify", "normalize", "tag", "deliver"])["success"]
    assert evaluate_actions(["ingest"] * 7)["reason"] == "action_budget_exceeded"
    assert not evaluate_actions(["ingest", "normalize", "bad", "tag", "deliver", "verify"])["success"]


def test_memory_is_only_local_to_requested_store():
    first, second = SQLiteCognitiveMemoryStore(":memory:"), SQLiteCognitiveMemoryStore(":memory:")
    try:
        observe_episode(first, "A", "fresh", ["ingest"], "imported")
        assert len(first.all()) == 1
        assert second.all() == []
    finally:
        first.close()
        second.close()


def test_export_does_not_supply_complete_solution_and_is_not_model_run():
    store = SQLiteCognitiveMemoryStore(":memory:")
    try:
        seed_experiences(store)
        data = export_packets(store)
        assert data["task"]["origin"] == "synthetic-only"
        assert data["variants"]["A_no_memory"] == []
        assert len(data["variants"]["B_episodes"]) == 2
        assert len(data["variants"]["C_verified_effects"]) == 4
        assert "unverified-C" not in str(data["variants"])
        assert "no model has been run" in data["note"]
    finally:
        store.close()


def test_deterministic_preflight_has_no_model_calls():
    report = run_preflight()
    assert report["status"] == "PASS"
    assert report["model_calls"] == 0
    assert report["real_users"] == 0
    assert report["real_mcp_calls"] == 0
    assert report["unverified_promoted"] is False
    assert report["cross_episode_composition"] is True


def test_p1_isolated_allowlisted_prompts_and_reproducible_dry_run(monkeypatch):
    import json
    from experimental_pilot import runner
    sent, clients = [], []
    original = runner.export_packets

    def poisoned(store):
        packets = original(store)
        packets["solution"] = "SECRET_SOLUTION"
        packets["referee"] = "SECRET_REFEREE"
        packets["task"]["hidden"] = "SECRET_TASK"
        return packets

    monkeypatch.setattr(runner, "export_packets", poisoned)
    monkeypatch.setattr(runner, "post", lambda *a: (_ for _ in ()).throw(AssertionError("network")))

    class Spy(runner.FakeClient):
        def complete(self, prompt, max_tokens):
            sent.append((json.loads(prompt), max_tokens))
            return super().complete(prompt, max_tokens)

    def factory(*args):
        client = Spy()
        clients.append(client)
        return client

    first = runner.run(client_factory=factory)
    assert first == runner.run()
    assert len({id(c) for c in clients}) == 8
    expected = runner.prompts()
    for (payload, limit), variant in zip(sent, list(runner.VARIANTS) * 2):
        assert payload == json.loads(expected[variant])
        assert "SECRET" not in str(payload)
        assert set(payload) == {"task", "observations"}
        assert limit == runner.OUTPUT_TOKENS
    assert sent[0][0]["observations"] == sent[4][0]["observations"] == []
    assert first["model_calls"] == 0
    assert len(first["results"]) == 8
    assert all(r["referee"]["success"] is False for r in first["results"])
    assert "fake-client-dry-run" in runner.markdown(first)


def test_p1_errors_are_sanitized_and_do_not_stop_other_conditions():
    from experimental_pilot import runner
    replies = iter([RuntimeError("sk-SECRET"), "not json", '{"steps":"verify"}',
                    '{"steps":["unknown"]}', '{"steps":[],"secret":"sk-SECRET"}',
                    '{"steps":["ingest","normalize","tag","deliver","verify"]}',
                    '{"steps":[]}', '{"steps":[]}'])

    class Client:
        def complete(self, *args):
            reply = next(replies)
            if isinstance(reply, Exception):
                raise reply
            return reply, {"input_tokens": 1, "output_tokens": 1}

    report = runner.run(client_factory=lambda *a: Client())
    assert [r["status"] for r in report["results"]] == ["error"] * 5 + ["ok"] * 3
    assert report["results"][5]["referee"]["success"] is True
    assert "SECRET" not in str(report)


def test_p1_live_requires_authorization_valid_budget_and_all_credentials():
    import pytest
    from experimental_pilot import runner
    env = {"PILOT_MAX_USD": "1"}
    for p in runner.PROVIDERS:
        env.update({p.upper() + "_MODEL": "test-model", p.upper() + "_API_KEY": "secret",
                    p.upper() + "_INPUT_USD_PER_MILLION": "1",
                    p.upper() + "_OUTPUT_USD_PER_MILLION": "2"})
    with pytest.raises(runner.PilotError, match="live_not_authorized"):
        runner.run(dry_run=False, env=env)
    for value in ("0", "nan", "inf", "-1", "0.000001"):
        with pytest.raises(runner.PilotError):
            runner.run(dry_run=False, authorized=True, env={**env, "PILOT_MAX_USD": value})
    models, budget = runner.live_config(env, True)
    assert len(models) == 2 and 0 < budget["reserved_usd"] < 1
    del env["ANTHROPIC_API_KEY"]
    with pytest.raises(runner.PilotError, match="missing_api_key"):
        runner.run(dry_run=False, authorized=True, env=env)


def test_p1_http_adapters_send_only_one_message_and_parse_usage(monkeypatch):
    from experimental_pilot import runner
    calls = []

    def post(url, headers, body):
        calls.append((url, body))
        usage = {"input_tokens": 12, "output_tokens": 4}
        if "openai.com" in url:
            assert body["store"] is False
            return {"status": "completed", "usage": usage, "output": [
                {"type": "message", "content": [{"type": "output_text", "text": '{"steps":[]}'}]}]}
        return {"stop_reason": "end_turn", "usage": usage,
                "content": [{"type": "text", "text": '{"steps":[]}'}]}

    monkeypatch.setattr(runner, "post", post)
    for p in runner.PROVIDERS:
        client = runner.APIClient(p, "test-model", {p.upper() + "_API_KEY": "secret"})
        assert client.complete("isolated", 256) == ('{"steps":[]}', {"input_tokens": 12, "output_tokens": 4})
    for url, body in calls:
        messages = body.get("input", body.get("messages"))
        assert messages == [{"role": "user", "content": "isolated"}]
        assert "previous_response_id" not in body and "conversation" not in body


def test_p1_transport_never_reports_secret_errors(monkeypatch):
    import pytest
    from experimental_pilot import runner

    def failing(*args, **kwargs):
        raise OSError("secret credential")

    monkeypatch.setattr(runner.request, "build_opener", failing)
    with pytest.raises(runner.PilotError, match="^transport_error$"):
        runner.post("https://api.openai.com/v1/responses", {}, {})


def test_p1_cli_writes_json_and_markdown(tmp_path, monkeypatch):
    import json
    import sys
    from experimental_pilot import runner
    monkeypatch.setattr(sys, "argv", ["runner", "--dry-run", "--output-dir", str(tmp_path)])
    runner.main()
    report = json.loads((tmp_path / "results.json").read_text())
    assert len(report["results"]) == 8
    assert (tmp_path / "summary.md").read_text() == runner.markdown(report)
