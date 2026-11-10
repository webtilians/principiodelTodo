"""Runtime-only parallelization of the *unchanged* S0-I scientific protocol.

Do not change security_lab.s0i's PILOT, CONFIRM, observed_training, one_world,
pilot_gate, confirm_gate, frozen model, memory or evaluator. A shard runs 8 PILOT
or 6 CONFIRM worlds; aggregation requires every expected world and three arms.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import json
import logging
import os
from pathlib import Path
import tempfile
import traceback

from security_lab import s0i
from security_lab.s0 import preflight
from security_lab.s0c import MODEL_REPO, MODEL_FILE, MODEL_REVISION, MODEL_SHA256
from security_lab.s0c import loopback_only
from security_lab.s0d import QwenD, assert_external_connections_blocked
from src.infinito3.mcp_server import AgentMemory

STUDY = "INFINITO-SECURITY-S0-I"
SHARDS = {"pilot": 3, "confirm": 8}
ARMS = ("A0", "A1", "A2")
STAGES = {"pilot": s0i.PILOT, "confirm": s0i.CONFIRM}
EXPECTED_TRAIN = {
    "episodes": 12, "mcp_calls": 36,
    "sqlite_experiences": 12, "versions": ["r1", "r2"],
}


def expected_worlds(stage: str, index: int) -> tuple[s0i.World, ...]:
    if stage not in SHARDS or not (0 <= index < SHARDS[stage]):
        raise ValueError("Invalid S0-I stage or shard")
    return STAGES[stage][index::SHARDS[stage]]


def execute_shard(stage: str, index: int) -> dict:
    """All algorithm and data calls are from unmodified S0-I. Only scope changes."""
    preflight()
    with loopback_only():
        blocked = assert_external_connections_blocked()
        if not blocked:
            raise s0i.Blocked("No external-network isolation")
        model = QwenD()
        with tempfile.TemporaryDirectory(prefix=f"infinito-s0i-{stage}-{index}-") as dirname:
            base = Path(dirname)
            memory = AgentMemory(str(base / "observed_episodes.db"))
            try:
                with s0i.clone(base / "oauth-mcp") as (probe, counter):
                    observations, train = s0i.observed_training(memory, probe)
                    lookup = s0i.literal_lookup(observations)
                    worlds = expected_worlds(stage, index)
                    traces, computed = s0i.trial(worlds, model, memory, lookup, probe)
                    calls = counter["count"]
            finally:
                memory.close()

    if train != EXPECTED_TRAIN:
        raise s0i.Blocked("TRAIN history differs from prospectively fixed experiments")
    if calls != 36 + 3 * len(worlds) or model.calls != 3 * len(worlds):
        raise s0i.Blocked("Unequal real MCP / model calls in shard")
    if len(traces) != 3 * len(worlds):
        raise s0i.Blocked("Missing Qwen model decisions in shard")

    return {
        "study": STUDY, "status": "COMPLETED_SHARD",
        "stage": stage, "shard_index": index, "shard_count": SHARDS[stage],
        "original_world_ids": [w.id for w in worlds],
        "model": {
            "real": True, "repo": MODEL_REPO, "file": MODEL_FILE,
            "revision": MODEL_REVISION, "weights_sha256": MODEL_SHA256,
            "backend": "llama-cpp-python",
            "backend_version": model.backend_version,
            "temperature": 0.0, "seed": 0,
            "calls": model.calls,
            "prompt_tokens": model.prompt_tokens,
            "completion_tokens": model.completion_tokens,
            "seconds": round(model.seconds, 3),
        },
        "train": train,
        "genuine_oauth_mcp_calls": calls,
        "external_connections_blocked": blocked,
        "traces": traces,
        "local_metrics": computed,
        "note": "Independent temporary OAuth/MCP clone and re-created observed SQLite TRAIN, not production.",
    }


def validate_shard(shard: dict, stage: str) -> list[dict]:
    """Fail closed on a missing/duplicated/out-of-split row or mismatched model."""
    if not isinstance(shard, dict) or shard.get("study") != STUDY:
        raise ValueError("Not an S0-I shard")
    index = shard.get("shard_index")
    if type(index) is not int or index not in range(SHARDS[stage]):
        raise ValueError("Unexpected shard index")
    worlds = expected_worlds(stage, index)
    if shard.get("status") != "COMPLETED_SHARD" or shard.get("stage") != stage:
        raise ValueError("Shard not complete or wrong stage")
    if shard.get("shard_count") != SHARDS[stage]:
        raise ValueError("Shard count differs from frozen runtime")
    if shard.get("original_world_ids") != [w.id for w in worlds]:
        raise ValueError("Worlds in shard differ from preregistered split")
    if shard.get("train") != EXPECTED_TRAIN:
        raise ValueError("TRAIN observations are missing")
    model = shard.get("model") or {}
    if (model.get("weights_sha256") != MODEL_SHA256
            or model.get("repo") != MODEL_REPO
            or model.get("revision") != MODEL_REVISION
            or model.get("file") != MODEL_FILE
            or model.get("backend") != "llama-cpp-python"
            or model.get("seed") != 0
            or model.get("temperature") != 0.0
            or model.get("real") is not True):
        raise ValueError("Unpinned or fake model metadata")
    if not shard.get("external_connections_blocked"):
        raise ValueError("External network was not blocked")
    if shard.get("genuine_oauth_mcp_calls") != 36 + len(worlds) * 3:
        raise ValueError("Shard observed incorrect OAuth/MCP calls")
    if model.get("calls") != len(worlds) * 3:
        raise ValueError("Shard inferred incorrect number of Qwen calls")

    rows = shard.get("traces")
    if not isinstance(rows, list) or len(rows) != len(worlds) * 3:
        raise ValueError("Wrong number of traces")
    by_id = {w.id: w for w in worlds}
    expected_pairs = {(w.id, arm) for w in worlds for arm in ARMS}
    actual_pairs = set()
    for r in rows:
        if not isinstance(r, dict):
            raise ValueError("Invalid trace type")
        pair = (r.get("world"), r.get("arm"))
        if pair not in expected_pairs or pair in actual_pairs:
            raise ValueError("Duplicated, unexpected or mislabeled trace")
        actual_pairs.add(pair)
        w = by_id[pair[0]]
        if r.get("profile") != w.profile or r.get("revision") != w.revision:
            raise ValueError("World profile/revision not as preregistered")
        if (r.get("target_present") is not (s0i.private_target(w.profile, w.revision) is not None)
                or r.get("selected") not in s0i.TOOLS):
            raise ValueError("Out-of-contract target/tool")
        # Recompute the oracle independently of the reporter. A positive marker
        # is ONLY the synthetic evaluator overlay on completed real MCP calls.
        matched = r.get("selected") == s0i.private_target(w.profile, w.revision) and r.get("target_present")
        if r.get("correct_probe") is not bool(matched) or r.get("marker_observed") is not bool(matched):
            raise ValueError("Reported outcome contradicts fixed oracle")
        if (r.get("mcp_calls") != 1
                or r.get("valid_json") is not True
                or r.get("token_budget_matched") is not True
                or r.get("retrieved_real_sqlite") is not True):
            raise ValueError("Invalid JSON, prompt mismatch, missing SQLite or MCP call")
        if (type(r.get("prompt_tokens")) is not int or r["prompt_tokens"] <= 0
                or type(r.get("completion_tokens")) is not int or r["completion_tokens"] <= 0):
            raise ValueError("Invalid model usage")
    if actual_pairs != expected_pairs:
        raise ValueError("Traces omitted world/arm")
    if sum(r["prompt_tokens"] for r in rows) != model.get("prompt_tokens"):
        raise ValueError("Tokens do not sum to pinned Qwen usage")
    if sum(r["completion_tokens"] for r in rows) != model.get("completion_tokens"):
        raise ValueError("Generated tokens do not sum correctly")
    # Bound token equality ACTUAL usage within each world (the original used
    # text token estimates; this is an extra fail-closed fairness control).
    for w in worlds:
        uses = [r["prompt_tokens"] for r in rows if r["world"] == w.id]
        if max(uses) - min(uses) > max(20, int(.02 * max(uses))):
            raise ValueError("Actual per-arm prompt tokens are not budget-matched")
    return rows


def aggregate(stage: str, files: list[Path]) -> dict:
    if len(files) != SHARDS[stage]:
        raise ValueError("Missing or extra shard evidence")
    evidence = [json.loads(p.read_text(encoding="utf-8")) for p in files]
    indices = [r.get("shard_index") for r in evidence]
    if set(indices) != set(range(SHARDS[stage])) or len(set(indices)) != len(indices):
        raise ValueError("Missing or duplicate shard index")
    by_pair = {}
    for shard in evidence:
        for row in validate_shard(shard, stage):
            pair = (row["world"], row["arm"])
            if pair in by_pair:
                raise ValueError("Duplicate world-arm across shards")
            by_pair[pair] = row
    # The result order and scoring are identical to original sequential trial.
    ordered = [by_pair[(w.id, arm)] for w in STAGES[stage] for arm in ARMS]
    if len(ordered) != len(STAGES[stage]) * 3:
        raise ValueError("Incomplete global evaluation")
    summary = {arm: s0i.metrics(ordered, arm) for arm in ARMS}
    gate = s0i.pilot_gate(summary) if stage == "pilot" else s0i.confirm_gate(summary)
    return {
        "study": STUDY, "status": "COMPLETED",
        "stage": stage, "gate": bool(gate),
        "gate_original": "s0i.pilot_gate" if stage == "pilot" else "s0i.confirm_gate",
        "worlds": len(STAGES[stage]),
        "model_calls": len(ordered),
        "genuine_mcp_eval_calls": len(ordered),
        "train_replay_mcp_calls": SHARDS[stage] * 36,
        "all_shards_successful": True,
        "no_production_access": True,
        "metrics": summary,
        "shard_hashes": [{"index": s["shard_index"],
                          "model_sha256": s["model"]["weights_sha256"],
                          "model_calls": s["model"]["calls"],
                          "prompt_tokens": s["model"]["prompt_tokens"],
                          "completion_tokens": s["model"]["completion_tokens"]}
                         for s in sorted(evidence, key=lambda x: x["shard_index"])],
        "traces": ordered,
        "decision": (
            "PILOT GO — run frozen CONFIRM without modifying protocol"
            if stage == "pilot" and gate else
            "PILOT NO-GO — skip CONFIRM"
            if stage == "pilot" else
            "CONFIRM GO provisional; further robust baselines needed"
            if gate else
            "CONFIRM NO-GO — not replicated on preregistered holdout"
        ),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S0-I scientific-identical parallel runtime")
    subs = parser.add_subparsers(dest="command", required=True)
    shard = subs.add_parser("shard")
    shard.add_argument("--stage", choices=list(SHARDS), required=True)
    shard.add_argument("--index", type=int, required=True)
    shard.add_argument("--out", required=True)
    merge = subs.add_parser("aggregate")
    merge.add_argument("--stage", choices=list(SHARDS), required=True)
    merge.add_argument("--inputs", required=True)
    merge.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    logging.disable(logging.CRITICAL)
    outfile = Path(args.out)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    try:
        if args.command == "shard":
            # Prevent OAuth URLs, tool outputs and private memory from CI logs.
            with open(os.devnull, "w", encoding="utf-8") as null:
                with redirect_stdout(null), redirect_stderr(null):
                    report = execute_shard(args.stage, args.index)
        else:
            files = sorted(Path(args.inputs).glob("*/report.json"))
            report = aggregate(args.stage, files)
        outfile.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
        print(json.dumps({
            "study": STUDY, "status": report["status"],
            "stage": args.stage,
            "gate": report.get("gate"),
            "decision": report.get("decision"),
            "metrics": report.get("metrics"),
            "shard_index": report.get("shard_index"),
            "model_calls": report.get("model_calls", report.get("model", {}).get("calls")),
        }, ensure_ascii=False))
        return 0
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        blocked = {
            "study": STUDY, "status": "BLOCKED", "stage": args.stage,
            "shard_index": getattr(args, "index", None),
            "exception_class": type(exc).__name__,
            "last_frames": [{"function": frame.name, "line": frame.lineno}
                            for frame in frames[-8:]],
        }
        outfile.with_name("blocked.json").write_text(
            json.dumps(blocked, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(blocked, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
