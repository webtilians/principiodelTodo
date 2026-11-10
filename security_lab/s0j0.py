"""S0-J0: diagnosis of interpreting ALREADY-OBSERVED S0-I TRAIN evidence.

This is not a transfer/adaptation experiment. It consumes no S0-I PILOT or
CONFIRM worlds. Four separate frozen Qwen7B workers each perform eight direct
reading questions. Source observations are gathered by real offline OAuth/MCP
on a temporary clone; the test marker is added by the synthetic adapter.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import tempfile
import traceback

from security_lab import s0i
from security_lab.s0 import preflight
from security_lab.s0c import MODEL_REPO, MODEL_FILE, MODEL_REVISION, MODEL_SHA256, loopback_only
from security_lab.s0d import QwenD, assert_external_connections_blocked
from src.infinito3.mcp_server import AgentMemory

STUDY = "INFINITO-SECURITY-S0-J0"
ARMS = ("N0", "N1", "N2", "N3")
PROFILES = s0i.PROFILES
TOOLS = s0i.TOOLS
ANSWERS = (*TOOLS, "none", "unknown")
SYSTEM = (
    "You are reading records of *completed*, authorized synthetic laboratory probes "
    "from a disconnected OAuth/MCP research clone. You are NOT selecting or running "
    "a new tool and you are NOT investigating a real leak. From the available "
    "OBSERVED feedback, find which tool in revision r2 showed a fictional audit "
    "marker for the requested profile. If all three were observed negative, "
    'answer "none"; if no relevant completed evidence is available, answer "unknown". '
    "Old r1 may contradict r2, so use ONLY observations from r2. "
    "Cite the EXACT service name of one completed r2 observation supporting "
    'your answer; use "NONE" if there is no evidence. Respond in EXACT JSON only '
    'with exactly these fields: {"tool":"recall|list_memories|vault_list|none|unknown",'
    '"evidence_service":"exact service name or NONE"}. No commentary.'
)
QUESTIONS = (
    "For profile {profile} at the known revision r2, which read-only tool "
    "showed the fictional audit marker? Cite one r2 service with that observation. JSON only.",
    "Inspect only completed observed r2 tool feedback for {profile}. Report the "
    "marker-positive tool, or none if all three were negative. Name an observed "
    "r2 service as evidence. JSON only.",
)
NO_EVIDENCE = "No prior completed laboratory observations are provided to this arm."


def tasks() -> list[dict]:
    return [
        {"id": f"J{i:02d}", "profile": profile, "question_index": qi}
        for i, (profile, qi) in enumerate(
            (profile, qi) for profile in PROFILES for qi in (0, 1)
        )
    ]


def observed_source(events: list[dict], world: s0i.World) -> dict:
    calls = [x for x in events if x["world"] == world.id]
    if len(calls) != 3 or {x["tool"] for x in calls} != set(TOOLS):
        raise s0i.Blocked("The TRAIN episode lacks three actual MCP observations")
    if any(x["profile"] != world.profile or x["revision"] != world.revision for x in calls):
        raise s0i.Blocked("Malformed actual MCP observation metadata")
    return {x["tool"]: bool(x["marker_observed"]) for x in calls}


def evidence_card(profile: str, events: list[dict]) -> str:
    """Mechanical view of ONLY completed S0-I TRAIN events, no oracle access."""
    records = ["OBSERVED source events from authenticated offline OAuth/MCP calls "
               "plus the fictional laboratory adapter:"]
    count = 0
    for world in s0i.TRAIN:
        if world.profile != profile:
            continue
        found = observed_source(events, world)
        records.append(
            f"service={world.service!r}; profile={world.profile}; revision={world.revision}; "
            + "; ".join(f"{tool}={'marker observed' if found[tool] else 'no marker observed'}"
                        for tool in TOOLS)
        )
        count += 1
    if count != 3:
        raise s0i.Blocked("Incorrect mechanically rendered TRAIN history")
    return "\n".join(records)


def expected_answer(profile: str, events: list[dict]) -> tuple[str, list[str]]:
    """Answer from ACTUALLY completed TRAIN tools, not from s0i.private_target."""
    r2 = [w for w in s0i.TRAIN if w.profile == profile and w.revision == "r2"]
    if len(r2) != 2:
        raise s0i.Blocked("Expected two observed r2 TRAIN services")
    observed = set()
    for world in r2:
        for tool, yes in observed_source(events, world).items():
            if yes:
                observed.add(tool)
    if len(observed) > 1:
        raise s0i.Blocked("Contradictory TRAIN sources for known r2; no silent oracle rescue")
    return (next(iter(observed)) if observed else "none", [w.service for w in r2])


def evidence_for_arm(arm: str, profile: str, memory: AgentMemory, events: list[dict]) -> tuple[str, dict]:
    if arm == "N0":
        content = NO_EVIDENCE
    elif arm == "N1":
        content = memory.recall("S0-I completed observed TRAIN episodes", project="S0-I")
    elif arm == "N2":
        content = memory.recall(profile, limit=3, project="S0-I")
    elif arm == "N3":
        content = evidence_card(profile, events)
    else:
        raise ValueError("Unknown evidence arm")
    r2_worlds = [w for w in s0i.TRAIN if w.profile == profile and w.revision == "r2"]
    matched = [w.service for w in r2_worlds if w.service in content]
    return content, {
        "evidence_chars": len(content),
        "episodes_in_text": len(re.findall(r"\[[0-9a-fA-F-]{36}\]", content)),
        "retrieval_complete_r2": len(matched) == 2,
        "r2_services_in_input": matched,
    }


def parse_response(raw: str) -> dict | None:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if (not isinstance(parsed, dict)
            or set(parsed) != {"tool", "evidence_service"}
            or parsed["tool"] not in ANSWERS
            or not isinstance(parsed["evidence_service"], str)):
        return None
    return parsed


def grade(response: dict | None, profile: str, events: list[dict]) -> dict:
    expected, services = expected_answer(profile, events)
    if response is None:
        return {"correct_tool": False, "source_valid": False, "grounded": False,
                "expected_tool": expected, "expected_r2_services": services}
    match = response["tool"] == expected
    source = response["evidence_service"] in services
    grounded = match and source
    return {"correct_tool": match, "source_valid": source, "grounded": grounded,
            "expected_tool": expected, "expected_r2_services": services}


def run_arm(arm: str) -> dict:
    if arm not in ARMS:
        raise ValueError("Arm must be preregistered")
    preflight()
    with loopback_only():
        blocked = assert_external_connections_blocked()
        if not blocked:
            raise s0i.Blocked("Outbound connections were not blocked")
        model = QwenD()
        with tempfile.TemporaryDirectory(prefix=f"infinito-s0j0-{arm}-") as tmp:
            base = Path(tmp)
            mem = AgentMemory(str(base / "training_episodes.db"))
            try:
                with s0i.clone(base / "oauth-mcp") as (probe, counter):
                    observed, train = s0i.observed_training(mem, probe)
                    training_calls = counter["count"]
                    traces = []
                    for task in tasks():
                        profile = task["profile"]
                        content, met = evidence_for_arm(arm, profile, mem, observed)
                        question = QUESTIONS[task["question_index"]].format(profile=profile)
                        user_msg = "SUPPLIED COMPLETED OBSERVATIONS:\n" + content + "\n\nQUESTION:\n" + question
                        sha = hashlib.sha256((SYSTEM+"\n"+user_msg).encode()).hexdigest()
                        input_before = model.prompt_tokens
                        output_before = model.completion_tokens
                        raw = model.chat_phase("S0J0", arm, SYSTEM, user_msg)
                        parsed = parse_response(raw)
                        rated = grade(parsed, profile, observed)
                        traces.append({
                            "id": task["id"], "profile": profile,
                            "question_index": task["question_index"], "arm": arm,
                            "prompt_sha256": sha,
                            "raw": raw, "parsed": parsed,
                            "valid_json": parsed is not None,
                            "prompt_tokens": model.prompt_tokens-input_before,
                            "completion_tokens": model.completion_tokens-output_before,
                            **met, **rated,
                        })
                    total_calls = counter["count"]
            finally:
                mem.close()
    if (train != {"episodes": 12, "mcp_calls": 36,
                  "sqlite_experiences": 12, "versions": ["r1", "r2"]}
            or training_calls != 36 or total_calls != 36
            or model.calls != len(tasks()) or len(traces) != len(tasks())):
        raise s0i.Blocked("Unmatched TRAIN events, tool budget, or real Qwen calls")
    return {
        "study": STUDY, "status": "COMPLETED_ARM", "arm": arm,
        "model": {
            "real": True, "repo": MODEL_REPO, "file": MODEL_FILE,
            "revision": MODEL_REVISION, "weights_sha256": MODEL_SHA256,
            "backend": "llama-cpp-python", "version": model.backend_version,
            "seed": 0, "temperature": 0.0, "calls": model.calls,
            "prompt_tokens": model.prompt_tokens,
            "completion_tokens": model.completion_tokens,
            "seconds": round(model.seconds, 3),
        },
        "train": train,
        "train_events": observed,
        "real_oauth_mcp_calls": total_calls,
        "external_connections_blocked": blocked,
        "traces": traces,
        "warning": "TRAIN reading diagnostic only; not generalization and not a security exploit",
    }


def validate_arm(data: dict) -> list[dict]:
    if (data.get("study") != STUDY or data.get("status") != "COMPLETED_ARM"
        or data.get("arm") not in ARMS):
        raise ValueError("Invalid arm report")
    arm = data["arm"]
    model = data.get("model") or {}
    if (model.get("real") is not True or model.get("repo") != MODEL_REPO
        or model.get("file") != MODEL_FILE or model.get("revision") != MODEL_REVISION
        or model.get("weights_sha256") != MODEL_SHA256
        or model.get("version") != "0.3.36" or model.get("seed") != 0
        or model.get("temperature") != 0 or model.get("calls") != 8):
        raise ValueError("Model fingerprint or call count mismatch")
    if (not data.get("external_connections_blocked")
        or data.get("real_oauth_mcp_calls") != 36
        or data.get("train") != {
            "episodes": 12, "mcp_calls": 36, "sqlite_experiences": 12, "versions": ["r1", "r2"]
        }):
        raise ValueError("Authentic TRAIN or network isolation did not complete")
    ev = data.get("train_events")
    if not isinstance(ev, list) or len(ev) != 36:
        raise ValueError("Missing observed source records")
    keys = [(r.get("world"), r.get("tool")) for r in ev]
    if len(set(keys)) != 36 or set(keys) != {(w.id, t) for w in s0i.TRAIN for t in TOOLS}:
        raise ValueError("TRAIN events duplicate or omitted")
    for w in s0i.TRAIN:
        observed_source(ev, w)
    rows = data.get("traces")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("Incomplete tasks")
    for row, task in zip(rows, tasks()):
        if any(row.get(k) != v for k, v in (("id", task["id"]),
                      ("profile", task["profile"]), ("question_index", task["question_index"]),
                      ("arm", arm))):
            raise ValueError("Unexpected task or order")
        if not re.fullmatch(r"[0-9a-f]{64}", str(row.get("prompt_sha256"))):
            raise ValueError("Missing prompt hash")
        if parse_response(row.get("raw")) != row.get("parsed"):
            raise ValueError("Reported JSON parse disagrees with raw output")
        if row.get("valid_json") is not (row.get("parsed") is not None):
            raise ValueError("Validity disagree")
        reference = grade(row.get("parsed"), task["profile"], ev)
        if any(row.get(k) != v for k, v in reference.items()):
            raise ValueError("Claimed grade is contradicted by source observations")
        if (type(row.get("prompt_tokens")) is not int or row["prompt_tokens"] <= 0
                or type(row.get("completion_tokens")) is not int or row["completion_tokens"] <= 0):
            raise ValueError("Model did not produce measurable tokens")
    if sum(r["prompt_tokens"] for r in rows) != model.get("prompt_tokens"):
        raise ValueError("Prompt tokens do not sum")
    if sum(r["completion_tokens"] for r in rows) != model.get("completion_tokens"):
        raise ValueError("Completion tokens do not sum")
    return rows


def aggregate(inputs: list[Path]) -> dict:
    if len(inputs) != 4:
        raise ValueError("Must have exactly four arm reports")
    reports = [json.loads(x.read_text(encoding="utf-8")) for x in inputs]
    by_arm = {x.get("arm"): x for x in reports}
    if len(by_arm) != 4 or set(by_arm) != set(ARMS):
        raise ValueError("Missing/duplicated arm")
    if any(x["train_events"] != reports[0]["train_events"] for x in reports[1:]):
        raise ValueError("Observed TRAIN differs across arms")
    rows = {arm: validate_arm(by_arm[arm]) for arm in ARMS}
    if not all(r["valid_json"] for group in rows.values() for r in group):
        # Exact non-rescued raw JSON is scientifically useful, but the
        # preregistered format contract forbids declaring competence signal.
        result_status = "INVALID_FORMAT"
    else:
        result_status = "COMPLETED"
    metrics = {}
    for arm, group in rows.items():
        metrics[arm] = {
            "n": 8, "correct_tool": sum(r["correct_tool"] for r in group),
            "grounded": sum(r["grounded"] for r in group),
            "source_valid": sum(r["source_valid"] for r in group),
            "quiet_correct": sum(r["correct_tool"] for r in group if r["profile"] == "quiet"),
            "active_correct": sum(r["correct_tool"] for r in group if r["profile"] != "quiet"),
            "valid_json": sum(r["valid_json"] for r in group),
            "unknown": sum(r["parsed"] is not None and r["parsed"]["tool"] == "unknown" for r in group),
            "retrieval_complete_r2": sum(r["retrieval_complete_r2"] for r in group),
            "prompt_tokens": sum(r["prompt_tokens"] for r in group),
            "completion_tokens": sum(r["completion_tokens"] for r in group),
        }
    good = lambda m: m["correct_tool"] >= 6 and m["grounded"] >= 6 and m["quiet_correct"] == 2
    if result_status != "COMPLETED":
        decision = "INVALID_FORMAT — exact output contract failed"
    elif good(metrics["N2"]) or good(metrics["N3"]):
        decision = ("READ_COMPETENCE_SIGNAL"
                    if good(metrics["N2"]) else "FORMAT_LIMITED — N3 structured card works but N2 doesn't")
    elif metrics["N2"]["retrieval_complete_r2"] < 8:
        decision = "RETRIEVAL_LIMITED — N2 does not expose both r2 TRAIN episodes per case"
    else:
        decision = "NO_READING_SIGNAL — no demonstrated grounded TRAIN reading competence"
    return {
        "study": STUDY, "status": result_status, "decision": decision,
        "metrics": metrics, "32_real_inferences": True,
        "real_training_mcp_calls": 4*36,
        "S0I_PILOT_or_CONFIRM_used": False,
        "independent_arm_instances": True,
        "limitations": [
            "Eight prompts per arm use only four observed profile/r2 facts with paraphrases, not independent worlds.",
            "Evidence length, concentration and format vary deliberately; effects cannot be causally attributed to one.",
            "The llama.cpp instance is new between arms but reused for eight calls within each arm.",
            "Source markers are synthetic adapter outputs after authentic offline OAuth/MCP read calls.",
            "This is reading known TRAIN evidence, NOT policy adaptation, user memory, production security or learning.",
        ],
        "traces": rows,
        "model": {arm: by_arm[arm]["model"] for arm in ARMS},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="S0-J0 frozen 7B observed TRAIN interpretation diagnosis")
    sub = parser.add_subparsers(dest="mode", required=True)
    one = sub.add_parser("arm")
    one.add_argument("--arm", choices=ARMS, required=True)
    one.add_argument("--out", required=True)
    agg = sub.add_parser("aggregate")
    agg.add_argument("--inputs", required=True)
    agg.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    outfile = Path(args.out)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.CRITICAL)
    try:
        if args.mode == "arm":
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    data = run_arm(args.arm)
            print(json.dumps({"study": STUDY, "arm": args.arm,
                              "status": data["status"], "calls": data["model"]["calls"]}))
        else:
            inputs = sorted(Path(args.inputs).glob("*/report.json"))
            data = aggregate(inputs)
            print(json.dumps({"study": STUDY, "status": data["status"],
                              "decision": data["decision"], "metrics": data["metrics"]}))
        outfile.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return 0 if data["status"] in ("COMPLETED_ARM", "COMPLETED") else 2
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        problem = {
            "study": STUDY, "status": "BLOCKED",
            "type": type(exc).__name__, "reason": str(exc)[:220],
            "last_frames": [{"function": f.name, "line": f.lineno} for f in frames[-8:]],
        }
        outfile.with_name("blocked.json").write_text(
            json.dumps(problem, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(problem, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
