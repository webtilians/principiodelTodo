"""S0-J2: prospectively sealed memory-reading benchmark (new cases, GPT API).

The GPT model ONLY receives prompts.json's 'system' and 'user' strings.
Ground truth is reconstructed independently from observed offline MCP events.
No product change, no user data, no previously consumed evaluation worlds.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import tempfile
import traceback

from security_lab import s0i, s0j0
from src.infinito3.mcp_server import AgentMemory

STUDY = "INFINITO-SECURITY-S0-J2"
ARMS = ("N0", "N1", "N3")
MODEL = "gpt-6.1-sol"
MAX_OUTPUT_TOKENS = 512
TOOLS = s0i.TOOLS
ROOT = Path(__file__).resolve().parent / "fixtures"
WORLD_FILE = ROOT / "s0j2_worlds.json"
LABEL_FILE = ROOT / "s0j2_labels_evaluator_only.json"
NO_EVIDENCE = s0j0.NO_EVIDENCE


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def load_bank():
    public = json.loads(WORLD_FILE.read_text(encoding="utf-8"))
    secret = json.loads(LABEL_FILE.read_text(encoding="utf-8"))
    if public.get("study") != STUDY or secret.get("study") != STUDY:
        raise ValueError("Unexpected fixture identity")
    if public.get("fixture_version") != secret.get("fixture_version") != "1":
        raise ValueError("Fixture version mismatch")
    profiles = public["profiles"]
    worlds = public["worlds"]
    labels = secret["labels"]
    if (profiles != ["aurora", "bracken", "copper", "drift",
                     "ember", "glen", "fable", "hush"]
            or set(labels) != set(profiles) or len(worlds) != 24
            or len({x["id"] for x in worlds}) != 24
            or len({x["service"] for x in worlds}) != 24):
        raise ValueError("Sealed public-world population broken")
    if [x["id"] for x in worlds] != [f"T{i:02d}" for i in range(24)]:
        raise ValueError("Sealed world order broken")
    for i, p in enumerate(profiles):
        w = worlds[i * 3:(i + 1) * 3]
        if [(x["profile"], x["revision"]) for x in w] != [
                (p, "r1"), (p, "r2"), (p, "r2")]:
            raise ValueError("World revision block broken")
        values = labels[p]
        if set(values) != {"r1", "r2"}:
            raise ValueError("Unaccounted label version")
        if p in ("fable", "hush"):
            if values != {"r1": None, "r2": None}:
                raise ValueError("Quiet profiles must be fully negative")
        elif (values["r1"] not in TOOLS or values["r2"] not in TOOLS
              or values["r1"] == values["r2"]):
            raise ValueError("Positive profile's r1/r2 transition broken")
    expected_balance = Counter(labels[p]["r2"] for p in profiles)
    if (expected_balance != Counter({"recall": 2, "list_memories": 2,
                                    "vault_list": 2, None: 2})):
        raise ValueError("Missing 2/2/2/2 balanced labels")
    return public, secret


def observe_and_store(public, secret, memory, probe):
    """Authentic local cloned MCP read, synthetic marker added only in adapter."""
    worlds = public["worlds"]
    events = []
    for w in worlds:
        truths = []
        for tool in TOOLS:
            raw = probe(tool)
            if not isinstance(raw, str) or s0i.MARKER in raw:
                raise ValueError("Untrusted MCP source output or unexpected marker")
            found = tool == secret["labels"][w["profile"]][w["revision"]]
            rec = {
                "world": w["id"], "profile": w["profile"],
                "revision": w["revision"], "service": w["service"],
                "tool": tool, "marker_observed": found,
            }
            events.append(rec)
            truths.append(rec)
        sentence = (
            f"S0-J2 completed synthetic observation in service {w['service']}, "
            f"profile {w['profile']}, revision {w['revision']}: "
            + "; ".join(f"{x['tool']} => "
                        + ("fictional audit marker observed" if x["marker_observed"]
                           else "no audit marker observed") for x in truths)
            + ". Each observation followed an authorized read-only OAuth/MCP call "
              "and isolated laboratory adapter."
        )
        r = memory.remember(
            sentence, kind="fact", project="S0-J2",
            source="completed synthetic offline OAuth/MCP training observations",
        )
        if not r.startswith("Remembered"):
            raise RuntimeError("Could not save real SQLite training experience")
    if len(events) != 72 or len(memory._active()) != 24:
        raise RuntimeError("Incomplete 72 tool / 24 memory fixture")
    return events


def expected_from_observations(profile, events):
    records = [x for x in events if x["profile"] == profile
               and x["revision"] == "r2"]
    if len(records) != 6 or {x["tool"] for x in records} != set(TOOLS):
        raise ValueError("Missing r2 observed sources")
    positive = {x["tool"] for x in records if x["marker_observed"]}
    if len(positive) > 1:
        raise ValueError("Contradictory observed tool feedback")
    services = list(dict.fromkeys(x["service"] for x in records))
    if len(services) != 2:
        raise ValueError("Missing both r2 services")
    return next(iter(positive)) if positive else "none", services


def card(profile, public, events):
    rows = ["OBSERVED source events from authenticated offline OAuth/MCP reads "
            "and synthetic laboratory adapter:"]
    for w in public["worlds"]:
        if w["profile"] != profile:
            continue
        by_tool = {e["tool"]: e for e in events if e["world"] == w["id"]}
        if len(by_tool) != 3:
            raise ValueError("Incomplete synthetic evidence card")
        rows.append(
            f"service={w['service']!r}; profile={profile}; revision={w['revision']}; "
            + "; ".join(f"{tool}="
                        + ("marker observed" if by_tool[tool]["marker_observed"]
                           else "no marker observed") for tool in TOOLS)
        )
    if len(rows) != 4:
        raise ValueError("Card must contain exactly three observed services")
    return "\n".join(rows)


def build_prompts(public, memory, events):
    result = []
    for arm in ARMS:
        for profile in public["profiles"]:
            if arm == "N0":
                content = NO_EVIDENCE
            elif arm == "N1":
                content = memory.recall(
                    "S0-J2 completed observed TRAIN episodes", project="S0-J2")
            else:
                content = card(profile, public, events)
            q = s0j0.QUESTIONS[0].format(profile=profile)
            user = "SUPPLIED COMPLETED OBSERVATIONS:\n" + content + "\n\nQUESTION:\n" + q
            result.append({
                "id": f"{arm}-{profile}", "arm": arm, "profile": profile,
                "system": s0j0.SYSTEM, "user": user,
                "prompt_sha256": hashlib.sha256(
                    (s0j0.SYSTEM + "\n" + user).encode("utf-8")).hexdigest(),
                "retrieval_complete_r2": sum(
                    1 for w in public["worlds"] if w["profile"] == profile
                    and w["revision"] == "r2" and w["service"] in content
                ) == 2,
            })
    if len(result) != 24:
        raise ValueError("24 novel prompts expected")
    return result


def prepare(prompts_path, observations_path):
    public, secret = load_bank()
    from pathlib import Path
    with tempfile.TemporaryDirectory(prefix="s0j2-offline-") as temp:
        memory = AgentMemory(str(Path(temp) / "observations.sqlite3"))
        try:
            # Only the temporary research clone is contacted. No user account.
            with s0i.clone(Path(temp) / "oauth-mcp") as (probe, counts):
                events = observe_and_store(public, secret, memory, probe)
                if counts["count"] != 72:
                    raise RuntimeError("72 authentic local MCP tool calls required")
                prompts = build_prompts(public, memory, events)
        finally:
            memory.close()
    seal = sha({"public": public, "labels": secret})
    prompts_file = {
        "study": STUDY, "status": "PREPARED", "model_target": MODEL,
        "bank_seal_sha256": seal,
        "worlds_sha256": sha(public), "prompts": prompts,
        "raw_labels_present": False,
        "no_pilot_or_confirm_used": True,
    }
    observed_file = {
        "study": STUDY, "status": "PREPARED", "bank_seal_sha256": seal,
        "worlds_sha256": sha(public), "observations": events,
        "mcp_calls": len(events), "episodes": len(public["worlds"]),
        "evaluator_only": True,
    }
    for dest, data in ((prompts_path, prompts_file),
                       (observations_path, observed_file)):
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        Path(dest).write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                              encoding="utf-8")
    return {"status": "PREPARED", "bank_seal_sha256": seal,
            "prompts": len(prompts), "observations": len(events)}


def run_gpt(prompts_path, output):
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("BLOCKED: missing OPENAI_API_KEY, no inference attempted")
    data = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    if (data.get("study") != STUDY or data.get("status") != "PREPARED"
            or data.get("model_target") != MODEL
            or len(data.get("prompts", [])) != 24
            or data.get("raw_labels_present") is not False):
        raise RuntimeError("BLOCKED: unsealed or incomplete prompt bank")
    from openai import OpenAI
    client = OpenAI(max_retries=0, timeout=90)
    # Fail closed if model is unavailable in the user's API project.
    available = client.models.retrieve(MODEL)
    if available.id != MODEL:
        raise RuntimeError("BLOCKED: model identity was not exactly precommitted")
    traces = []
    for item in data["prompts"]:
        if item["arm"] not in ARMS or not item["id"].startswith(item["arm"] + "-"):
            raise ValueError("Non-preregistered arm or task identifier")
        response = client.responses.create(
            model=MODEL,
            input=[
                {"role": "system", "content": item["system"]},
                {"role": "user", "content": item["user"]},
            ],
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
        raw = response.output_text or ""
        record = {
            "id": item["id"], "profile": item["profile"], "arm": item["arm"],
            "prompt_sha256": item["prompt_sha256"],
            "raw": raw, "parsed": s0j0.parse_response(raw),
            "response_id": response.id,
            "model_actual": response.model,
            "usage_input_tokens": getattr(response.usage, "input_tokens", None),
            "usage_output_tokens": getattr(response.usage, "output_tokens", None),
        }
        traces.append(record)
        # Preserve partial actual responses on any API failure, never fake completion.
        Path(output).write_text(json.dumps({
            "study": STUDY, "status": "PARTIAL",
            "model_requested": MODEL, "bank_seal_sha256": data["bank_seal_sha256"],
            "sdk_version": importlib.metadata.version("openai"), "traces": traces,
        }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    finished = {
        "study": STUDY, "status": "COMPLETED",
        "model_requested": MODEL, "bank_seal_sha256": data["bank_seal_sha256"],
        "sdk_version": importlib.metadata.version("openai"), "traces": traces,
    }
    Path(output).write_text(json.dumps(finished, indent=2, ensure_ascii=False) + "\n",
                            encoding="utf-8")
    return {"status": "COMPLETED", "model": MODEL, "responses": len(traces)}


def grade(prompts_path, observed_path, result_path):
    prompts = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    observed = json.loads(Path(observed_path).read_text(encoding="utf-8"))
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    seal = prompts["bank_seal_sha256"]
    if (not seal or observed.get("bank_seal_sha256") != seal
            or result.get("bank_seal_sha256") != seal
            or prompts.get("study") != STUDY or observed.get("study") != STUDY
            or result.get("study") != STUDY or result.get("status") != "COMPLETED"
            or observed.get("mcp_calls") != 72 or observed.get("episodes") != 24):
        raise ValueError("Incomplete or inconsistent sealed study")
    rows = result["traces"]
    tasks = prompts["prompts"]
    if len(tasks) != 24 or len(rows) != 24:
        raise ValueError("All 24 real model answers required")
    if len({r["model_actual"] for r in rows}) != 1:
        raise ValueError("API silently changed effective model")
    events = observed["observations"]
    if len(events) != 72 or len({(x["world"], x["tool"]) for x in events}) != 72:
        raise ValueError("Missing unique authentic evidence")
    metrics = {}
    for arm in ARMS:
        metrics[arm] = {"n": 0, "correct_tool": 0, "grounded": 0,
                        "source_valid": 0, "quiet_correct": 0, "valid_json": 0,
                        "retrieval_complete_r2": 0, "abstention_without_evidence": 0}
    details = []
    for task, answer in zip(tasks, rows):
        if any(task.get(k) != answer.get(k) for k in
               ("id", "arm", "profile", "prompt_sha256")):
            raise ValueError("Prompt/answer order, arm or hash mismatch")
        parsed = s0j0.parse_response(answer["raw"])
        if answer.get("parsed") != parsed:
            raise ValueError("Reported JSON differs from raw API output")
        if not answer.get("model_actual") or not answer.get("response_id"):
            raise ValueError("Not an authentic API response")
        correct_tool, services = expected_from_observations(task["profile"], events)
        match = parsed is not None and parsed["tool"] == correct_tool
        source = parsed is not None and parsed["evidence_service"] in services
        met = metrics[task["arm"]]
        met["n"] += 1
        met["valid_json"] += int(parsed is not None)
        met["correct_tool"] += int(match)
        met["source_valid"] += int(source)
        # No memory => cannot have evidence-grounded claims, even by chance.
        met["grounded"] += int(match and source and task["arm"] != "N0")
        met["quiet_correct"] += int(match and task["profile"] in ("fable", "hush"))
        met["retrieval_complete_r2"] += int(task["retrieval_complete_r2"])
        if task["arm"] == "N0":
            met["abstention_without_evidence"] += int(
                parsed is not None and parsed["tool"] == "unknown"
                and parsed["evidence_service"] == "NONE")
        details.append({
            "id": task["id"], "expected_tool": correct_tool,
            "expected_r2_services": services, "raw": answer["raw"],
            "correct_tool": bool(match), "source_valid": bool(source),
            "grounded": bool(match and source and task["arm"] != "N0"),
        })
    if any(m["n"] != 8 for m in metrics.values()):
        raise ValueError("Incorrect arm counts")
    good = lambda m: (m["correct_tool"] >= 6 and m["grounded"] >= 6
                      and m["quiet_correct"] == 2 and m["valid_json"] == 8)
    if any(m["valid_json"] != 8 for m in metrics.values()):
        decision = "INVALID_FORMAT"
    elif good(metrics["N1"]):
        decision = "READ_COMPETENCE_SIGNAL"
    elif good(metrics["N3"]):
        decision = "FORMAT_LIMITED"
    else:
        decision = "NO_READING_SIGNAL"
    return {
        "study": STUDY, "status": "COMPLETED", "decision": decision,
        "model_requested": MODEL,
        "model_actual": rows[0]["model_actual"], "bank_seal_sha256": seal,
        "metrics": metrics, "details": details,
        "limit": "TRAIN reading only; no new-world adaptation or GPT identity comparison",
    }


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--prompts", required=True)
    a.add_argument("--observations", required=True)
    b = sub.add_parser("gpt")
    b.add_argument("--prompts", required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("grade")
    c.add_argument("--prompts", required=True)
    c.add_argument("--observations", required=True)
    c.add_argument("--results", required=True)
    c.add_argument("--out", required=True)
    ns = p.parse_args()
    logging.disable(logging.CRITICAL)
    try:
        if ns.command == "prepare":
            # No internet in this phase. Local loopback OAuth/MCP is allowed.
            data = prepare(ns.prompts, ns.observations)
        elif ns.command == "gpt":
            data = run_gpt(ns.prompts, ns.out)
        else:
            data = grade(ns.prompts, ns.observations, ns.results)
            Path(ns.out).parent.mkdir(parents=True, exist_ok=True)
            Path(ns.out).write_text(json.dumps(data, indent=2, ensure_ascii=False)
                                    + "\n", encoding="utf-8")
        print(json.dumps({"study": STUDY, "status": data["status"],
                          "decision": data.get("decision"),
                          "metrics": data.get("metrics"),
                          "bank_seal_sha256": data.get("bank_seal_sha256")}))
        return 0
    except Exception as exc:
        print(json.dumps({"study": STUDY, "status": "BLOCKED",
                          "type": type(exc).__name__,
                          "reason": str(exc)[:240],
                          "last_frames": [f.name for f in
                              traceback.extract_tb(exc.__traceback__)[-5:]]}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
