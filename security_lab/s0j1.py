"""S0-J1: preregistered 14B short reading preflight on S0-J0 consumed TRAIN.

Exploratory model-capacity diagnostic, NOT independent confirmation or a product test.
The scientific prompts, observations and grading are imported unchanged from S0-J0.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import traceback

from security_lab import s0j0, s0i
from security_lab.s0d import QwenD

STUDY = "INFINITO-SECURITY-S0-J1"
ARMS = ("N0", "N1", "N3")
MODEL_REPO = "bartowski/Qwen2.5-14B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-14B-Instruct-Q4_K_M.gguf"
MODEL_REVISION = "05244aa5d871c661c80082a15d3bce44714d068d"
ORIGINAL_TASKS = s0j0.tasks


def tasks():
    return [t for t in ORIGINAL_TASKS() if t["question_index"] == 0]


class Frozen14B(QwenD):
    """Only the model identity differs from the frozen S0-J0 driver."""

    def __init__(self):
        from llama_cpp import Llama
        path = Path("/models/model.gguf")
        if not path.is_file():
            raise RuntimeError("Frozen 14B GGUF was not mounted")
        h = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(block)
        actual = h.hexdigest()
        expected = os.environ.get("GGUF_SHA256", "")
        if len(expected) != 64 or actual != expected:
            raise RuntimeError("14B downloaded weights differ from recorded preflight SHA256")
        self.generator = Llama(model_path=str(path), n_ctx=4096, n_threads=4,
                              n_gpu_layers=0, n_batch=256, seed=0,
                              chat_format="chatml", verbose=False)
        self.calls = self.prompt_tokens = self.completion_tokens = 0
        self.seconds = 0.0
        self.by_arm = {}
        self.backend_version = importlib.metadata.version("llama-cpp-python")


def run(arm):
    if arm not in ARMS:
        raise ValueError("Not a preregistered arm")
    old_class, old_tasks = s0j0.QwenD, s0j0.tasks
    try:
        s0j0.QwenD, s0j0.tasks = Frozen14B, tasks
        report = s0j0.run_arm(arm)
    finally:
        s0j0.QwenD, s0j0.tasks = old_class, old_tasks
    report["study"] = STUDY
    report["model"].update({
        "repo": MODEL_REPO, "file": MODEL_FILE, "revision": MODEL_REVISION,
        "weights_sha256": os.environ["GGUF_SHA256"],
    })
    report["warning"] = "Exploratory recycled TRAIN, no independent validation, no production data"
    return report


def validate(data):
    if data.get("study") != STUDY or data.get("status") != "COMPLETED_ARM":
        raise ValueError("Arm report absent/invalid")
    arm = data.get("arm")
    if arm not in ARMS:
        raise ValueError("Arm outside preregistered set")
    m = data.get("model", {})
    if (not m.get("real") or m.get("repo") != MODEL_REPO
            or m.get("file") != MODEL_FILE or m.get("revision") != MODEL_REVISION
            or m.get("version") != "0.3.36" or m.get("calls") != 4
            or m.get("seed") != 0 or m.get("temperature") != 0
            or len(m.get("weights_sha256", "")) != 64):
        raise ValueError("Model version, parameters or call count mismatch")
    if (data.get("real_oauth_mcp_calls") != 36
            or not data.get("external_connections_blocked")
            or data.get("train") != {"episodes": 12, "mcp_calls": 36,
                                      "sqlite_experiences": 12,
                                      "versions": ["r1", "r2"]}):
        raise ValueError("Real isolated 36-call TRAIN was not reproduced")
    ev = data.get("train_events")
    if not isinstance(ev, list) or len(ev) != 36:
        raise ValueError("Missing observed evidence")
    if {(x["world"], x["tool"]) for x in ev} != {
            (world.id, tool) for world in s0i.TRAIN for tool in s0i.TOOLS}:
        raise ValueError("Incomplete unique observed events")
    rows = data.get("traces")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("Four questions per arm required")
    for row, task in zip(rows, tasks()):
        if any(row.get(k) != v for k, v in (
                ("id", task["id"]), ("profile", task["profile"]),
                ("question_index", 0), ("arm", arm))):
            raise ValueError("Unexpected case, arm or order")
        parsed = s0j0.parse_response(row.get("raw"))
        if parsed != row.get("parsed") or row.get("valid_json") is not (parsed is not None):
            raise ValueError("Original unmodified JSON contract misreported")
        grade = s0j0.grade(parsed, task["profile"], ev)
        if any(row.get(k) != v for k, v in grade.items()):
            raise ValueError("Claimed accuracy contradicts ACTUAL training evidence")
        if not (isinstance(row.get("prompt_sha256"), str)
                and len(row["prompt_sha256"]) == 64):
            raise ValueError("Missing prompt fingerprint")
        if row.get("prompt_tokens", 0) <= 0 or row.get("completion_tokens", 0) <= 0:
            raise ValueError("Real token accounting missing")
    if sum(x["prompt_tokens"] for x in rows) != m.get("prompt_tokens"):
        raise ValueError("Input token sum inconsistent")
    if sum(x["completion_tokens"] for x in rows) != m.get("completion_tokens"):
        raise ValueError("Output token sum inconsistent")
    return rows


def aggregate(folder):
    paths = sorted(Path(folder).glob("*/report.json"))
    if len(paths) != 3:
        raise ValueError("Require exactly three model-arm artifacts")
    reports = [json.loads(p.read_text()) for p in paths]
    by_arm = {r.get("arm"): r for r in reports}
    if set(by_arm) != set(ARMS) or len(by_arm) != len(reports):
        raise ValueError("Arms missing or duplicated")
    if any(r["train_events"] != reports[0]["train_events"] for r in reports):
        raise ValueError("Different MCP evidence supplied to different arms")
    if len({r["model"]["weights_sha256"] for r in reports}) != 1:
        raise ValueError("Different 14B weights across arms")
    rows = {arm: validate(by_arm[arm]) for arm in ARMS}
    metrics = {}
    for arm, cases in rows.items():
        metrics[arm] = {
            "n": 4,
            "correct_tool": sum(r["correct_tool"] for r in cases),
            "grounded": (0 if arm == "N0" else sum(r["grounded"] for r in cases)),
            "source_valid": sum(r["source_valid"] for r in cases),
            "quiet_correct": int(cases[3]["correct_tool"]),
            "valid_json": sum(r["valid_json"] for r in cases),
            "retrieval_complete_r2": sum(r["retrieval_complete_r2"] for r in cases),
            "abstention_without_evidence": (
                sum(r["parsed"] is not None and r["parsed"]["tool"] == "unknown"
                    for r in cases) if arm == "N0" else None),
            "prompt_tokens": sum(r["prompt_tokens"] for r in cases),
            "completion_tokens": sum(r["completion_tokens"] for r in cases),
        }
    status = ("COMPLETED" if all(r["valid_json"] for group in rows.values()
                                  for r in group) else "INVALID_FORMAT")
    good = lambda x: (x["correct_tool"] >= 3 and x["grounded"] >= 3
                      and x["quiet_correct"] == 1 and x["valid_json"] == 4)
    if status == "INVALID_FORMAT":
        decision = "INVALID_FORMAT — exact JSON contract failed"
    elif good(metrics["N1"]):
        decision = "EXPLORATORY_READ_COMPETENCE_SIGNAL"
    elif good(metrics["N3"]):
        decision = "FORMAT_LIMITED — structured source evidence only"
    else:
        decision = "NO_READING_SIGNAL"
    return {
        "study": STUDY, "status": status, "decision": decision, "metrics": metrics,
        "model": by_arm["N1"]["model"],
        "train_events": by_arm["N1"]["train_events"],
        "traces": rows,
        "real_inferences": 12, "real_training_mcp_calls": 108,
        "S0I_PILOT_or_CONFIRM_used": False,
        "scientific_limit": "Reuses S0-J0 already-consumed TRAIN with only four profiles; exploratory, NOT independent confirmation or learning",
        "no_user_memories_or_production": True,
    }


def main():
    p = argparse.ArgumentParser()
    subs = p.add_subparsers(dest="mode", required=True)
    a = subs.add_parser("arm")
    a.add_argument("--arm", required=True, choices=ARMS)
    a.add_argument("--out", required=True)
    b = subs.add_parser("aggregate")
    b.add_argument("--inputs", required=True)
    b.add_argument("--out", required=True)
    ns = p.parse_args()
    output = Path(ns.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    logging.disable(logging.CRITICAL)
    try:
        if ns.mode == "arm":
            with open(os.devnull, "w", encoding="utf-8") as devnull:
                with redirect_stdout(devnull), redirect_stderr(devnull):
                    result = run(ns.arm)
        else:
            result = aggregate(ns.inputs)
        print(json.dumps({"study": STUDY, "status": result["status"],
                          "arm": result.get("arm"), "decision": result.get("decision"),
                          "metrics": result.get("metrics")}))
        output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        return 0 if result["status"] in ("COMPLETED_ARM", "COMPLETED") else 2
    except Exception as exc:
        frames = traceback.extract_tb(exc.__traceback__)
        problem = {"study": STUDY, "status": "BLOCKED",
                   "type": type(exc).__name__, "reason": str(exc)[:250],
                   "last_frames": [{"function": f.name, "line": f.lineno}
                                   for f in frames[-8:]]}
        output.with_name("blocked.json").write_text(json.dumps(problem) + "\n")
        print(json.dumps(problem))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
