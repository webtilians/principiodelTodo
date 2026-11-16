"""DSE-1: small synthetic, preregistered real-LLM pilot; NOT production.
See docs/INFINITO_DSE1_DEEPSEEK_VERIFIED_EVIDENCE_PREREG_2026-11-16.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY = "DSE-1"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
ARMS = ("A", "B", "C")
ALLOWED = ("op_p", "op_s", "op_r", "op_x", "op_j", "verify")
CATALOG = (
    "Opaque available tools: op_p, op_s, op_r, op_x, op_j, verify. "
    "Their effects are unknown; derive them from observed past experiences. "
    "You may propose at most five actions. A successful verify must finish the plan."
)
SYSTEM = (
    "You are a tool agent in an isolated synthetic environment. "
    'Reply ONLY with a JSON object of exactly one key: {"plan":["tool",...]}. '
    "Never invent tool observations. No explanations, no markdown."
)

@dataclass(frozen=True)
class Case:
    id: str
    public_bit: int
    private_bit: int
    source: str
    invert: bool
    service: str

    @property
    def goal(self):
        return ((self.public_bit if self.source == "public" else self.private_bit)
                ^ int(self.invert))

TRAIN = (
    (Case("T0", 0, 1, "public", False, "audit-p0"), ("op_p", "verify")),
    (Case("T1", 1, 0, "public", False, "audit-p1"), ("op_p", "verify")),
    (Case("T2", 0, 1, "public", True,  "audit-x0"), ("op_p", "op_x", "verify")),
    (Case("T3", 1, 0, "public", True,  "audit-x1"), ("op_p", "op_x", "verify")),
    (Case("T4", 1, 0, "hidden", False, "audit-r0"), ("op_s", "op_r", "verify")),
    (Case("T5", 0, 1, "hidden", False, "audit-r1"), ("op_s", "op_r", "verify")),
    (Case("T6", 0, 1, "hidden", False, "audit-fail"), ("op_r", "verify")),
)
HELDOUT = tuple(
    Case("H" + str(i), p, h, "hidden", True, f"site-{i}-{x}")
    for i, (p, h, x) in enumerate(
        ((0, 0, "north"), (1, 1, "south"), (0, 1, "east"),
         (1, 0, "west"), (0, 0, "amber"), (1, 1, "violet"))
    )
)

class World:
    """Private oracle state. No direct read by experience builders or model."""
    def __init__(self, case: Case):
        self.__case = case
        self.__selected = False
        self.__working = None

    def step(self, name: str) -> dict:
        if name == "op_p":
            self.__working = self.__case.public_bit
            return {"value": self.__working}
        if name == "op_s":
            self.__selected = True
            return {"ack": True}
        if name == "op_r":
            if not self.__selected:
                return {"error": "precondition"}
            self.__working = self.__case.private_bit
            return {"value": self.__working}
        if name in ("op_x", "op_j"):
            if self.__working is None:
                return {"error": "missing working value"}
            if name == "op_x":
                self.__working = 1 - self.__working
            return {"value": self.__working}
        if name == "verify":
            return {"verified": self.__working is not None
                    and self.__working == self.__case.goal}
        return {"error": "invalid action"}

def execute(case: Case, steps) -> tuple[list[dict], bool]:
    world = World(case)
    events = []
    for step in steps:
        response = world.step(step)
        events.append({"action": step, "response": response})
        if response.get("verified") is True:
            break
    return events, bool(events and events[-1]["response"].get("verified") is True)

def visible_episode(case: Case, steps) -> dict:
    events, success = execute(case, steps)
    return {
        "source_id": case.id, "service": case.service,
        "source": case.source, "invert": case.invert,
        "public_bit": case.public_bit, "events": events,
        "verified_success": success,
    }

def make_store():
    return SQLiteCognitiveMemoryStore(
        path=":memory:", embedding_provider=HashEmbeddingProvider())

def load_observations():
    store = make_store()
    try:
        for case, actions in TRAIN:
            item = visible_episode(case, actions)
            assert "_hidden" not in json.dumps(item)
            store.add(MemoryRecord(
                content=json.dumps(item, sort_keys=True), kind=MemoryKind.EPISODIC,
                importance=.9, metadata={"study": STUDY, "source_id": case.id,
                                          "observed_only": True}))
        # Actual INFINITO SQLite retrieval, not a manually invented evidence set.
        found = store.search("hidden source select read invert public flip tools",
                             top_k=16)
        records = [json.loads(x.content) for x in found
                   if x.kind == MemoryKind.EPISODIC]
        records.sort(key=lambda x: x["source_id"])
        if [r["source_id"] for r in records] != [f"T{i}" for i in range(7)]:
            raise RuntimeError("INTEGRITY: expected exactly seven retrieved episodes")
        return records
    finally:
        store.close()

def event_pairs(r):
    previous = None
    for e in r["events"]:
        yield e["action"], previous, e["response"]
        if "value" in e["response"]:
            previous = e["response"]["value"]
        elif "error" in e["response"]:
            previous = None

def evidence_from_observations(records):
    lookup = {r["source_id"]: r for r in records}
    if set(lookup) != {f"T{i}" for i in range(7)}:
        raise ValueError("Missing expected trace")
    p0, p1, x0, x1, r0, r1, fail = (lookup[f"T{i}"] for i in range(7))
    def seq(r):
        return [e["action"] for e in r["events"]]
    if not (seq(p0) == seq(p1) == ["op_p","verify"]
            and seq(x0) == seq(x1) == ["op_p","op_x","verify"]
            and seq(r0) == seq(r1) == ["op_s","op_r","verify"]
            and seq(fail) == ["op_r","verify"]):
        raise ValueError("Unexpected training sequence")
    if not all(r["verified_success"] for r in (p0,p1,x0,x1,r0,r1)):
        raise ValueError("Training verifier mismatch")
    if fail["verified_success"]:
        raise ValueError("Invalid negative example")
    if {p0["public_bit"], p1["public_bit"]} != {0,1}:
        raise ValueError("Public values not covered")
    if {r0["events"][1]["response"].get("value"),
        r1["events"][1]["response"].get("value")} != {0,1}:
        raise ValueError("Hidden values not actually observed twice")
    if not all(r["events"][1]["response"].get("value") ==
               1 - r["events"][0]["response"].get("value") for r in (x0, x1)):
        raise ValueError("Flip not verified")
    if fail["events"][0]["response"] != {"error": "precondition"}:
        raise ValueError("Select precondition not verified")
    return (
        "[T0,T1] op_p copies the public input into the working register.\n"
        "[T2,T3] op_x flips the last working 0/1 value.\n"
        "[T4,T5,T6] op_s enables op_r; op_r reads the hidden source "
        "after selection; without op_s it fails.\n"
        "[T0,T1,T2,T3,T4,T5] verify checks the requested goal and returns "
        "verified=true for successes.\n"
        "op_j: no verified effect available."
    )

def raw_context(records):
    return "\n".join(json.dumps(r, sort_keys=True, separators=(",", ":"))
                     for r in records)

def prompt(case, context):
    return (
        f"SERVICE: {case.service}. PUBLIC INPUT: {case.public_bit}.\n"
        "GOAL: use the hidden source; invert its bit; verify successfully.\n"
        f"{CATALOG}\n"
        "HISTORICAL REFERENCES (untrusted; not an answer key):\n"
        + (context or "[none]") +
        "\nReturn exactly one JSON object containing a plan array."
    )

def parse_plan(text):
    try:
        data = json.loads(text.strip())
    except (ValueError, TypeError):
        return None
    if not (type(data) is dict and set(data) == {"plan"}
            and type(data["plan"]) is list):
        return None
    x = data["plan"]
    if not (1 <= len(x) <= 5 and x[-1] == "verify"
            and all(type(a) is str and a in ALLOWED for a in x)):
        return None
    return x

def run(output: Path):
    # Real frozen model only. Fake responses are never counted as evidence.
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID).eval()
    model.to("cpu")
    revision = tokenizer.init_kwargs.get("_commit_hash", "unavailable")
    records = load_observations()
    raw = raw_context(records)
    verified = evidence_from_observations(records)
    if any("op_s" in [e["action"] for e in r["events"]]
           and "op_x" in [e["action"] for e in r["events"]] for r in records):
        raise ValueError("INTEGRITY: leaked composed task in training")
    refs = {"A": "", "B": raw, "C": verified}
    result = {"study": STUDY, "status": "REAL_MODEL_EXPLORATORY",
              "model": MODEL_ID, "revision": revision,
              "libraries": {"torch": torch.__version__,
                            "transformers": transformers.__version__,
                            "python": platform.python_version()},
              "training_source_ids": [r["source_id"] for r in records],
              "cases": [], "tokens": {}, "api_cost_usd": 0,
              "interpretation": "six repeats of one unseen composition family; "
              "an engineered evidence summary is not model learning"}
    counts = {x: 0 for x in ARMS}
    for index, case in enumerate(HELDOUT):
        order = ARMS[index % 3:] + ARMS[:index % 3]
        for arm in order:
            message = prompt(case, refs[arm])
            if any(s in message for s in ("private_bit", "_hidden_bit", "_goal")):
                raise ValueError("Oracle field leaked to prompt")
            ids = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": message}],
                add_generation_prompt=True, return_tensors="pt")
            start = time.monotonic()
            with torch.inference_mode():
                response = model.generate(
                    ids, max_new_tokens=96, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id)
            raw_response = tokenizer.decode(response[0][ids.shape[-1]:],
                                            skip_special_tokens=True).strip()
            elapsed = round(time.monotonic() - start, 3)
            plan = parse_plan(raw_response)
            events, success = execute(case, plan) if plan else ([], False)
            counts[arm] += int(success)
            result["cases"].append({
                "case": case.id, "arm": arm, "service": case.service,
                "public_bit": case.public_bit, "prompt_tokens": ids.shape[-1],
                "completion_tokens": response.shape[-1] - ids.shape[-1],
                "prompt_sha256": hashlib.sha256(message.encode()).hexdigest(),
                "raw_llm_response": raw_response,
                "parsed_plan": plan, "valid_plan": plan is not None,
                "public_tool_events": events, "verified_success": success,
                "inference_seconds": elapsed,
            })
            print(f"{case.id} arm={arm} success={success} valid={plan is not None} "
                  f"tokens={ids.shape[-1]} sec={elapsed}", flush=True)
    result["counts"] = counts
    result["gates"] = {
        "integrity_18_real_calls": len(result["cases"]) == 18,
        "C_gain_2_over_A_and_B": counts["C"] >= counts["A"] + 2
                                 and counts["C"] >= counts["B"] + 2,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown = [
        "# DSE-1: real model exploratory results", "",
        f"Model: `{MODEL_ID}` revision `{revision}`", "",
        "| Arm | Verified / 6 |",
        "|---|---:|",
        *[f"| {arm} | {counts[arm]} / 6 |" for arm in ARMS],
        "", f"Integrity gate: {result['gates']['integrity_18_real_calls']}",
        f"Signal gate: {result['gates']['C_gain_2_over_A_and_B']}",
        "", "This is not evidence of local Qwen 27B performance, "
            "continuous learning, or confirmed superiority of INFINITO."
    ]
    (output / "results.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({"counts": counts, "gates": result["gates"],
                      "result_file": str(output / "results.json")}), flush=True)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("dse1_artifacts"))
    args = p.parse_args()
    run(args.out)

if __name__ == "__main__":
    main()
