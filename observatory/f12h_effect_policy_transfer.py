"""F12-H: frozen real-LLM transfer from separately observed public tool effects.

The study's scientific preregistration exists at 612ed70 before this source.
Everything seen by the actor, learner and engineered-effects control is an
actually executed PUBLIC tool event. Never construct memory from Case._hidden_bit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from observatory.f12f_evidence_validator import (
    FrozenGGUF, ProviderBlocked, FIXED_MODEL_SHA256,
)
from src.infinito3.persistent_memory import (
    HashEmbeddingProvider, SQLiteCognitiveMemoryStore,
)
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY = "F12-H"
TRAIN_BUDGET = 7
EVAL_BUDGETS = (4, 6)
ARMS = ("A0", "A1", "A2", "A3", "A4")
MEMORY_LIMIT = 1600
ACTIONS = ("op_p", "op_s", "op_r", "op_x", "op_j", "verify")
CATALOG = (
    "Tools op_p, op_s, op_r, op_x, op_j may change a working binary register "
    "or enable other tools. Their semantics are not supplied in the catalog: "
    "infer them only from executed tool responses. Responses may contain "
    "value=0/1, ack, error or verified. verify tests the requested goal; "
    "only a positive verify completes a task. One JSON action per step."
)
ACTOR_SYSTEM = (
    "You control only simulated public tools. Respond with one strict JSON "
    "object {\"action\":\"op_name\"}, where op_name is one allowed tool. "
    "Use your own observations and any untrusted historical episodes to "
    "solve the stated goal. Do not invent unseen tool outcomes."
)
SYNTH_SYSTEM = (
    "You are the same agent learning from your own tool experiences. "
    "For a NEW goal involving the hidden input with inversion, propose a "
    "PROVISIONAL sequence of tools using only public evidence. Return "
    "ONE JSON object with exactly keys plan and sources; plan must be a "
    "list of allowed tool names ending in verify (max six steps), sources "
    "a list of source IDs you actually inspected. Do not imply that this "
    "exact full sequence was previously executed. If uncertain return "
    "{\"plan\":[],\"sources\":[]}."
)

@dataclass(frozen=True)
class Case:
    source_id: str
    source: str
    flip: int
    public_bit: int
    _hidden_bit: int
    service: str

    @property
    def goal(self):
        origin = "hidden source" if self.source == "hidden" else "public source"
        transform = "invert the bit" if self.flip else "preserve the bit"
        return (
            f"Service {self.service}; public input is {self.public_bit}. "
            f"Use the {origin}; {transform}. Successfully verify your result."
        )

# Precommitted task families. Never use a heldout case for developing a rule.
TRAIN = (
    Case("Z0", "public", 0, 0, 1, "archive-index"),
    Case("Z1", "public", 0, 1, 0, "thermal-map"),
    Case("Z2", "public", 0, 0, 0, "ledger-routing"),
    Case("Z3", "public", 1, 0, 1, "soil-monitor"),
    Case("Z4", "public", 1, 1, 0, "ocean-tracker"),
    Case("Z5", "public", 1, 0, 0, "census-export"),
    Case("Z6", "hidden", 0, 1, 0, "marina-alert"),
    Case("Z7", "hidden", 0, 0, 1, "orchid-reporter"),
    Case("Z8", "hidden", 0, 1, 0, "subway-audit"),
)
HELDOUT = (
    Case("Q0", "hidden", 1, 1, 0, "reef-transponder"),
    Case("Q1", "hidden", 1, 0, 1, "observatory-feed"),
    Case("Q2", "hidden", 1, 0, 0, "weather-validator"),
    Case("Q3", "hidden", 1, 1, 1, "habitat-relay"),
    Case("Q4", "hidden", 1, 1, 0, "school-telemetry"),
    Case("Q5", "hidden", 1, 0, 1, "hospital-inventory"),
)

class ToolWorld:
    """Private semantics. Only execute(action) is visible to the actor."""
    def __init__(self, case):
        self._hidden = case._hidden_bit
        self._goal = (case.public_bit if case.source == "public"
                      else case._hidden_bit) ^ case.flip
        self._public = case.public_bit
        self._selected = False
        self._working = None
        self.verified = False

    def execute(self, action):
        if action == "op_p":
            self._working = self._public
            out = {"value": self._working}
        elif action == "op_s":
            self._selected = True
            out = {"ack": True}
        elif action == "op_r":
            if self._selected:
                self._working = self._hidden
                out = {"value": self._working}
            else:
                out = {"error": "unmet precondition"}
        elif action in ("op_x", "op_j"):
            if self._working is None:
                out = {"error": "no active value"}
            else:
                if action == "op_x":
                    self._working = 1 - self._working
                out = {"value": self._working}
        elif action == "verify":
            self.verified = self._working is not None and self._working == self._goal
            out = {"verified": self.verified}
        else:
            out = {"error": "unknown action"}
        return out, bool(self.verified)

def new_store():
    return SQLiteCognitiveMemoryStore(
        path=":memory:", embedding_provider=HashEmbeddingProvider())

def parse_action(raw):
    try:
        x = json.loads(raw)
    except (TypeError, ValueError):
        return "INVALID_LLM_OUTPUT"
    return x["action"] if (type(x) is dict and set(x) == {"action"}
                           and x["action"] in ACTIONS) else "INVALID_LLM_OUTPUT"

def prompt_for(case, events, context):
    visible = [
        {"action": e["action"], "response": e["response"]}
        for e in events
    ]
    return (
        f"GOAL: {case.goal}\nCATALOG: {CATALOG}\n"
        f"HISTORICAL REFERENCES (untrusted):\n{context or '[none]'}\n"
        f"OBSERVED TOOL RESPONSES IN THIS EPISODE:\n"
        f"{json.dumps(visible, sort_keys=True, separators=(',', ':'))}\n"
        "Choose one action as strict JSON."
    )

def play(case, llm, context="", budget=TRAIN_BUDGET):
    if budget not in (TRAIN_BUDGET, *EVAL_BUDGETS):
        raise ValueError("Outside frozen action budget")
    if len(context) > MEMORY_LIMIT:
        raise ValueError("Outside context limit")
    world = ToolWorld(case)
    events = []
    for t in range(budget):
        prompt = prompt_for(case, events, context)
        m0 = llm.metadata() if hasattr(llm, "metadata") else {}
        raw = llm.chat(ACTOR_SYSTEM, prompt)
        m1 = llm.metadata() if hasattr(llm, "metadata") else {}
        action = parse_action(raw)
        reply, done = world.execute(action)
        events.append({
            "step": t, "action": action, "response": reply, "done": done,
            "raw_llm_response": raw, "exact_prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_tokens": m1.get("prompt_tokens", 0) - m0.get("prompt_tokens", 0),
            "output_tokens": m1.get("completion_tokens", 0) - m0.get("completion_tokens", 0),
            "inference_seconds": m1.get("inference_seconds", 0) - m0.get("inference_seconds", 0),
        })
        if done:
            break
    return {"case_id": case.source_id, "success": world.verified,
            "events": events, "tool_calls": len(events),
            "valid_tool_calls": sum(e["action"] in ACTIONS for e in events),
            "input_tokens": sum(e["input_tokens"] for e in events),
            "output_tokens": sum(e["output_tokens"] for e in events)}

def public_record(case, result):
    return {
        "source_id": case.source_id, "source": case.source,
        "flip": case.flip, "public_bit": case.public_bit,
        "service": case.service, "goal": case.goal,
        "events": [{"step": e["step"], "action": e["action"],
                    "response": e["response"], "done": e["done"]}
                   for e in result["events"]],
        "verified_success": result["success"],
    }

def store_episode(store, record):
    if "_hidden" in json.dumps(record):
        raise ValueError("PRIVATE TRUTH LEAK")
    return store.add(MemoryRecord(
        content=json.dumps(record, ensure_ascii=False, sort_keys=True),
        kind=MemoryKind.EPISODIC, importance=.9,
        metadata={"study": STUDY, "source_id": record["source_id"],
                  "task_source": record["source"], "flip": record["flip"],
                  "observed_only": True}))

def sources_from_memory(store):
    result = {}
    for rec in store.all():
        if rec.kind == MemoryKind.EPISODIC:
            obj = json.loads(rec.content)
            if set(obj) != {"source_id", "source", "flip", "public_bit",
                            "service", "goal", "events", "verified_success"}:
                raise ValueError("Unknown/private fields in stored experience")
            result[obj["source_id"]] = obj
    return result

def compact_episode(source):
    # No expert rewriting: retain every actually observed response in sequence.
    ops = ", ".join(
        e["action"] + "=" +
        json.dumps(e["response"], sort_keys=True, separators=(",", ":"))
        for e in source["events"])
    return (f"[{source['source_id']} {source['source']} flip={source['flip']} "
            f"public={source['public_bit']} success={source['verified_success']}] {ops}")

def episodic_context(store):
    # Genuine INFINITO search is required. Holdout is hidden+flip, so include
    # separately learned hidden-copy and public-flip examples, not a lookup
    # on an already observed hidden-flip combination.
    matches = store.search(
        "hidden read source select public invert transformations episodes",
        top_k=24)
    available = [json.loads(r.content) for r in matches
                 if r.kind == MemoryKind.EPISODIC]
    selected = []
    for source, flip in (("hidden", 0), ("public", 1), ("public", 0)):
        selected.extend(sorted(
            (r for r in available if r["source"] == source and r["flip"] == flip),
            key=lambda r: r["source_id"])[:2])
    rendered = "\n".join(compact_episode(x) for x in selected)
    return rendered[:MEMORY_LIMIT], [x["source_id"] for x in selected]

def discover_effect_evidence(sources):
    """F10-P style counts ONLY from actual tool response transitions."""
    x_seen = set()
    sr_sources = set()
    x_sources = set()
    for sid, source in sources.items():
        last_value = None
        selected = False
        for event in source["events"]:
            a, response = event["action"], event["response"]
            if a == "op_s" and response == {"ack": True}:
                selected = True
            if a == "op_r" and selected and set(response) == {"value"}:
                sr_sources.add(sid)
            if a == "op_x" and set(response) == {"value"} and last_value in (0, 1):
                if response["value"] == 1 - last_value:
                    x_seen.add(last_value)
                    x_sources.add(sid)
            if "value" in response:
                last_value = response["value"]
            elif "error" in response:
                last_value = None
    return {
        "x_input_values_seen": sorted(x_seen),
        "x_source_ids": sorted(x_sources),
        "sr_source_ids": sorted(sr_sources),
    }

def has_train_composition(source):
    order = ("op_s", "op_r", "op_x", "verify")
    index = 0
    for e in source["events"]:
        if e["action"] == order[index]:
            index += 1
            if index == len(order):
                return True
    return False

def parse_plan(raw, known_ids, *, allow_no_sources=False):
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        return {"plan": [], "sources": [], "parse_status": "INVALID_JSON"}
    if (type(parsed) is not dict or set(parsed) != {"plan", "sources"}
            or type(parsed["plan"]) is not list or type(parsed["sources"]) is not list):
        return {"plan": [], "sources": [], "parse_status": "INVALID_SCHEMA"}
    p, s = parsed["plan"], parsed["sources"]
    if (len(p) > 6 or any(type(a) is not str or a not in ACTIONS for a in p)
            or (p and p[-1] != "verify")
            or any(type(a) is not str or a not in known_ids for a in s)
            or (not allow_no_sources and not s and p)):
        return {"plan": [], "sources": [], "parse_status": "INVALID_PLAN_OR_SOURCES"}
    return {"plan": p, "sources": s, "parse_status": "OK"}

def propose_policy(llm, observed_sources, *, with_experiences):
    references = (
        "\n".join(compact_episode(x) for x in observed_sources.values())
        if with_experiences else "[no episodes available]"
    )
    prompt = (
        "Target task: use the hidden source, invert its bit, then verify. "
        f"Available tools: {', '.join(ACTIONS)}. {CATALOG}\n"
        f"EXPERIENCES:\n{references}\n"
        "Return one JSON provisional plan and exact source IDs."
    )
    raw = llm.chat(SYNTH_SYSTEM, prompt, max_tokens=192)
    parsed = parse_plan(raw, set(observed_sources) if with_experiences else set(),
                        allow_no_sources=not with_experiences)
    return {**parsed, "raw_llm_response": raw, "exact_prompt": prompt,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}

def engineered_effect_plan(evidence):
    # Researcher-coded planner: explicit strong comparator, NOT an LLM learned
    # policy. It has no Case access or private simulator state.
    if (evidence["x_input_values_seen"] == [0, 1]
            and len(evidence["sr_source_ids"]) >= 2):
        return {"plan": ["op_s", "op_r", "op_x", "verify"],
                "sources": sorted(set(evidence["x_source_ids"] +
                                       evidence["sr_source_ids"])),
                "parse_status": "SUPPORTED_OBSERVED_EFFECT_COMPOSITION"}
    return {"plan": [], "sources": [], "parse_status": "INSUFFICIENT_EVIDENCE"}

def plan_context(plan, tag, episode_text):
    if not plan["plan"]:
        return episode_text
    proposed = (f"{tag} (provisional; never executed end-to-end in TRAIN): "
                + json.dumps({"plan": plan["plan"], "sources": plan["sources"]},
                             separators=(",", ":")) + "\n")
    return (proposed + episode_text[:max(0, MEMORY_LIMIT - len(proposed))])[:MEMORY_LIMIT]

def frozen_snapshot(store):
    return sorted((x.id, x.content, x.kind.value)
                  for x in store.all())

def scores_for(outcomes, arm, horizon):
    xs = [x for x in outcomes if x["arm"] == arm and x["budget"] == horizon]
    return {
        "episodes": len(xs), "successes": sum(x["success"] for x in xs),
        "tool_calls": sum(x["tool_calls"] for x in xs),
        "valid_tool_calls": sum(x["valid_tool_calls"] for x in xs),
        "input_tokens": sum(x["input_tokens"] for x in xs),
        "output_tokens": sum(x["output_tokens"] for x in xs),
    }

def run(llm, out=None):
    if out:
        out.mkdir(parents=True, exist_ok=True)
    episodic, semantic = new_store(), new_store()
    try:
        observed_train = []
        for c in TRAIN:
            played = play(c, llm, budget=TRAIN_BUDGET)
            record = public_record(c, played)
            store_episode(episodic, record)
            observed_train.append(played)
        sources = sources_from_memory(episodic)
        evidence = discover_effect_evidence(sources)
        policy = propose_policy(llm, sources, with_experiences=True)
        control_policy = propose_policy(llm, sources, with_experiences=False)
        planner = engineered_effect_plan(evidence)
        if policy["plan"]:
            semantic.add(MemoryRecord(
                content=json.dumps({"plan": policy["plan"], "sources": policy["sources"]},
                                   sort_keys=True),
                kind=MemoryKind.SEMANTIC, importance=.5,
                metadata={"study": STUDY, "status": "unverified_hypothesis"}))
        saved = frozen_snapshot(episodic), frozen_snapshot(semantic)
        epis_context, epis_ids = episodic_context(episodic)
        all_contexts = {
            "A0": "",
            "A1": epis_context,
            "A2": plan_context(policy, "EXPERIENCE-INFERRED PLAN", epis_context),
            "A3": plan_context(control_policy, "NO-EVIDENCE PLAN", epis_context),
            "A4": plan_context(planner, "ENGINEERED-EFFECT PLAN", epis_context),
        }
        if any(len(x) > MEMORY_LIMIT for x in all_contexts.values()):
            raise AssertionError("Context over budget")
        outcomes = []
        for case in HELDOUT:
            for budget in EVAL_BUDGETS:
                offset = (int(case.source_id[1:]) + budget) % len(ARMS)
                rotated = ARMS[offset:] + ARMS[:offset]
                for arm in rotated:
                    result = play(case, llm, all_contexts[arm], budget=budget)
                    result.update({
                        "arm": arm, "budget": budget,
                        "context_text": all_contexts[arm],
                        "context_sha256": hashlib.sha256(
                            all_contexts[arm].encode()).hexdigest(),
                    })
                    outcomes.append(result)
        frozen = saved == (frozen_snapshot(episodic), frozen_snapshot(semantic))
        score = {str(b): {arm: scores_for(outcomes, arm, b)
                          for arm in ARMS} for b in EVAL_BUDGETS}
        tight, slack = score["4"], score["6"]
        cited = set(policy["sources"])
        supported_x = bool(cited & set(evidence["x_source_ids"]))
        supported_sr = bool(cited & set(evidence["sr_source_ids"]))
        model = llm.metadata() if hasattr(llm, "metadata") else {"real_model": False}
        gates = {
            "P0_integrity_9_train_60_eval_frozen_real_model": (
                getattr(llm, "real_model", False)
                and model.get("weights_sha256") == FIXED_MODEL_SHA256
                and len(observed_train) == 9 and len(outcomes) == 60
                and all(s["episodes"] == 6 for by_budget in score.values()
                        for s in by_budget.values())
                and frozen),
            "P1_sufficient_public_training_evidence": (
                sum(x["success"] for x in observed_train) >= 5
                and evidence["x_input_values_seen"] == [0, 1]
                and len(evidence["sr_source_ids"]) >= 2),
            "P2_no_train_hidden_invert_composition": (
                not any(has_train_composition(x) for x in sources.values())),
            "P3_policy_beats_episodic_and_no_evidence": (
                tight["A2"]["successes"] >= tight["A1"]["successes"] + 2
                and slack["A2"]["successes"] >= slack["A1"]["successes"]
                and all(score[str(b)]["A2"]["successes"] >=
                        score[str(b)]["A3"]["successes"] for b in EVAL_BUDGETS)),
            "P4_specific_observed_provenance_not_same_as_control": (
                bool(policy["plan"]) and supported_x and supported_sr
                and policy["plan"] != control_policy["plan"]
                and tight["A2"]["successes"] >= tight["A3"]["successes"] + 2),
            "P5_engineered_effect_planner_diagnostic": bool(planner["plan"]),
            "P6_evaluation_input_cost_within_25pct": (
                sum(score[str(b)]["A2"]["input_tokens"] for b in EVAL_BUDGETS)
                <= 1.25 * max(1, sum(score[str(b)]["A1"]["input_tokens"]
                                      for b in EVAL_BUDGETS))),
        }
        status = ("PILOT_GO_TO_PREREG_CONFIRM" if all(gates.values())
                  else "PILOT_NO_GO")
        if not getattr(llm, "real_model", False):
            status = "FAKE_TEST_ONLY"
        elif not gates["P0_integrity_9_train_60_eval_frozen_real_model"]:
            status = "INVALID_OR_INCOMPLETE"
        elif not gates["P1_sufficient_public_training_evidence"]:
            status = "INCONCLUSIVE_INSUFFICIENT_TRAIN_EVIDENCE"
        elif not gates["P2_no_train_hidden_invert_composition"]:
            status = "INVALID_TRAIN_COMPOSITION_LEAK"
        result = {
            "study": STUDY, "status": status, "model": model,
            "train": observed_train, "train_sources": sources,
            "policy": policy, "no_experience_policy": control_policy,
            "engineered_effect_planner": planner, "public_effect_evidence": evidence,
            "episodic_source_ids": epis_ids, "contexts": all_contexts,
            "outcomes": outcomes, "scores": score, "gates": gates,
            "memory_frozen": frozen, "scientific_pass": False,
            "dev_opened": False, "confirm_opened": False,
        }
        if out:
            (out / "f12h_full_report.json").write_text(
                json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8")
            (out / "f12h_summary.json").write_text(
                json.dumps({k: v for k, v in result.items()
                            if k not in ("train", "train_sources", "outcomes", "contexts")},
                           ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8")
        return result
    finally:
        episodic.close()
        semantic.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("f12h_artifacts"))
    args = parser.parse_args()
    llm = FrozenGGUF()
    result = run(llm, args.out)
    print(json.dumps({
        "study": result["study"], "status": result["status"],
        "scores": result["scores"], "gates": result["gates"],
        "model": result["model"], "policy": result["policy"]["plan"],
        "effect_evidence": result["public_effect_evidence"],
    }, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
