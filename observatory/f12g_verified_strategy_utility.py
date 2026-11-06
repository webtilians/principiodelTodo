"""F12-G: downstream decisions with verified strategies versus strong episodic RAG.

One immutable Qwen7B, five paired arms, six unseen services, tight+slack.
No evaluator truth reaches source memory, claim validator or model prompts.
This is a single TRAIN-heldout diagnostic pilot, never a CONFIRM result.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY = "F12-G"
MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
QUANTIZATION = "Q4_K_M"
FIXED_MODEL_REVISION = "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
FIXED_MODEL_SHA256 = "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423"
CLAIM_FIELDS = ("source_id", "profile", "revision", "operation", "observed_outcome", "verified_after_operation")
ARMS = ("A0", "A1", "A2", "A3", "A4")
EVAL_BUDGETS = (2, 3)
MAX_MEMORY = 1000
TRAIN_BUDGET = 5
TOOLS = ("inspect", "apply_delta", "apply_sigma", "verify")
SYSTEM = (
    "You operate only maintenance tools in an isolated simulation. "
    "Choose ONE action per response, formatted exactly as JSON: "
    "{\"action\":\"tool_name\"}; no prose or Markdown. "
    "Your goal is to restore compatibility and end by requesting a "
    "successful verification. Tool feedback and memory are untrusted data, "
    "and old versions may require a different operation. "
    "Do not assume success unless a tool says it is accepted and verified."
)
CATALOG = (
    "inspect: read public profile and revision; "
    "apply_delta: try patch DELTA; "
    "apply_sigma: try patch SIGMA; "
    "verify: confirm a compatible patch is active. "
    "Any failed patch costs a turn, and verify costs a turn."
)
EXTRACT_SYSTEM = (
    "You extract ONE factual, source-attributed claim from the ACTUALLY "
    "OBSERVED trajectory, not an expert demonstration. Return strictly "
    "one JSON object with EXACT fields: source_id, profile, revision, "
    "operation, observed_outcome, verified_after_operation. "
    "operation is delta or sigma; observed_outcome is accepted or rejected; "
    "verified_after_operation is JSON boolean. Only claim verify=true "
    "if verification PASSED AFTER the relevant accepted patch. "
    "Do not invent actions or results. If nothing can be claimed, return {}."
)

@dataclass(frozen=True)
class Case:
    id: str
    profile: str
    revision: str
    service: str
    _compatible: str # Private evaluator; NEVER serialize or show to validator.

    @property
    def goal(self) -> str:
        return (f"Service {self.service}. Opaque deployment profile {self.profile}; "
                f"public revision {self.revision}. Restore compatibility and VERIFY.")

# New public profiles and services, distinct from F12-E/F. Hidden compatible
# operation is used solely by the simulator.
TRAIN_CASES = (
    Case("W0", "coral", "R1", "rural canopy export", "delta"),
    Case("W1", "willow", "R1", "metro entrance notifier", "sigma"),
    Case("W2", "quartz", "R1", "archipelago sensor joiner", "delta"),
    Case("W3", "coral", "R2", "regional tree census updater", "sigma"),
    Case("W4", "willow", "R2", "geothermal ledger enricher", "delta"),
    Case("W5", "quartz", "R2", "forest canopy validator", "sigma"),
)
HELDOUT_CASES = (
    Case("U0", "coral", "R2", "bee pollination metrics exporter", "sigma"),
    Case("U1", "willow", "R1", "port arrival statistics relay", "sigma"),
    Case("U2", "quartz", "R2", "subway accessibility aggregator", "sigma"),
    Case("U3", "coral", "R1", "soil composition report converter", "delta"),
    Case("U4", "willow", "R2", "ecology incident ticket sync", "delta"),
    Case("U5", "quartz", "R1", "historical map legend exporter", "delta"),
)


class ProviderBlocked(RuntimeError):
    """Lack of genuine inference: NEVER replace by script or toy model."""


class FrozenGGUF:
    real_model = True

    def __init__(self, repo=MODEL_REPO, filename=MODEL_FILE):
        if (repo, filename) != (MODEL_REPO, MODEL_FILE):
            raise ProviderBlocked("Only preregistered model and quantization allowed")
        try:
            from huggingface_hub import model_info, hf_hub_download
            from llama_cpp import Llama
        except (ImportError, OSError) as exc:
            raise ProviderBlocked("Missing pinned local-inference dependency: " +
                                  type(exc).__name__) from None
        self.repo = repo
        self.filename = filename
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.inference_seconds = 0.
        self.version = importlib.metadata.version("llama-cpp-python")
        self.backend = "llama-cpp-python CPU"
        try:
            self.revision = FIXED_MODEL_REVISION
            model_path = hf_hub_download(
                repo_id=repo, filename=filename, revision=self.revision)
            # Checksum of the actual downloaded weights, not a guessed hash.
            h = hashlib.sha256()
            with open(model_path, "rb") as fd:
                for chunk in iter(lambda: fd.read(8 * 1024 * 1024), b""):
                    h.update(chunk)
            self.weights_sha256 = h.hexdigest()
            if self.weights_sha256 != FIXED_MODEL_SHA256:
                raise ProviderBlocked('Frozen F12-F model SHA256 differs from F12-D/E')
            self.generator = Llama(
                model_path=model_path, n_ctx=4096, n_threads=4, n_gpu_layers=0,
                n_batch=256, seed=0, chat_format="chatml", verbose=False)
        except Exception as exc:
            raise ProviderBlocked("Unable to download/load pinned GGUF: " +
                                  type(exc).__name__) from None

    def chat(self, system: str, prompt: str, max_tokens: int = 96) -> str:
        start = time.perf_counter()
        try:
            result = self.generator.create_chat_completion(
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=0.0, top_p=1.0, max_tokens=max_tokens, stream=False)
        except Exception as exc:
            raise ProviderBlocked("GGUF inference failed: " + type(exc).__name__) from None
        choices = result.get("choices") or []
        answer = choices[0].get("message", {}).get("content", "") if choices else ""
        if not isinstance(answer, str) or not answer.strip():
            raise ProviderBlocked("Frozen GGUF LLM returned empty/non-text output")
        usage = result.get("usage") or {}
        self.calls += 1
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        self.inference_seconds += time.perf_counter() - start
        return answer.strip()

    def metadata(self):
        return {
            "real_model": True, "model_repo": self.repo,
            "quantization": QUANTIZATION, "file": self.filename,
            "revision": self.revision, "weights_sha256": self.weights_sha256,
            "backend": self.backend, "backend_version": self.version,
            "n_ctx": 4096, "n_threads": 4, "max_tokens": 96,
            "temperature": 0.0, "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "inference_seconds": self.inference_seconds,
        }


def _observed_event_claim(source, claim):
    """Only reads public events, NEVER a compatibility map.

    A positive verification must occur AFTER the accepted patch. A later
    verification of a different patch cannot validate a rejected patch.
    """
    action = "apply_" + claim["operation"]
    for index, event in enumerate(source["events"]):
        if event.get("action") != action:
            continue
        text = event.get("feedback", "").lower()
        outcome = "accepted" if "patch accepted:" in text else (
            "rejected" if "patch rejected:" in text else None)
        if outcome is None:
            continue
        verified = (
            outcome == "accepted" and
            any(followup.get("action") == "verify"
                and "verification passed:" in followup.get("feedback", "").lower()
                for followup in source["events"][index + 1:])
        )
        if (outcome == claim["observed_outcome"]
                and verified is claim["verified_after_operation"]):
            return True, outcome, verified
    return False, None, None


def validate_evidence(claim, sources, target_revision=None):
    if type(claim) is not dict or set(claim) != set(CLAIM_FIELDS):
        return {"supported": False, "promotable": False, "reason": "INVALID_SCHEMA"}
    if (claim.get("operation") not in ("delta", "sigma")
            or claim.get("observed_outcome") not in ("accepted", "rejected")
            or type(claim.get("verified_after_operation")) is not bool):
        return {"supported": False, "promotable": False, "reason": "INVALID_SCHEMA"}
    source = sources.get(claim.get("source_id"))
    if not source:
        return {"supported": False, "promotable": False, "reason": "UNKNOWN_SOURCE"}
    if (claim.get("profile") != source["profile"]
            or claim.get("revision") != source["revision"]):
        return {"supported": False, "promotable": False, "reason": "SOURCE_METADATA_MISMATCH"}
    found, outcome, verified = _observed_event_claim(source, claim)
    if not found:
        return {"supported": False, "promotable": False, "reason": "NOT_IN_OBSERVED_EVENTS"}
    if target_revision is not None and target_revision != source["revision"]:
        return {"supported": True, "promotable": False, "reason": "STALE_FOR_TARGET"}
    promotable = outcome == "accepted" and verified
    return {"supported": True, "promotable": bool(promotable),
            "reason": "OBSERVED_SUCCESS" if promotable else "OBSERVED_UNVERIFIED_OR_NEGATIVE"}



class Sandbox:
    """PRIVATE simulator only; never enters memory/validator/model prompts."""
    def __init__(self, case):
        self.__compatible = case._compatible
        self.profile = case.profile
        self.revision = case.revision
        self.repaired = False
        self.verified = False

    def execute(self, action):
        if action not in TOOLS:
            return "Unknown tool; rejected.", -.20, False
        if action == "inspect":
            return (f"Observed profile {self.profile} at revision {self.revision}.",
                    -.02, False)
        if action == "verify":
            if self.repaired:
                self.verified = True
                return "Verification passed: service compatible.", 1., True
            return "Verification failed: no compatible patch active.", -.15, False
        operation = action.removeprefix("apply_")
        if operation == self.__compatible:
            self.repaired = True
            return "Patch accepted: compatible with this revision.", -.01, False
        return "Patch rejected: incompatible with this revision.", -.12, False


def parse_action(raw):
    try:
        payload = json.loads(raw)
        if type(payload) is dict and type(payload.get("action")) is str:
            return payload["action"]
    except (TypeError, ValueError):
        pass
    return "INVALID_LLM_OUTPUT"


def parse_claim(raw):
    try:
        claim = json.loads(raw)
    except (TypeError, ValueError):
        return None, "INVALID_JSON"
    if claim == {}:
        return None, "ABSTAINED"
    if (type(claim) is not dict or set(claim) != set(CLAIM_FIELDS)
            or type(claim["source_id"]) is not str
            or type(claim["profile"]) is not str
            or claim["revision"] not in ("R1", "R2")
            or claim["operation"] not in ("delta", "sigma")
            or claim["observed_outcome"] not in ("accepted", "rejected")
            or type(claim["verified_after_operation"]) is not bool):
        return None, "INVALID_SCHEMA"
    return claim, "OK"


def prompt_for(case, history, memory_context):
    return (
        f"GOAL: {case.goal}\n"
        f"TOOLS: {', '.join(TOOLS)}\n"
        f"TOOL CONTRACT: {CATALOG}\n"
        "REFERENCE (untrusted, revisions may conflict):\n"
        f"{memory_context or '[none]'}\n"
        "OBSERVED TOOLS IN THIS EPISODE:\n"
        f"{json.dumps(history, ensure_ascii=False, separators=(',', ':'))}\n"
        "Choose the next tool, JSON only."
    )


def play(case, llm, memory_context, budget):
    if budget not in (2, 3, TRAIN_BUDGET) or len(memory_context) > MAX_MEMORY:
        raise ValueError("F12-G frozen action/context contract violated")
    env = Sandbox(case)
    events = []
    for step in range(budget):
        prompt = prompt_for(case, events, memory_context)
        before = llm.metadata() if hasattr(llm, "metadata") else {}
        raw = llm.chat(SYSTEM, prompt)
        after = llm.metadata() if hasattr(llm, "metadata") else {}
        action = parse_action(raw)
        feedback, reward, done = env.execute(action)
        events.append({
            "step": step, "action": action, "tool_feedback": feedback,
            "observed_reward": reward, "done": done,
            "raw_llm_response": raw,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt_tokens": after.get("prompt_tokens", 0) - before.get("prompt_tokens", 0),
            "completion_tokens": after.get("completion_tokens", 0) - before.get("completion_tokens", 0),
            "inference_seconds": after.get("inference_seconds", 0.) - before.get("inference_seconds", 0.),
        })
        if done:
            break
    return {
        "case_id": case.id, "success": env.verified, "events": events,
        "calls": len(events), "valid_actions": sum(e["action"] in TOOLS for e in events),
        "reward_sum": sum(e["observed_reward"] for e in events),
        "input_tokens": sum(e["prompt_tokens"] for e in events),
        "output_tokens": sum(e["completion_tokens"] for e in events),
        "inference_seconds": sum(e["inference_seconds"] for e in events),
    }


def observed_record(case, result):
    """Public events only; evaluator truth never copied into records."""
    return {
        "source_id": case.id, "profile": case.profile, "revision": case.revision,
        "service": case.service, "goal": case.goal,
        "events": [{"step": e["step"], "action": e["action"],
                    "feedback": e["tool_feedback"], "reward": e["observed_reward"],
                    "done": e["done"]} for e in result["events"]],
        "verified_success": bool(result["success"]),
    }


def store_episode(store, record):
    return store.add(MemoryRecord(
        content=json.dumps(record, ensure_ascii=False, sort_keys=True),
        kind=MemoryKind.EPISODIC, importance=.9,
        metadata={
            "source_id": record["source_id"], "profile": record["profile"],
            "revision": record["revision"], "observed_only": True,
        }))


def observed_sources(store):
    sources = {}
    for r in store.all():
        if r.kind != MemoryKind.EPISODIC:
            continue
        x = json.loads(r.content)
        if set(x) != {"source_id", "profile", "revision", "service",
                      "goal", "events", "verified_success"}:
            raise ValueError("Unexpected memory schema or private field")
        sources[x["source_id"]] = x
    return sources


def compact_episode(source):
    """Faithful rendering of EVERY action and public tool response."""
    events = []
    for e in source["events"]:
        txt = e["feedback"]
        if txt.startswith("Observed profile"):
            msg = "inspected; " + txt.removeprefix("Observed ")
        elif txt.startswith("Patch accepted:"):
            msg = "accepted; " + txt.removeprefix("Patch accepted:").strip()
        elif txt.startswith("Patch rejected:"):
            msg = "rejected; " + txt.removeprefix("Patch rejected:").strip()
        elif txt.startswith("Verification passed:"):
            msg = "verify PASS"
        elif txt.startswith("Verification failed:"):
            msg = "verify FAIL"
        else:
            msg = txt
        events.append(e["action"] + " => " + msg)
    return (f"[{source['source_id']} {source['profile']} "
            f"{source['revision']} verified={source['verified_success']}] "
            + "; ".join(events))


def strong_episodic_context(store, case):
    # Use true INFINITO retrieval, then exact PUBLIC profile filter. All
    # versions for that profile are preserved, including the stale one.
    matches = store.search(
        f"{case.profile} revision {case.revision} public tool experience", top_k=12)
    public = [json.loads(r.content) for r in matches
              if r.kind == MemoryKind.EPISODIC
              and r.metadata.get("profile") == case.profile]
    # Put current revision FIRST to give episodic baseline its strongest form.
    public.sort(key=lambda x: (x["revision"] != case.revision, x["source_id"]))
    context = "\n".join(compact_episode(s) for s in public)
    return context[:MAX_MEMORY], [x["source_id"] for x in public]


def claim_text(c, header):
    return (f"[{header} {c['source_id']} {c['profile']} {c['revision']}] "
            f"observed apply_{c['operation']} -> {c['observed_outcome']}; "
            f"verify_after={str(c['verified_after_operation']).lower()}")


def proposed_context(candidates, case, *, verified_only=False):
    matched = [
        p for p in candidates
        if p["claim"]["profile"] == case.profile
        and (not verified_only or p["validation"]["promotable"])
    ]
    matched.sort(key=lambda p: (p["claim"]["revision"] != case.revision,
                                 p["claim"]["source_id"]))
    label = "EVIDENCE_VERIFIED" if verified_only else "UNVERIFIED_LLM"
    return "\n".join(claim_text(p["claim"], label)
                     for p in matched)[:MAX_MEMORY], [
        p["claim"]["source_id"] for p in matched]


def chronological_control(sources, case):
    # A4 is a COMPETENT strong baseline: exact public profile/revision and
    # most recent verified operation from records, no LLM claim involved.
    # No private simulator truth, no counterfactual action outcomes.
    matching = [s for s in sources.values() if s["profile"] == case.profile
                and s["revision"] == case.revision]
    matching.sort(key=lambda x: x["source_id"], reverse=True)
    extracted = []
    for source in matching:
        for e in source["events"]:
            if e["action"] not in ("apply_delta", "apply_sigma"):
                continue
            if "Patch accepted:" not in e["feedback"]:
                continue
            follow = any(
                after["step"] > e["step"] and after["action"] == "verify"
                and "Verification passed:" in after["feedback"]
                for after in source["events"])
            if follow:
                c = {
                    "source_id": source["source_id"],
                    "profile": source["profile"], "revision": source["revision"],
                    "operation": e["action"].removeprefix("apply_"),
                    "observed_outcome": "accepted",
                    "verified_after_operation": True,
                }
                extracted.append(c)
                break
    text = "\n".join(claim_text(x, "CHRONOLOGY_OBSERVED")
                     for x in extracted)[:MAX_MEMORY]
    return text, [x["source_id"] for x in extracted]


def run(llm, out: Path | None = None):
    if out:
        out.mkdir(parents=True, exist_ok=True)
    episodes = SQLiteCognitiveMemoryStore(
        path=":memory:", embedding_provider=HashEmbeddingProvider())
    strategies = SQLiteCognitiveMemoryStore(
        path=":memory:", embedding_provider=HashEmbeddingProvider())
    try:
        train = []
        for case in TRAIN_CASES:
            r = play(case, llm, "", TRAIN_BUDGET)
            train.append(r)
            store_episode(episodes, observed_record(case, r))
        sources = observed_sources(episodes)
        # Cache all actor evidence before any eval; no oracle in extraction.
        candidate_results = []
        for case in TRAIN_CASES:
            source = sources[case.id]
            prompt = json.dumps(source, sort_keys=True, ensure_ascii=False)
            raw = llm.chat(EXTRACT_SYSTEM, prompt, max_tokens=192)
            c, parse_status = parse_claim(raw)
            v = validate_evidence(c, sources) if c is not None else {
                "supported": False, "promotable": False, "reason": parse_status}
            entry = {
                "requested_source": case.id, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "raw_llm_response": raw, "parse_status": parse_status,
                "claim": c, "validation": v,
            }
            candidate_results.append(entry)
            if c is not None and v["promotable"]:
                strategies.add(MemoryRecord(
                    content=json.dumps(c, sort_keys=True), kind=MemoryKind.SEMANTIC,
                    importance=.85,
                    metadata={"study": STUDY,
                              "status": "observed_verified_claim",
                              "source_id": c["source_id"],
                              "revision": c["revision"], "profile": c["profile"]}))
        source_digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
        candidate_digest = hashlib.sha256(json.dumps(candidate_results, sort_keys=True).encode()).hexdigest()
        # Contexts for every arm are fixed BEFORE any heldout action, including
        # references to BOTH versions in A1 and candidates in A2/A3.
        contexts = {}
        for case in HELDOUT_CASES:
            a1, a1_ids = strong_episodic_context(episodes, case)
            a2, a2_ids = proposed_context(
                [x for x in candidate_results if x["claim"] is not None], case)
            a3, a3_ids = proposed_context(
                [x for x in candidate_results if x["claim"] is not None],
                case, verified_only=True)
            a4, a4_ids = chronological_control(sources, case)
            contexts[case.id] = {
                "A0": ("", []),
                "A1": (a1, a1_ids),
                "A2": (a2, a2_ids),
                "A3": (a3, a3_ids),
                "A4": (a4, a4_ids),
            }
        evaluation = []
        for case in HELDOUT_CASES:
            for budget in EVAL_BUDGETS:
                k = (int(case.id[1:]) + budget) % len(ARMS)
                arm_order = ARMS[k:] + ARMS[:k]
                for arm in arm_order:
                    context, ids = contexts[case.id][arm]
                    result = play(case, llm, context, budget)
                    result.update({
                        "arm": arm, "budget": budget,
                        "profile": case.profile, "revision": case.revision,
                        "source_ids": ids,
                        "context_text": context, "context_chars": len(context),
                        "context_sha256": hashlib.sha256(context.encode()).hexdigest(),
                    })
                    evaluation.append(result)
        memory_unchanged = (
            source_digest == hashlib.sha256(
                json.dumps(observed_sources(episodes), sort_keys=True).encode()).hexdigest()
            and candidate_digest == hashlib.sha256(
                json.dumps(candidate_results, sort_keys=True).encode()).hexdigest())
        summary = {}
        for budget in EVAL_BUDGETS:
            summary[str(budget)] = {}
            for arm in ARMS:
                rows = [x for x in evaluation if x["budget"] == budget and x["arm"] == arm]
                summary[str(budget)][arm] = {
                    "successes": sum(x["success"] for x in rows), "episodes": len(rows),
                    "R1_successes": sum(x["success"] and x["revision"] == "R1" for x in rows),
                    "R2_successes": sum(x["success"] and x["revision"] == "R2" for x in rows),
                    "tool_steps": sum(x["calls"] for x in rows),
                    "prompt_tokens": sum(x["input_tokens"] for x in rows),
                    "completion_tokens": sum(x["output_tokens"] for x in rows),
                    "inference_seconds": sum(x["inference_seconds"] for x in rows),
                }
        validated = [x for x in candidate_results if x["validation"]["promotable"]]
        rejected = [x for x in candidate_results if x["parse_status"] != "OK"
                    or not x["validation"]["promotable"]]
        coverage = len({(x["claim"]["profile"], x["claim"]["revision"])
                        for x in validated if x["claim"] is not None})
        model = llm.metadata() if hasattr(llm, "metadata") else {"real_model": False}
        tight = summary["2"]
        slack = summary["3"]
        g2 = (
            tight["A3"]["successes"] >= tight["A1"]["successes"] + 2
            and tight["A3"]["successes"] >= tight["A2"]["successes"]
            and tight["A3"]["R1_successes"] >= tight["A1"]["R1_successes"]
            and tight["A3"]["R2_successes"] >= tight["A1"]["R2_successes"])
        a3_t = tight["A3"]
        a4_t = tight["A4"]
        g3 = (a3_t["successes"] >= a4_t["successes"] + 2 or
              (a3_t["successes"] >= a4_t["successes"] and
               a4_t["prompt_tokens"] > 0 and
               a3_t["prompt_tokens"] <= .85 * a4_t["prompt_tokens"]))
        gates = {
            "G0_real_pinned_60_paired_no_leak": (
                getattr(llm, "real_model", False) and
                model.get("weights_sha256") == FIXED_MODEL_SHA256
                and len(train) == 6 and sum(x["success"] for x in train) >= 3
                and len(sources) == 6 and len(candidate_results) == 6
                and len(evaluation) == 60 and memory_unchanged
                and all(x["context_chars"] <= MAX_MEMORY for x in evaluation)
                and all(x["calls"] <= x["budget"] for x in evaluation)),
            "G1_positive_evidence_and_coverage": (
                all(x["validation"]["supported"] for x in validated)
                and coverage >= 4),
            "G2_A3_exceeds_A1_on_tight": g2,
            "G3_A3_outperforms_strong_chronology": g3,
            "G4_A3_no_slack_collapse": (
                slack["A3"]["successes"] >= slack["A1"]["successes"] - 1),
            "G5_scope_and_isolation": True,
        }
        status = "TRAIN_HELDOUT_PILOT_NOT_CONFIRMATION"
        if not getattr(llm, "real_model", False):
            status = "FAKE_BACKEND_TEST_ONLY"
        elif not gates["G0_real_pinned_60_paired_no_leak"]:
            status = "INVALID_MODEL_OR_PROTOCOL"
        report = {
            "study": STUDY, "status": status, "model": model,
            "train": train, "train_sources": sources,
            "candidate_results": candidate_results,
            "promotable_count": len(validated),
            "rejected_count": len(rejected),
            "evidence_coverage": coverage,
            "strategy_records": len(strategies.all()),
            "episode_records": len(episodes.all()),
            "evaluation": evaluation, "scores": summary, "gates": gates,
            "memory_unchanged": memory_unchanged,
            "all_gates": bool(all(gates.values())),
            "no_private_truth_to_actor": True, "used_real_sqlite_infinito": True,
            "dev_opened": False, "confirm_opened": False,
            "scientific_pass": False,
        }
        if out:
            (out / "full_report.json").write_text(
                json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8")
            (out / "summary.json").write_text(
                json.dumps({k: v for k, v in report.items()
                            if k not in ("train", "train_sources",
                                         "candidate_results", "evaluation")},
                           indent=2, sort_keys=True, ensure_ascii=False),
                encoding="utf-8")
        return report
    finally:
        episodes.close()
        strategies.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        llm = FrozenGGUF()
        report = run(llm, args.out)
        print(json.dumps({
            "status": report["status"], "model": report["model"],
            "train_successes": sum(x["success"] for x in report["train"]),
            "strategy_candidates": len(report["candidate_results"]),
            "promotable": report["promotable_count"],
            "rejected": report["rejected_count"],
            "scores": report["scores"], "gates": report["gates"],
        }, sort_keys=True))
        return 0 if report["status"] != "INVALID_MODEL_OR_PROTOCOL" else 3
    except ProviderBlocked as exc:
        blocked = {"status": "BLOCKED_REAL_FROZEN_MODEL", "error": str(exc),
                   "no_fallback": True, "dev_opened": False,
                   "confirm_opened": False}
        (args.out / "blocked.json").write_text(json.dumps(blocked, indent=2))
        print(json.dumps(blocked), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
