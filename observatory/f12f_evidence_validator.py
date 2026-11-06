"""F12-F: source-grounded strategy verification WITHOUT a simulator oracle.

No hidden compatibility map is passed to validators, source memory, or LLM.
The only oracle is the action->feedback->verify trace the agent truly observed.
This is a technical provenance test, not proof of learning or OOD transfer.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord

STUDY = "F12-F"
MODEL_REPO = "bartowski/Qwen2.5-7B-Instruct-GGUF"
MODEL_FILE = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
QUANTIZATION = "Q4_K_M"
FIXED_MODEL_REVISION = "8911e8a47f92bac19d6f5c64a2e2095bd2f7d031"
FIXED_MODEL_SHA256 = "65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423"
MAX_ACTOR_ACTIONS = 5
TOOLS = ("inspect", "apply_delta", "apply_sigma", "verify")
CLAIM_FIELDS = ("source_id", "profile", "revision", "operation", "observed_outcome", "verified_after_operation")
ACTOR_SYSTEM = (
    "You are a maintenance agent operating only within a sandbox simulation. "
    "Return EXACTLY one JSON object with an allowed tool action: "
    "{\"action\":\"tool_name\"}. You must choose every action yourself. "
    "Use tool feedback to figure out which patch is compatible, and request "
    "verify after an accepted patch. All tools are simulated. No outside "
    "shell, account, or API access exists."
)
TOOLS_DESC = (
    "inspect = read public profile/revision; "
    "apply_delta = attempt maintenance change DELTA, may be accepted or rejected; "
    "apply_sigma = attempt maintenance change SIGMA, may be accepted or rejected; "
    "verify = check if a compatible change has taken effect (success ends task). "
    "Only observed tool responses establish whether an attempt worked."
)
EXTRACT_SYSTEM = (
    "You extract SOURCE-ATTRIBUTED, REVISABLE observations from a REAL tool "
    "trajectory. Never invent a tool call, outcome or verification. "
    "Return ONLY a single JSON object, no code fences or explanations, "
    "with exactly these fields: source_id, profile, revision, operation, "
    "observed_outcome, verified_after_operation. "
    "operation must be delta or sigma; observed_outcome must be accepted "
    "or rejected; verified_after_operation must be a JSON boolean. "
    "If neither operation was executed, return {}. "
    "A verification counts only if it occurred AFTER the claimed patch "
    "and the tool reported verification passed."
)
GLOBAL_SYSTEM = (
    "Create structured, source-attributed claims from these observed tool "
    "trajectories. This is an evidence audit; no expert answers are supplied. "
    "Return ONLY a JSON array of UP TO EIGHT objects, each with the fields "
    "source_id, profile, revision, operation, observed_outcome, "
    "verified_after_operation. Cite real source_id values. "
    "Different versions can contradict each other. Do not invent outcomes. "
    "All rules are hypotheses grounded only in the tool results."
)

@dataclass(frozen=True)
class Case:
    source_id: str
    profile: str
    revision: str
    service: str
    _compatible: str # Evaluator-private, consumed only by simulated tool.

    @property
    def goal(self):
        return (f"Service {self.service}, opaque profile {self.profile}, "
                f"public revision {self.revision}. Restore compatibility "
                "and verify the result using simulated tools.")

# Eight NEW instances, R1 first then R2, with inverted hidden compatibility.
# The valditator never receives a Case or this table.
CASES = (
    Case("V0", "ember", "R1", "orbital weather relay", "delta"),
    Case("V1", "moss", "R1", "subterranean telemetry worker", "sigma"),
    Case("V2", "lunar", "R1", "wildflower inventory linker", "delta"),
    Case("V3", "harbor", "R1", "solar activity report joiner", "sigma"),
    Case("V4", "ember", "R2", "transit status reconciler", "sigma"),
    Case("V5", "moss", "R2", "water treatment aggregator", "delta"),
    Case("V6", "lunar", "R2", "museum collections notifier", "sigma"),
    Case("V7", "harbor", "R2", "mountain rescue audit gateway", "delta"),
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



class ToolSandbox:
    """Only place in F12-F that ever knows private compatible operation."""
    def __init__(self, case: Case):
        self.__compatible = case._compatible
        self.public_profile = case.profile
        self.public_revision = case.revision
        self.changed = False
        self.verified = False

    def execute(self, action):
        if action not in TOOLS:
            return "Unknown tool; request rejected.", -.20, False
        if action == "inspect":
            return "Observed profile %s at revision %s." % (
                self.public_profile, self.public_revision), -.02, False
        if action == "verify":
            if self.changed:
                self.verified = True
                return "Verification passed: service compatible.", 1., True
            return "Verification failed: no compatible patch active.", -.15, False
        operation = action.removeprefix("apply_")
        if operation == self.__compatible:
            self.changed = True
            return "Patch accepted: compatible with this revision.", -.01, False
        return "Patch rejected: incompatible with this revision.", -.12, False


def parse_action(raw):
    try:
        parsed = json.loads(raw)
        if type(parsed) is dict and type(parsed.get("action")) is str:
            return parsed["action"]
    except (TypeError, ValueError):
        pass
    return "INVALID_LLM_OUTPUT"


def actor_prompt(case, observed):
    # No private compatibility, optimal path, or evaluator information.
    return (
        "GOAL: %s\nALLOWED TOOLS: %s\nTOOL CONTRACT: %s\n"
        "OBSERVED TOOL RESPONSES:\n%s\nNext tool, JSON only." %
        (case.goal, ", ".join(TOOLS), TOOLS_DESC,
         json.dumps(observed, ensure_ascii=False, separators=(",", ":")))
    )


def run_actor(case, llm):
    tools = ToolSandbox(case)
    history = []
    for step in range(MAX_ACTOR_ACTIONS):
        prompt = actor_prompt(case, history)
        raw = llm.chat(ACTOR_SYSTEM, prompt)
        action = parse_action(raw)
        feedback, reward, done = tools.execute(action)
        history.append({
            "step": step, "action": action,
            "tool_feedback": feedback, "reward": reward, "done": done,
            "llm_response": raw, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        })
        if done:
            break
    # No hidden truth in this record, or in the memory/validator.
    return {
        "source_id": case.source_id, "profile": case.profile,
        "revision": case.revision, "service": case.service,
        "goal": case.goal, "events": [
            {"step": h["step"], "action": h["action"],
             "feedback": h["tool_feedback"], "reward": h["reward"],
             "done": h["done"]} for h in history
        ],
        "verified_success": tools.verified,
        "actor_trace": history,  # Also public and action source provenance.
    }


def public_record(record):
    """Projection of actor observations, never case or world truth."""
    return {
        "source_id": record["source_id"],
        "profile": record["profile"], "revision": record["revision"],
        "service": record["service"], "goal": record["goal"],
        "events": [dict(e) for e in record["events"]],
        "verified_success": record["verified_success"],
    }


def store_experience(store, record):
    safe = public_record(record)
    return store.add(MemoryRecord(
        content=json.dumps(safe, sort_keys=True, ensure_ascii=False),
        kind=MemoryKind.EPISODIC, importance=.9,
        metadata={
            "study": STUDY, "source_id": safe["source_id"],
            "profile": safe["profile"], "revision": safe["revision"],
            "observed_only": True
        }))


def get_observed_sources(store):
    """Fetch evidence through REAL memory store, no private simulator fields."""
    entries = store.all()
    result = {}
    for entry in entries:
        if entry.kind != MemoryKind.EPISODIC:
            continue
        safe = json.loads(entry.content)
        if set(safe) != {"source_id", "profile", "revision", "service",
                         "goal", "events", "verified_success"}:
            raise ValueError("Private or unexpected field leaked to evidence store")
        result[safe["source_id"]] = safe
    return result


def parse_candidate(raw, multiple=False):
    try:
        obj = json.loads(raw)
    except (TypeError, ValueError):
        return [], "INVALID_JSON"
    if multiple:
        if type(obj) is not list or len(obj) > 8:
            return [], "INVALID_ARRAY"
        values = obj
    else:
        if type(obj) is not dict:
            return [], "INVALID_OBJECT"
        if not obj:
            return [], "ABSTAINED"
        values = [obj]
    if any(type(c) is not dict or set(c) != set(CLAIM_FIELDS)
           or type(c.get("source_id")) is not str
           or type(c.get("profile")) is not str
           or c.get("revision") not in ("R1", "R2")
           or c.get("operation") not in ("delta", "sigma")
           or c.get("observed_outcome") not in ("accepted", "rejected")
           or type(c.get("verified_after_operation")) is not bool
           for c in values):
        return [], "INVALID_SCHEMA"
    return values, "OK"


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


def cited_only(claim, sources):
    return (type(claim) is dict and
            claim.get("source_id") in sources and
            claim.get("observed_outcome") == "accepted" and
            claim.get("verified_after_operation") is True)


def recency_only(claim, sources):
    if not cited_only(claim, sources):
        return False
    source = sources[claim["source_id"]]
    newest_revision = max(s["revision"] for s in sources.values()
                          if s["profile"] == source["profile"])
    return source["revision"] == newest_revision


def checked_proposal(claim, sources):
    return {
        "citation_only": cited_only(claim, sources),
        "recency_only": recency_only(claim, sources),
        "evidence": validate_evidence(claim, sources),
    }


def evidence_claims_from_public_trace(source):
    """Public-trace benchmark positives, NOT expert or model claims."""
    for index, event in enumerate(source["events"]):
        if event["action"] not in ("apply_delta", "apply_sigma"):
            continue
        text = event["feedback"].lower()
        if "patch accepted:" in text:
            outcome = "accepted"
        elif "patch rejected:" in text:
            outcome = "rejected"
        else:
            continue
        verified = outcome == "accepted" and any(
            followup["action"] == "verify" and
            "verification passed:" in followup["feedback"].lower()
            for followup in source["events"][index + 1:])
        yield {
            "source_id": source["source_id"], "profile": source["profile"],
            "revision": source["revision"],
            "operation": event["action"].removeprefix("apply_"),
            "observed_outcome": outcome,
            "verified_after_operation": bool(verified),
        }


def make_adversarial(source, source_ids):
    """Fixed adversarial controls generated from ACTUAL public observations.

    This is a specification check, not independent evidence of cognitive gain.
    """
    claims = list(evidence_claims_from_public_trace(source))
    if not claims:
        return []
    base = next((c for c in claims
                 if c["observed_outcome"] == "accepted"
                 and c["verified_after_operation"]), claims[0])
    flip = lambda x, a, b: b if x == a else a
    mutated = [
        ("observed_positive_or_negative", base, True),
        ("operation_swapped", {**base,
                               "operation": flip(base["operation"], "delta", "sigma")}, False),
        ("outcome_inverted", {**base,
                              "observed_outcome": flip(base["observed_outcome"],
                                                        "accepted", "rejected")}, False),
        ("revision_falsified", {**base,
                                "revision": flip(base["revision"], "R1", "R2")}, False),
        ("fake_source", {**base, "source_id": "NEVER_EXECUTED"}, False),
        ("verification_inverted", {**base,
                                   "verified_after_operation": not base["verified_after_operation"]}, False),
        ("historical_revision_used_in_other_target", base, False),
    ]
    results = []
    for kind, claim, valid in mutated:
        if kind == "historical_revision_used_in_other_target":
            target = flip(source["revision"], "R1", "R2")
        else:
            target = source["revision"]
        audited = validate_evidence(claim, source_ids, target_revision=target)
        is_positive = valid and base["observed_outcome"] == "accepted" and base["verified_after_operation"]
        results.append({
            "source_id": source["source_id"], "kind": kind, "claim": claim,
            "label_promotable": bool(is_positive),
            "public_trace_label_supported": bool(valid),
            "target_revision": target,
            "baseline_citation_promotes": cited_only(claim, source_ids),
            "baseline_recency_promotes": recency_only(claim, source_ids),
            "evidence_validation": audited,
        })
    return results


def _confusion(rows, key):
    tp = fp = tn = fn = 0
    for row in rows:
        positive = bool(row["label_promotable"])
        predicted = bool(row[key] if key != "evidence"
                         else row["evidence_validation"]["promotable"])
        if positive and predicted:
            tp += 1
        elif positive and not predicted:
            fn += 1
        elif not positive and predicted:
            fp += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None}


def run(llm, out: Path | None = None):
    if out:
        out.mkdir(parents=True, exist_ok=True)
    store = SQLiteCognitiveMemoryStore(
        path=":memory:", embedding_provider=HashEmbeddingProvider())
    try:
        actors = []
        for case in CASES:
            trace = run_actor(case, llm)
            actors.append(trace)
            store_experience(store, trace)
        # All decisions in the validators are made using public-source records.
        sources = get_observed_sources(store)
        frozen_sources_sha256 = hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode()).hexdigest()
        # Live LLM generates one claim per episode, plus a single global batch.
        generations = []
        for case in CASES:
            visible = sources[case.source_id]
            prompt = json.dumps(visible, sort_keys=True, ensure_ascii=False)
            raw = llm.chat(EXTRACT_SYSTEM, prompt, max_tokens=192)
            parsed, status = parse_candidate(raw)
            generations.append({
                "scope": "per_source", "requested_source": case.source_id,
                "raw_response": raw, "parse_status": status,
                "claims": parsed, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            })
        full = json.dumps(list(sources.values()), sort_keys=True, ensure_ascii=False)
        raw = llm.chat(GLOBAL_SYSTEM, full, max_tokens=600)
        parsed, status = parse_candidate(raw, multiple=True)
        generations.append({
            "scope": "global_synthesis", "raw_response": raw,
            "parse_status": status, "claims": parsed,
            "prompt_sha256": hashlib.sha256(full.encode()).hexdigest(),
        })
        candidates = [candidate for item in generations for candidate in item["claims"]]
        checked = [{"claim": claim, **checked_proposal(claim, sources)} for claim in candidates]
        adversarial = [row for source in sources.values()
                       for row in make_adversarial(source, sources)]
        confusion = {
            "citation": _confusion(adversarial, "baseline_citation_promotes"),
            "recency": _confusion(adversarial, "baseline_recency_promotes"),
            "evidence": _confusion(adversarial, "evidence"),
        }
        positives = sum(r["label_promotable"] for r in adversarial)
        altered = [r for r in adversarial
                   if not r["label_promotable"] and
                   r["kind"] not in ("observed_positive_or_negative",)]
        rejected = sum(not row["evidence_validation"]["promotable"] for row in altered)
        actual_promoted = sum(x["evidence"]["promotable"] for x in checked)
        raw_ep = [x for x in actors if x["verified_success"]]
        record_integrity = (len(sources) == len(CASES) == len(store.all()) and
                            frozen_sources_sha256 == hashlib.sha256(
                                json.dumps(get_observed_sources(store), sort_keys=True).encode()).hexdigest())
        model = llm.metadata() if hasattr(llm, "metadata") else {"real_model": False}
        gates = {
            "G0_real_model_same_weights_public_sources_only":
                bool(getattr(llm, "real_model", False)
                     and model.get("weights_sha256") == FIXED_MODEL_SHA256
                     and record_integrity),
            "G1_zero_false_promotions":
                positives > 0 and confusion["evidence"]["FP"] == 0
                and all(x["evidence"]["supported"] for x in checked
                        if x["evidence"]["promotable"]),
            "G2_preserve_supported_and_reject_mutations":
                positives > 0 and
                confusion["evidence"]["TP"] / positives >= .95 and
                (rejected / len(altered) if altered else 0) >= .95,
            "G3_comparators_all_scored":
                len(adversarial) >= 7 and len(confusion) == 3,
            "G4_no_cognitive_claim_and_limited_scope": True,
        }
        status = "TECHNICAL_GUARDRAIL_PILOT" if all(gates.values()) else "NO_GO_GUARDRAIL_PILOT"
        if not getattr(llm, "real_model", False):
            status = "FAKE_BACKEND_TECHNICAL_TEST_ONLY"
        report = {
            "study": STUDY, "status": status,
            "model": model, "actor_episodes": [public_record(x) for x in actors],
            "actor_raw_traces": [{
                "source_id": x["source_id"], "trace": x["actor_trace"]}
                for x in actors],
            "source_count": len(sources), "source_sha256": frozen_sources_sha256,
            "episode_count": len(actors), "verified_actor_episodes": len(raw_ep),
            "model_generations": generations, "llm_candidate_checks": checked,
            "generated_claims_total": len(candidates),
            "llm_candidate_promotions": actual_promoted,
            "adversarial": adversarial, "confusion": confusion,
            "adversarial_mutations": len(altered),
            "adversarial_rejected": rejected, "control_positives": positives,
            "gates": gates, "truth_map_shared_with_verifier": False,
            "real_infinito_sqlite": True, "training_policy_learned": False,
            "memory_product_changed": False, "dev_opened": False,
            "confirm_opened": False, "scientific_pass": False,
        }
        if out:
            (out / "full_audit.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
            (out / "summary.json").write_text(
                json.dumps({k: v for k, v in report.items()
                            if k not in ("actor_episodes", "actor_raw_traces",
                                         "model_generations", "llm_candidate_checks",
                                         "adversarial")},
                           indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
        return report
    finally:
        store.close()


def main():
    parser = argparse.ArgumentParser(description="F12-F verify claims against agent-observed tool events")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        model = FrozenGGUF()
        report = run(model, args.out)
        print(json.dumps({
            "status": report["status"], "gates": report["gates"],
            "verified_actor_episodes": report["verified_actor_episodes"],
            "model": report["model"], "confusion": report["confusion"],
            "generated_claims_total": report["generated_claims_total"],
            "llm_candidate_promotions": report["llm_candidate_promotions"],
            "generation_statuses": [g["parse_status"] for g in report["model_generations"]],
        }, sort_keys=True))
    except ProviderBlocked as exc:
        status = {"status": "BLOCKED_GENUINE_LLM", "reason": str(exc),
                  "no_mock_fallback": True, "dev_opened": False,
                  "confirm_opened": False}
        (args.out / "blocked.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
