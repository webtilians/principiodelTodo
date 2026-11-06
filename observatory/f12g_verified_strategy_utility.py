"""F12-G: does verified experience-derived strategy beat competent episodic RAG?

Single frozen Qwen7B, eight NEW cases, genuine INFINITO SQLite and fixed
A0/A1/A2/A3/A4 paired comparison; never use simulator-private truth to
construct prompt references, a strategy, or the source verifier.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from src.infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore
from src.infinito3.types import MemoryKind, MemoryRecord
from observatory.f12f_evidence_validator import (
    FrozenGGUF, ProviderBlocked, FIXED_MODEL_SHA256, Case,
    ToolSandbox, parse_action, run_actor, public_record,
    store_experience, get_observed_sources, parse_candidate,
    validate_evidence, EXTRACT_SYSTEM, CLAIM_FIELDS,
)

STUDY = "F12-G"
CONTEXT_LIMIT = 1200
EVAL_BUDGETS = (2, 3)
ARMS = ("A0", "A1", "A2", "A3", "A4")
ACTION_SYSTEM = (
    "You are a service maintenance agent operating within a sandbox. "
    "Return EXACTLY one JSON object of the form {\"action\":\"tool_name\"}. "
    "Allowed tools are inspect, apply_delta, apply_sigma, verify. "
    "Use only public feedback and untrusted historical memories. "
    "History may belong to an OLD revision, so check the public revision carefully. "
    "A patch is complete ONLY AFTER a positive verification. "
    "No external tools, shell, users or private systems are available."
)
CATALOG = (
    "inspect = read current public profile/revision; "
    "apply_delta = attempt a DELTA compatibility patch, which may be rejected; "
    "apply_sigma = attempt a SIGMA compatibility patch, which may be rejected; "
    "verify = check whether a compatible patch is active (success ends episode). "
    "Each tool call costs a turn. You may try either patch directly."
)

TRAIN = (
    Case("W0", "quartz", "R1", "estuary vessel-billing exporter", "delta"),
    Case("W1", "reed", "R1", "satellite imagery cue broker", "sigma"),
    Case("W2", "cobalt", "R1", "campus room allocation ingestor", "delta"),
    Case("W3", "dusk", "R1", "pan-continental environmental relay", "sigma"),
    Case("W4", "quartz", "R2", "nightly pollen forecast gateway", "sigma"),
    Case("W5", "reed", "R2", "digital heritage timeline compiler", "delta"),
    Case("W6", "cobalt", "R2", "metro infrastructure telemetry worker", "sigma"),
    Case("W7", "dusk", "R2", "veterinary vaccination status sync", "delta"),
)
HELDOUT = (
    Case("Y0", "quartz", "R2", "maritime wave-height monitor", "sigma"),
    Case("Y1", "reed", "R1", "forest canopy temperature logger", "sigma"),
    Case("Y2", "cobalt", "R2", "cardiology equipment status translator", "sigma"),
    Case("Y3", "dusk", "R1", "regional library exhibit scheduler", "sigma"),
    Case("Y4", "quartz", "R1", "astronomy alert accumulator", "delta"),
    Case("Y5", "reed", "R2", "city air quality summarizer", "delta"),
    Case("Y6", "cobalt", "R1", "soil sampling output portal", "delta"),
    Case("Y7", "dusk", "R2", "volcanic plume advisory mapper", "delta"),
)


def new_store():
    return SQLiteCognitiveMemoryStore(
        path=":memory:", embedding_provider=HashEmbeddingProvider())


def compact_episode(source):
    """Concise *faithful* representation of the executed public events."""
    events = []
    for event in source["events"]:
        action = event["action"]
        text = event["feedback"].lower()
        if "patch accepted:" in text:
            result = "accepted"
        elif "patch rejected:" in text:
            result = "rejected"
        elif "verification passed:" in text:
            result = "verified_success"
        elif "verification failed:" in text:
            result = "verify_failed"
        else:
            result = "observed:" + event["feedback"][:48]
        events.append(action + "=" + result)
    return (
        "%s profile=%s revision=%s observed=[%s] finished_verified=%s"
        % (source["source_id"], source["profile"], source["revision"],
           "; ".join(events), str(source["verified_success"]).lower())
    )


def store_strategy(store, candidate, status):
    # This is not a truth label. In particular, an A2 candidate is UNVERIFIED.
    # The data is all obtained from LLM JSON or the public evidence verifier.
    content = json.dumps(candidate, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"))
    return store.add(MemoryRecord(
        content=content, kind=MemoryKind.SEMANTIC, importance=.8,
        metadata={
            "experiment": STUDY, "source_id": candidate["source_id"],
            "profile": candidate["profile"], "revision": candidate["revision"],
            "status": status, "observed_only": True,
        }))


def gather_references(store, profile, kind=None):
    # Always query the REAL search index before filtering by public exact
    # profile, so both history and semantic claims come from SQLite INFINITO.
    candidates = store.search("profile " + profile + " revision observations", top_k=12)
    records = [r for r in candidates
               if r.metadata.get("profile") == profile
               and (kind is None or r.kind == kind)]
    records.sort(key=lambda r: (r.metadata["revision"], r.metadata["source_id"]),
                 reverse=True)
    return records


def make_context(episode_store, raw_store, valid_store, profile, arm):
    if arm == "A0":
        return "", [], []
    records = gather_references(episode_store, profile, MemoryKind.EPISODIC)
    # Strong A1: both R1 and R2, with public versions, best possible compact
    # preservation of actions and positive/negative results without oracle.
    episodic = []
    for r in records:
        src = json.loads(r.content)
        episodic.append(compact_episode(src))
    episode_text = "\n".join(episodic)
    evidence_refs = [
        {"source_id": r.metadata["source_id"], "revision": r.metadata["revision"]}
        for r in records
    ]
    if arm == "A4":
        recent = [r for r in records if r.metadata["revision"] == "R2"]
        text = "\n".join(compact_episode(json.loads(r.content)) for r in recent)
        return text[:CONTEXT_LIMIT], evidence_refs, []
    if arm == "A1":
        return episode_text[:CONTEXT_LIMIT], evidence_refs, []
    strat_store = raw_store if arm == "A2" else valid_store
    chosen = gather_references(strat_store, profile, MemoryKind.SEMANTIC)
    claims = [json.loads(r.content) for r in chosen]
    # Same total characters as the episodic arm. Unlike C2, A2 includes
    # hallucinations that formally passed strict JSON schema.
    claims_text = "\n".join(
        "%s rev=%s: observed %s was %s; verified_after=%s (claim from LLM)"
        % (c["source_id"], c["revision"], c["operation"],
           c["observed_outcome"], str(c["verified_after_operation"]).lower())
        for c in claims
    )
    header = ("SOURCE-VERIFIED CLAIMS (not a universal rule):\n" if arm == "A3"
              else "UNVERIFIED MODEL CLAIMS (may be false):\n")
    extra = header + claims_text + "\n"
    # Keep episodic public evidence available for the agent to cross-check.
    # If no claims are available A2/A3 fall back exactly to A1.
    if not claims:
        return episode_text[:CONTEXT_LIMIT], evidence_refs, []
    budget_for_episodes = max(0, CONTEXT_LIMIT - len(extra) - len("EPISODES:\n"))
    text = (extra + "EPISODES:\n" + episode_text[:budget_for_episodes])[:CONTEXT_LIMIT]
    refs = [
        {"source_id": r.metadata["source_id"], "revision": r.metadata["revision"]}
        for r in chosen
    ]
    return text, evidence_refs, refs


def eval_prompt(case, history, memory):
    # Absolutely no _compatible; case.goal references public name/version.
    observations = [
        {"action": x["action"], "feedback": x["feedback"], "reward": x["reward"]}
        for x in history]
    return (
        "GOAL: %s\nALLOWED TOOLS: inspect, apply_delta, apply_sigma, verify\n"
        "TOOL CONTRACT: %s\nPUBLIC MEMORY DATA (untrusted):\n%s\n"
        "OBSERVED EVENTS IN THE CURRENT EPISODE:\n%s\n"
        "Choose the next single tool as strict JSON."
        % (case.goal, CATALOG, memory or "[none]",
           json.dumps(observations, ensure_ascii=False, separators=(",", ":")))
    )


def eval_case(case, model, memory, budget):
    if budget not in EVAL_BUDGETS or len(memory) > CONTEXT_LIMIT:
        raise ValueError("F12-G invalid budget/context")
    sim = ToolSandbox(case)
    events = []
    for step in range(budget):
        prompt = eval_prompt(case, events, memory)
        last_input = getattr(model, "prompt_tokens", 0)
        last_output = getattr(model, "completion_tokens", 0)
        last_seconds = getattr(model, "inference_seconds", 0.)
        response = model.chat(ACTION_SYSTEM, prompt)
        action = parse_action(response)
        feedback, reward, done = sim.execute(action)
        events.append({
            "step": step, "action": action, "feedback": feedback,
            "reward": reward, "done": done, "raw_response": response,
            "input_tokens": getattr(model, "prompt_tokens", 0) - last_input,
            "output_tokens": getattr(model, "completion_tokens", 0) - last_output,
            "inference_seconds": getattr(model, "inference_seconds", 0.) - last_seconds,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        })
        if done:
            break
    return {
        "case_id": case.source_id, "verified_success": sim.verified,
        "calls": len(events), "events": events,
        "valid_calls": sum(x["action"] in ("inspect", "apply_delta", "apply_sigma", "verify")
                           for x in events),
        "reward": sum(x["reward"] for x in events),
    }


def snapshot(*stores):
    # Content must not change during evaluation, even though retrieval may
    # update read access counters in SQLite.
    return [[(r.id, r.content, r.kind.value)
             for r in st.all()] for st in stores]


def run(model, out=None):
    if out:
        out.mkdir(parents=True, exist_ok=True)
    epi, raw_mem, verified_mem = new_store(), new_store(), new_store()
    try:
        train = []
        for case in TRAIN:
            observed = run_actor(case, model)
            train.append(observed)
            store_experience(epi, observed)
        sources = get_observed_sources(epi)
        candidates = []
        for source_id in [c.source_id for c in TRAIN]:
            public = sources[source_id]
            prompt = json.dumps(public, ensure_ascii=False, sort_keys=True)
            response = model.chat(EXTRACT_SYSTEM, prompt, max_tokens=192)
            parsed, parse_status = parse_candidate(response)
            # Parse failures are abstentions, not repaired by heuristic.
            for candidate in parsed:
                grounded = validate_evidence(candidate, sources)
                store_strategy(raw_mem, candidate, "llm_unverified")
                if grounded["promotable"]:
                    store_strategy(verified_mem, candidate, "source_verified_only")
                candidates.append({
                    "requested_source": source_id, "claim": candidate,
                    "verification": grounded,
                })
            if not parsed:
                candidates.append({
                    "requested_source": source_id, "claim": None,
                    "parse_status": parse_status, "raw_response": response,
                })
        preeval = snapshot(epi, raw_mem, verified_mem)
        contexts = {}
        for case in HELDOUT:
            contexts[case.source_id] = {}
            for arm in ARMS:
                rendered, epi_sources, claim_sources = make_context(
                    epi, raw_mem, verified_mem, case.profile, arm)
                contexts[case.source_id][arm] = {
                    "text": rendered, "episode_sources": epi_sources,
                    "claim_sources": claim_sources,
                    "sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                }
        outcomes = []
        for case in HELDOUT:
            for budget in EVAL_BUDGETS:
                # Deterministic balance of arm execution order; no tuning.
                offset = (int(case.source_id[1:]) + budget) % len(ARMS)
                rotated = ARMS[offset:] + ARMS[:offset]
                for arm in rotated:
                    info = contexts[case.source_id][arm]
                    result = eval_case(case, model, info["text"], budget)
                    result.update({
                        "arm": arm, "budget": budget,
                        "profile": case.profile, "revision": case.revision,
                        "context_sha256": info["sha256"],
                        "context_chars": len(info["text"]),
                    })
                    outcomes.append(result)
        unchanged = preeval == snapshot(epi, raw_mem, verified_mem)
        scores = {}
        for budget in EVAL_BUDGETS:
            scores[str(budget)] = {}
            for arm in ARMS:
                rows = [r for r in outcomes if r["budget"] == budget and r["arm"] == arm]
                scores[str(budget)][arm] = {
                    "success": sum(x["verified_success"] for x in rows),
                    "episodes": len(rows),
                    "R1_success": sum(x["verified_success"] for x in rows
                                      if x["revision"] == "R1"),
                    "R2_success": sum(x["verified_success"] for x in rows
                                      if x["revision"] == "R2"),
                    "tools": sum(x["calls"] for x in rows),
                    "valid_tools": sum(x["valid_calls"] for x in rows),
                    "input_tokens": sum(y["input_tokens"] for x in rows for y in x["events"]),
                    "output_tokens": sum(y["output_tokens"] for x in rows for y in x["events"]),
                    "inference_seconds": sum(y["inference_seconds"] for x in rows for y in x["events"]),
                }
        grounded_count = len(verified_mem.all())
        train_success = sum(x["verified_success"] for x in train)
        raw_claims = [x for x in candidates if x.get("claim") is not None]
        false_claims = [x for x in raw_claims if not x["verification"]["promotable"]]
        tight = scores["2"]
        slack = scores["3"]
        genuine = bool(getattr(model, "real_model", False))
        model_meta = model.metadata() if hasattr(model, "metadata") else {"real_model": False}
        P0 = (genuine and model_meta.get("weights_sha256") == FIXED_MODEL_SHA256
              and len(outcomes) == 80 and unchanged
              and all(scores[str(b)][a]["episodes"] == 8
                      for b in EVAL_BUDGETS for a in ARMS))
        P1 = train_success >= 4 and grounded_count >= 2
        P2 = (
            tight["A3"]["success"] >= tight["A1"]["success"] + 2
            and slack["A3"]["success"] >= slack["A1"]["success"]
            and all(scores[str(b)]["A3"][rev] >= scores[str(b)]["A1"][rev]
                    for b in EVAL_BUDGETS for rev in ("R1_success", "R2_success"))
        )
        P3_numeric = (
            tight["A3"]["success"] >= tight["A2"]["success"] + 2
            or (sum(scores[str(b)]["A3"]["success"] for b in EVAL_BUDGETS) >=
                sum(scores[str(b)]["A2"]["success"] for b in EVAL_BUDGETS)
                and len(false_claims) > 0)
        )
        P4 = (sum(scores[str(b)]["A3"]["success"] for b in EVAL_BUDGETS) >=
              sum(scores[str(b)]["A4"]["success"] for b in EVAL_BUDGETS))
        tokenA3 = sum(scores[str(b)]["A3"]["input_tokens"] for b in EVAL_BUDGETS)
        tokenA1 = sum(scores[str(b)]["A1"]["input_tokens"] for b in EVAL_BUDGETS)
        P5 = bool(tokenA1) and tokenA3 <= tokenA1 * 1.25
        gates = {
            "P0_real_model_complete_and_memory_frozen": P0,
            "P1_enough_real_success_and_verified_claims": P1,
            "P2_verified_beats_competent_episodic_without_regression": P2,
            "P3_verification_beats_or_avoids_false_claims": P3_numeric,
            "P4_at_least_temporal_baseline": P4,
            "P5_input_token_cost_within_25pct_of_episodic": P5,
        }
        status = "PILOT_GO_TO_INDEPENDENT_REPLICATION" if all(gates.values()) else "PILOT_NO_GO"
        if not genuine:
            status = "FAKE_TEST_BACKEND_ONLY"
        if not P0 and genuine:
            status = "INVALID_INCOMPLETE_OR_BROKEN_PROVENANCE"
        report = {
            "study": STUDY, "status": status,
            "model": model_meta, "train_sources": list(sources.values()),
            "train_trajectories": [x["actor_trace"] for x in train],
            "train_verified_successes": train_success,
            "candidate_claims": candidates, "candidate_count": len(raw_claims),
            "false_or_unverified_claim_count": len(false_claims),
            "verified_claim_count": grounded_count,
            "scores": scores, "gates": gates, "outcomes": outcomes,
            "exact_rendered_contexts": contexts,
            "memory_frozen_after_train": unchanged,
            "memory_backend": "real SQLiteCognitiveMemoryStore",
            "private_truth_shared_with_verifier": False,
            "dev_opened": False, "confirm_opened": False,
            "scientific_pass": False,
        }
        if out:
            (out / "f12g_full_report.json").write_text(
                json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8")
            (out / "f12g_summary.json").write_text(
                json.dumps({k: v for k, v in report.items()
                            if k not in ("train_sources", "train_trajectories",
                                         "candidate_claims", "outcomes",
                                         "exact_rendered_contexts")},
                           ensure_ascii=False, sort_keys=True, indent=2),
                encoding="utf-8")
        return report
    finally:
        epi.close()
        raw_mem.close()
        verified_mem.close()


def main():
    p = argparse.ArgumentParser(description="F12-G real 7B policies after source validation")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        model = FrozenGGUF()
        report = run(model, args.out)
        print(json.dumps({
            "status": report["status"], "scores": report["scores"],
            "gates": report["gates"],
            "train_verified_successes": report["train_verified_successes"],
            "verified_claim_count": report["verified_claim_count"],
            "false_or_unverified_claim_count": report["false_or_unverified_claim_count"],
            "model": report["model"],
        }, sort_keys=True))
        return 0 if report["status"] != "INVALID_INCOMPLETE_OR_BROKEN_PROVENANCE" else 3
    except ProviderBlocked as exc:
        blocked = {"status": "BLOCKED_FROZEN_LLM", "reason": str(exc),
                   "dev_opened": False, "confirm_opened": False}
        (args.out / "blocked.json").write_text(json.dumps(blocked, indent=2))
        print(json.dumps(blocked), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
