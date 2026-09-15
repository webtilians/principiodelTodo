# INFINITO 3.0 — V11 freeze handoff — 2026-09-15

## Status

V11 is frozen and has **not** been executed live.

- Development branch: `fix/v11-lossless-current-planner-20260915`
- Frozen branch: `freeze/v11-lossless-current-planner-20260915`
- Exact frozen candidate: `d8fa0b37e889529202b0102097a81b656b6cc983`
- V11 bank: `src/infinito3/trajectory_holdout_v11_cases.py`
- V11 bank Git blob SHA-1: `1b429d451424925bc966a74361546d01e9c61e2f`
- V11 bank creation commit: `b3942be9029ca884964499acccbb38495ea1e294`
- V11 protocol revision: `v11_r1_lossless_current_planner`
- Integration status: `NOT_APPROVED`

The frozen branch currently has zero GitHub Actions runs. No V11 provider calls have been made.

## Why V11 exists

V10 was strong automatically but its manual audit exposed three architectural boundaries:

1. An exclusive current fact could retain update prose containing a superseded value.
2. Literal data used normalized `fact_value` rather than the lossless stored payload.
3. An explicit read imperative such as `State only my current home city.` could fall through to planner fallback retrieval.

V10 is therefore development evidence only and must not be replayed against the corrected candidate.

## Post-V10 corrections

### Canonical current-state projection

`SemanticTemporalCognitiveState` now projects an exclusive `ASSERT_FACT` through replacement semantics for durable storage. The original cognitive event remains unchanged for provenance, while the active memory contains only the canonical current value.

### Lossless literal data

Literal fast paths now treat `MemoryRecord.content` as the lossless output representation. Normalized `fact_value` remains available for semantic matching but is no longer used to reproduce literal payloads. Hyphens, punctuation and instruction-like text are preserved as inert data.

### Typed imperative reads

`CognitiveQueryPlan` now recognizes explicit `State ...` / `Report ...` read imperatives and reuses the shared `ContextIntent` contract to derive typed fact predicates. The fix introduces no entity dictionary and avoids `FALLBACK_RETRIEVAL` for the audited current-home query family.

## Deterministic evidence

GitHub Actions run `35015227614`, job `104536860175`:

- `258 passed`
- `4 deselected`
- `0 failed`
- OpenAI SDK pinned to `3.14.0`
- no live/provider evaluation

The four deselected tests are the exact-tree acceptance tests for previously frozen V7–V10 candidates. A new post-V10 regression explicitly verifies that the V10 preflight rejects the modified candidate.

New regressions cover:

- typed current-home imperative with no fallback;
- literal `cedar-642` preservation over a deliberately lossy semantic representation;
- instruction-like literal data with punctuation;
- canonical exclusive current state without stale-value leakage;
- rejection by the frozen V10 preflight.

## V11 holdout

V11 is a post-V10 diagnostic holdout, not a statistically blind benchmark. It was authored after the fixes were complete and uses new entities, dates, wording and distractors.

Structure:

- 4 trajectories
- 122 user turns
- 28 probes
- 4 strict `empty_context` controls
- `history_limit <= 5`

Audited families include:

- current facts and predecessor lineage;
- canonical current-state rendering after mixed old/new update prose;
- typed imperative current-home read;
- schedule windows, closure and reschedule identity;
- preference membership, retractions and latest ordering;
- exact lossless literal data containing instruction-like text and `cobalt-731`;
- mixed location/pet/preference/goal mutations;
- self-contained query isolation.

## V11 runner contract

Runner: `scripts/run_infinito3_trajectory_holdout_v11.py`

Default invocation is preflight-only and does not import/replay V11 or call a provider.

A live execution requires all of:

- explicit `--authorize-live-v11`;
- exact `--expected-revision d8fa0b37e889529202b0102097a81b656b6cc983`;
- clean worktree;
- configured `OPENAI_API_KEY`;
- OpenAI SDK `3.14.0`;
- all manifest-pinned file hashes unchanged.

The runner keeps SDK retries at zero, writes provider/turn audit evidence, requires a fresh output directory and leaves integration status `NOT_APPROVED` pending manual semantic, planner, literal, current-state, temporal and final-state review.

## Next permitted action

Do not edit or replay V11 after this freeze.

The next live step requires a **new explicit user authorization for V11**. One authorization permits one attempt only. Failure, cancellation or provider error does not authorize an automatic rerun.

Do not merge to `master` or `infinito-3.0` without a separate explicit instruction.
