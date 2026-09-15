# INFINITO 3.0 — V9 freeze (2026-09-15)

## Purpose

V9 is the first held-out trajectory bank authored after the V8 live audit and after two bounded corrective families were completed:

1. **Answer grounding contract** — direct, non-conflicting retrieved evidence must be used; literal instruction-like text may be reported when the user explicitly asks for the stored data, but must never be executed.
2. **Goal temporal canonicalization** — after a reschedule, structured `due_at` is the authoritative current time and stale weekday/date/time wording must not leak from the original goal description into current context.

ContextIntent itself is not changed for this milestone. V8 is development evidence for this candidate.

## Frozen V9 bank

- File: `src/infinito3/trajectory_holdout_v9_cases.py`
- Git blob SHA-1: `25d5857cd8ec8f60cb8df5e1007f46e1ba1e9453`
- Authoring commit: `9fcb205714cf6d4ce2b648a5c3e87cf4e7192e44`
- Scenarios: **4**
- Turns: **122**
- Probes: **28**
- Strict empty-context controls: **4**
- Includes explicit answer-grounding probes, literal-data reporting, reschedule-identity checks and stale-schedule exclusions.

The entities, dates, distractors and wording are new relative to V8. This is still a **post-diagnostic holdout**: the families being tested were selected after observing V8, so V9 must not be described as statistically independent of the research process as a whole.

The bank must never be edited after this freeze. If a defect is discovered in the bank itself, preserve V9 and create a new version.

## Frozen evaluation protocol

- Runner: `scripts/run_infinito3_trajectory_holdout_v9.py`
- Runner Git blob SHA-1: `f89c3d08f1b78d98558b5e522712b5b660f9e224`
- Manifest: `docs/INFINITO_3_V9_RUNNER_MANIFEST.json`
- Protocol revision: `v9_r1_grounding_goalcanon`
- Answer/event/reranker model: `gpt-5.6-luna`
- Embedding model: `text-embedding-3-small`
- Answer quality gate: `>= 0.85`
- Context quality gate: `>= 0.90`
- SDK retries: **0**

The default runner command is preflight-only. It does not import or replay the V9 bank and does not contact the model provider. Live mode requires all of the following:

- explicit `--authorize-live-v9`;
- exact expected Git revision;
- clean worktree;
- configured provider secret;
- exact frozen SDK version;
- successful manifest and bank hash validation.

A completed run remains `NOT_APPROVED` until manual semantic, grounding, temporal-canonicalization and final-state audit.

## Deterministic validation before any V9 live run

After correcting two manifest-only freeze-wiring mistakes, the final deterministic CI result is:

- **225 passed**
- **2 deselected**
- no provider calls

The two deselected tests are acceptance guards for the old frozen V7 and V8 candidates. They are expected not to accept the post-V8 candidate because its implementation intentionally changed. Their historical frozen references remain untouched.

The two intermediate CI failures did **not** alter the V9 bank or candidate behavior:

1. the first manifest draft copied an obsolete model configuration;
2. the second draft contained a non-canonical runner blob hash.

Both were corrected in the manifest only. The frozen V9 bank remained at the same Git blob SHA throughout.

## Live-execution boundary

As of this freeze document:

- **V9 live executions: 0**
- no V9 turn has been sent to the provider;
- no live workflow trigger exists for V9;
- no merge to `master` or `infinito-3.0` is authorized or performed.

A first live V9 execution requires a new explicit user authorization. It must be exactly one run against the frozen candidate revision, followed by probe-by-probe and final-state audit. No automatic rerun is permitted.
