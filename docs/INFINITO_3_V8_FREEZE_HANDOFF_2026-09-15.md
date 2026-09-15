# INFINITO 3.0 — V8 frozen candidate handoff — 2026-09-15

## Current decision

V7 was executed exactly once on frozen revision `ab6678b0cbde899226535514bd7f24e67bd3b4cb` (GitHub Actions run `34992914704`). Aggregate gates passed, but manual audit found three beta-blocking mixed-state defects: residence-move/reschedule ambiguity, substring corruption in lifecycle target cleanup, and failure to store generic literal verification-note commands. V7 is therefore development evidence and must not be replayed against the corrected candidate.

## Corrected candidate

Development branch: `fix/context-intent-v3-mixed-state-20260915`

The post-V7 fixes are:

1. First-person `moved from X to Y` / `me he mudado de X a Y` is resolved as an exclusive location replacement before goal-lifecycle extraction.
2. Literal verification/test phrase commands using store/save/keep/remember are stored as inert note data before semantic fallback.
3. Goal lifecycle scaffolding is removed only at lexical boundaries, preventing short tokens from corrupting ordinary words.
4. A reschedule mutates due time while preserving the canonical goal description/identity.

Regression tests use entities and wording different from V7. The development deterministic suite reached `202 passed, 1 deselected` after the fixes, then `203 passed, 1 deselected` after V8 structural freeze, and `208 passed, 1 deselected` after V8 runner tests. The deselected test is the old V7 assertion that the current tree must match the V7 frozen manifest; a new test explicitly verifies that V7 preflight rejects this modified candidate.

## Frozen V8

Frozen candidate branch: `freeze/v8-context-intent-v3-20260915`

Exact revision: `ea0f499dc1415c77d49f1bc2fa5495d1a2cf7305`

Frozen bank: `src/infinito3/trajectory_holdout_v8_cases.py`

Git blob SHA-1: `a7dc174fa173ef248cb732144f6a1e319dd23fc6`

Structure:

- 4 trajectories
- 122 user turns
- 28 scored probes
- 4 empty-context controls
- explicit closure audit
- literal-data audit
- at least two reschedule-identity audits
- new entities, dates, distractors and wording relative to V7

V8 has not been replayed against the candidate and has made zero live provider calls.

## Runner and protocol

Runner: `scripts/run_infinito3_trajectory_holdout_v8.py`

Manifest: `docs/INFINITO_3_V8_RUNNER_MANIFEST.json`

Protocol revision: `v8_r1_mixed_state_boundaries`

The runner is manual-only and fail-closed. It requires:

- explicit `--authorize-live-v8`
- exact `--expected-revision ea0f499dc1415c77d49f1bc2fa5495d1a2cf7305`
- clean worktree
- configured `OPENAI_API_KEY`
- frozen OpenAI SDK version
- frozen file/blob checks
- full request/response audit
- manual semantic/final-state review after completion

A one-shot GitHub Actions workflow is staged at `.github/workflows/infinito3-v8-live-once.yml` on the development branch. It checks out the exact frozen V8 revision. There is deliberately no `.github/infinito3-v8-live-trigger` file yet, so no live V8 execution is authorized or scheduled.

## Next action

Obtain explicit authorization specifically for the first V8 live execution. Then create the one-shot trigger, run V8 exactly once, preserve the artifact, audit every failing/partial probe and final state, and only then decide beta readiness. Do not merge to `master` or `infinito-3.0` before that decision.
