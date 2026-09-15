# INFINITO 3.0 — V10 Freeze Boundary — 2026-09-15

## Purpose

V10 is the first post-V9 holdout for the typed Cognitive Query Planner boundary.
It is frozen only after the following post-V9 families were implemented and made
deterministically green:

1. deterministic multi-slot query decomposition and explicit daypart windows;
2. semantic membership separated from deterministic temporal ordering;
3. unique literal-data read fast path;
4. refactoring of ContextIntent into an auditable typed CognitiveQueryPlan without
   changing the existing interpretation contract.

V9 and all earlier live banks are development/diagnostic evidence for this candidate.
V10 uses new entities, dates, wording and distractors. Because its tested architectural
families were chosen after earlier diagnostics, V10 must be described as a post-diagnostic
holdout, not as a statistically blind benchmark.

## Frozen evaluation shape

- 4 independent full trajectories
- 122 user turns
- 28 evaluation probes
- 4 strict empty-personal-context controls
- same answer gate: >= 0.85
- same context gate: >= 0.90
- zero automatic merges
- manual semantic, planner, literal-grounding and final-state audit required

The bank source is `src/infinito3/trajectory_holdout_v10_cases.py` with Git blob SHA-1
`fea6f27c3f730f1797f652b6cc82f249a3663e11`.

## Planner boundaries explicitly exercised

V10 contains probes for:

- a five-slot profile read;
- current-state replacement and immediate predecessor reads;
- an explicit Tuesday-through-Friday calendar range;
- a `morning` window containing a 12:35 commitment;
- open/closed goal state and temporal canonicalization after rescheduling;
- semantic preference membership followed by deterministic `ORDER_LATEST`;
- historical preference retractions with new wording;
- a uniquely retrieved literal datum returned through the deterministic fast path;
- a residence move that must not be interpreted as a goal reschedule;
- four self-contained arithmetic controls that must receive zero personal context.

## Execution protocol

`scripts/run_infinito3_trajectory_holdout_v10.py` is manual-only.

Its default command performs preflight and does not import or replay the V10 bank. A live
run requires all of the following:

- explicit `--authorize-live-v10`;
- exact `--expected-revision` equal to `HEAD`;
- a clean worktree;
- the frozen OpenAI SDK version;
- repository-provided `OPENAI_API_KEY`;
- all manifest Git-blob hashes unchanged.

There are no automatic provider retries. An incomplete run preserves evidence and is not
authority for an automatic whole-run retry.

## Deterministic status before freeze

The full applicable deterministic suite passed with:

- 245 passed
- 3 deselected
- 0 failed

The three deselected tests are the acceptance guards whose purpose is to require exact old
V7, V8 and V9 frozen candidates; a post-V9 candidate must not satisfy those old hashes.

No V10 live provider execution has occurred at freeze time.

## Integration boundary

A successful workflow or automatic score never authorizes integration. V10 remains
`NOT_APPROVED` until its single authorized live execution, probe-by-probe manual audit and
final-state review are complete. No merge to `master` or `infinito-3.0` is authorized by
this document.
