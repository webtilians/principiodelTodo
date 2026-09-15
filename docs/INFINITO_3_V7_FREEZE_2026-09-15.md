# INFINITO 3.0 — V7 freeze

V7 was authored and frozen on 2026-09-15 only after ContextIntent v2 reached a
green deterministic suite. It has not been executed against a model or candidate.

Candidate branch: `feat/context-intent-v2-20260915`.
Candidate predecessor: V6 R2 revision `befa7636ea2db4e1a70ee235590743dedf11ac8b`.
Frozen bank: `src/infinito3/trajectory_holdout_v7_cases.py`.
Frozen source SHA256: `0ea8d349d9789cff2f8dbef0b3d53d73bc061c7e839a7ecd250fd2df3596f3f0`.

Structure:
- 4 trajectories
- 122 user turns
- 28 scored probes
- 4 strict standalone/empty-context controls
- explicit closure and literal-data audits
- history limit <= 5 for every trajectory

V7 deliberately uses entities, dates, activities, commitments and distractors that
do not occur in V6. Its formulations test the architectural families motivated by
V6 (compositional facts, temporal lineage, schedule/lifecycle intent, historical
preferences, literal-data aliases and standalone isolation) without copying V6
probe strings.

This is a post-diagnostic holdout, not a claim of a statistically blind benchmark:
the architectural families were selected after inspecting V6. V6 must therefore
remain development evidence for ContextIntent v2, and any V7 result must be
reported with this provenance.

The V7 file is immutable after this freeze. Any edit to the bank changes its hash
and creates a new evaluation version. Structural/hash tests may import and inspect
it, but may not replay turns through INFINITO or make provider calls.

No V7 live run is authorized by this freeze. A live run requires a separately
frozen runner/configuration and explicit user authorization. No merge to `master`
or `infinito-3.0` is authorized.
