# INFINITO 3.0 — V7 runner freeze

ContextIntent v2 and its V7 evaluation protocol are frozen for the next external
check. No V7 model/provider execution has occurred.

Frozen evaluation ref: `freeze/v7-context-intent-v2-20260915`
Frozen revision: `ab6678b0cbde899226535514bd7f24e67bd3b4cb`
Development branch: `feat/context-intent-v2-20260915`
V7 bank commit: `ba05e38a84e67f557fafb20e76229cb6607ace91`
V7 source SHA256: `0ea8d349d9789cff2f8dbef0b3d53d73bc061c7e839a7ecd250fd2df3596f3f0`
Runner manifest: `docs/INFINITO_3_V7_RUNNER_MANIFEST.json`

At the frozen evaluation revision, GitHub Actions ran the complete deterministic
suite successfully: 198 passed. Runner-specific tests use synthetic data and do
not replay V7. The default V7 runner command performs preflight only and returns
before importing the V7 suite or constructing a provider client.

The live configuration retains the R2 provider budgets and quality gates:
- answer model: `gpt-5.6-luna`, reasoning `none`
- answer output floor: 1024
- event output floor: 2048
- reranker output floor: 512
- answer gate: >= 0.85
- context gate: >= 0.90
- strict empty-context controls: 4
- no automatic merge; manual semantic and final-state audit required

V6 R2 run `34988210410` is explicitly recorded as development evidence used to
choose the architectural families addressed by ContextIntent v2. V7 uses new
entities and formulations, but it is correctly described as a post-diagnostic
holdout rather than a statistically blind benchmark.

A live V7 execution requires explicit user authorization. If authorized, the
launcher must check out exactly revision
`ab6678b0cbde899226535514bd7f24e67bd3b4cb`, run deterministic tests and V7
preflight first, then execute at most one authorized attempt. Any incomplete run
must preserve partial evidence and must not retry automatically.

No merge to `master` or `infinito-3.0` is authorized by this freeze.
