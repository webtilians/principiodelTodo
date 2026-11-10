# Arena analyst phase 2 — competency before memory claims

**Status: code and tests prepared; no phase-2 LLM run or novel repair has been confirmed.**
This work builds on draft PR #163, preserving its four rejected Qwen 7B
responses, their prompts, and the supervised positive control unchanged.

## Why a second preflight is necessary

For the original `levels-off` fault, Qwen's four genuine answers were rejected:

- Run02 baseline: cited `remote_apps.py:allowed` instead of a provided path
  and confused tool names with permission levels (`LEVELS`).
- Run02 with memory: cited a function signature as a code path and failed to
  enforce the full read/add permission matrix.
- Run03 baseline: recognized the unconditional allow but incorrectly suggested
  that `read` may `remember`; its patch also broke permitted `all` operations.
- Run03 with memory: denied legitimate reads and allowed unreviewed behavior.
  It also proposed incorrect vault regression expectations.

These are **semantic failures**, not merely citation-format mistakes. The
original prompt was amended with the correct policy and exact function shape
after Run02. Run03 is a disclosed development retry, not independent proof.

## A different scenario, with the fix omitted

This preflight uses the existing synthetic **disconnect-off** recording, not
levels-off. The model is given a small set of synthetic network observations
(including the previously disconnected app's requests), a deliberately broken
in-memory excerpt of `remote_accounts.py`, and source summaries. The prompt
does **not** name the failing method, provide the SQL repair, or describe the
expected replacement AST. However, developers know the planted fault and the
independent referee knows the original method: this is a competency control,
**not a blind discovery of a previously unknown vulnerability**.

Run this preparation without Ollama:

```sh
python -m pytest tests/test_security_analyst_v3.py tests/test_security_competence_v3.py -q
python -m security_lab.analyst.competence --synthetic --prepare-only --out work/arena-competence-preflight
```

The output directory must not already exist. To run two genuine calls using an
already installed local Ollama/Qwen model, use a fresh output directory:

```sh
python -m security_lab.analyst.competence --synthetic --model lain-qwen7b:latest --out work/arena-competence-qwen7b
```

No model is downloaded. The local Ollama endpoint is fixed to loopback and
cannot be selected by model content. The operator confirms the input is
synthetic. No external account is accessed.

## What the memory arm really gets

A prior **reviewed synthetic authorization incident**, ID
`AUTHZ-LEVELS-001`, is recorded through local INFINITO `AgentMemory`. The
memory records its concrete symptom, observed cause, defensive principle, and
both permitted and forbidden regression cases. A gate checks the known
`levels-off` recording: the attacks must succeed with the fault and be
blocked in replay. The lesson is a fixed reviewed sentence: it cannot be
replaced by model prose or attacker-supplied logs. Its source hash identifies
the local recording, **not cryptographic attestation of an external event**.

The baseline receives the same trace, source summaries, code and prompt, but a
blank memory field. The memory arm gets the prior lesson retrieved via
`AgentMemory.recall`; the new case's answer is never stored before inference.
Input token counts will differ, and cache/order effects may still exist.
**Two calls cannot demonstrate a memory benefit**; a prospective balanced
multiple-case evaluation is needed after competency is demonstrated.

## Independent gates and restrictions

1. Closed response schema and citations to supplied event/source/code IDs.
2. At least one cited failing revocation event from the provided trace.
3. A patch AST identical to the trusted reviewed `Grants.drop_client` method.
4. Only after passing the AST gate: fresh Arena clones of the deliberately
   broken and healthy server, expecting A15/A15b to pass the broken clone and
   be blocked by the healthy one while other routes retain normal behavior.

Generated Python and generated tests are **never executed**. Only trusted
repository code runs; there is no general patch sandbox. A semantic
alternative repair, even if correct, is rejected by the exact AST gate.
A successful gate therefore shows that the model reproduced one known repair
contract, **not** that it can repair arbitrary vulnerabilities.

Production code, Caddy, public Arena, real users, personal memories and server
deployments are untouched. No model-provided URLs or tools are followed.
Traces and memory are untrusted model input. Any model call or failed test must
be reported as such; no inferred model improvement should be claimed.

## Decision after the preflight

- If schema or citations fail: inspect formatting and contextual sufficiency;
  do not count a malformed answer as a repaired vulnerability.
- If diagnosis is grounded but patch is rejected: separately report diagnosis
  competence and patch-generation failure. Do not loosen the AST gate.
- If one arm passes: repeat on multiple preregistered new synthetic cases
  and workers, with matched model versions, context budgets and condition
  order; include no-memory, relevant verified-memory and irrelevant-memory.
- Only if competence and causal memory benefit are shown should generic repair
  execution be introduced behind an OS-enforced sandbox and independent tests.

The baseline and memory arms' outputs should be archived with full prompts,
responses, model version, checksums and actual evaluation statuses.

## CI invocation

The repository's normal server workflow runs for pull requests targeting `master`.
The phase-2 PR is stacked on PR #163; GitHub Actions tests must be reported
against the exact head commit, and cannot be inferred from PR #163's green checks.
This document is not an LLM result and no Qwen success is claimed.
