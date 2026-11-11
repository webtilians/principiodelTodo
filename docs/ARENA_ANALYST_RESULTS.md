# Arena analyst MVP — observed results, 2026-11-10

**The supervised repair circuit works. The local Qwen analyst did not autonomously
produce an acceptable patch. No learning or memory improvement was demonstrated.**

PR: https://github.com/webtilians/principiodelTodo/pull/163 (draft, no merge/deploy).
Initial product base: `664f4554739438b1a7691de8e7c64a27dfaaf9fc`.
Protocol frozen in `245d3c3`; development amendment and implementation `90f9b58`.
The catalog was present during inference and is hashed in the manifests; commit
`57ac6e5` explicitly added it because the repository ignores JSON files by default.

## Review before modifications

Read repository guidance and README, Arena recorder and tests, permission module,
INFINITO AgentMemory API, server-test workflow and Caddy deployment template.
Existing Arena tests: **8 passed in 21.69 s**, on Windows/Python 3.11.9 in a new venv.
Arena uses temporary synthetic accounts and loopback server clones. Public Caddy
only exposes the existing index/run recording, with remaining traffic proxied to
8500. No live server connection, configuration or real account was needed.

## Actual LLM executions

Local Ollama `lain-qwen7b:latest`, manifest digest
`d04a742596b9` (full digest in evidence manifests), underlying weights blob
`65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423` as reported by
`ollama show --modelfile`. CPU execution. Seed 1729, temperature 0, output cap 1400,
context 8192. These identify the local installation; no independent weight audit.

| Run | Arm | Input tokens | Output tokens | Seconds | Outcome |
|---|---|---:|---:|---:|---|
| 02, original prompt | No memory | 4758 | 351 | 142.375 | Rejected: code citation format; patch also incorrect |
| 02, original prompt | INFINITO memory | 4936 | 280 | 47.094 | Rejected: code citation format; patch also incorrect |
| 03, policy-guided prompt | No memory | 4869 | 257 | 125.734 | Rejected: not the trusted restoration |
| 03, policy-guided prompt | INFINITO memory | 5049 | 253 | 42.625 | Rejected: not the trusted restoration |

All four are genuine local model calls. Raw prompts/responses and evaluations are
under [evidence](../security_lab/analyst/evidence/2026-11-10/). No rejected patch was
executed or silently repaired. Run 03 was a disclosed development retry after
observing run 02; do not pool it as independent confirmation. Memory held a fixed
prior synthetic observation, frozen before either arm's result. Token inputs differ;
warm cache/order may affect timings. Timing differences are not a memory benefit.

Run 01 failed before inference: `AgentMemory.recall` does not accept `max_chars`.
The adapter was fixed to bound the returned text. This was not a model outcome.

Example failures: the first baseline compared a tool name against LEVELS (which
contains level names); later candidates denied legitimate read/default operations.
Some generated test descriptions were themselves wrong. Schema and exact-AST gates
contained these proposals, but they do not establish reliable diagnosis by Qwen.

## Supervised positive control and independent evaluation

Codex authored `reviewed-proposal.json` with full task context and knowledge of the
correct function. It is explicitly **not blinded, not a successful Qwen call, and
not an arm of the memory comparison**. It restores the read/add membership checks
while preserving other behavior. The diff exists only as a proposed synthetic
repair artifact; the actual product already has these checks.

The independent deterministic verifier accepted its AST and ran fresh real-server
Arena clones. Results in [review.json](../security_lab/analyst/evidence/2026-11-10/supervised/review.json):

- Faulty clone: A10, A11, A13 and A14 got through.
- Restored clone: **15 attacks, zero through**.
- Six permission controls passed, including read/recall, add/remember and all/forget.
- Generated tests and arbitrary generated code were **not executed**.
- Incident, hypothesis and verified-defence metadata were saved with report hashes
  in a dedicated local INFINITO database, not the user's personal memory.

An initial supervised response cited CWE-862 although it was not in that retrieval
result and was rejected before lab execution. The citation was removed, not allowed
through the validator. The final supervised run includes both the proposed diff
and memory recording. No negative result was reclassified as success.

The verifier substitutes the reviewed trusted function after proving exact AST
identity; it does not apply arbitrary patches to an OS sandbox. This intentionally
narrow repair recipe supports the positive control, not general automatic repair.

## Tests executed

- Initial Arena baseline: 8 PASS.
- New boundary suite initially: 18 PASS.
- All product `tests/test_*v3.py`: **582 PASS, 1 SKIP**, 185.09 s.
- After adding the imported-trace boundary test: **19 analyst tests PASS**, latest
  targeted run 0.97 s. The added test ensures imported traces never use the demo
  verifier. Its first fixture omitted a relevant retrieval keyword and was corrected.
- Existing CI at `57ac6e5`: [server](https://github.com/webtilians/principiodelTodo/actions/runs/42837379794),
  [core](https://github.com/webtilians/principiodelTodo/actions/runs/42837379762), and
  [integration](https://github.com/webtilians/principiodelTodo/actions/runs/42837379787)
  all succeeded. Later PR commits have their own checks; consult the PR for final status.
- A broader `pytest tests` attempt stopped during collection with 23 missing-dependency
  errors (research/portal dependencies such as numpy and fastapi). It was not a pass;
  the relevant product suite above was run explicitly instead.

## Sources and scope

The permission recommendation follows [OWASP Authorization Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authorization_Cheat_Sheet.html).
The input-handling design follows privilege restriction, external-content separation
and output validation from [OWASP LLM01:2025](https://genai.owasp.org/llmrisk/llm01-prompt-injection/).
The catalog also contains reviewed pointers to [ASVS 5.0](https://asvs.owasp.org/),
[CWE-862](https://cwe.mitre.org/data/definitions/862.html) and [NIST CSF 2.0](https://www.nist.gov/cyberframework).
No compliance certification, comprehensive coverage or product CVE is claimed.

Prompt injection can still corrupt model prose. This implementation constrains
what that prose can cause: no tools, bounded read-only code retrieval, no arbitrary
URLs, no model-directed memory writes, closed output schema and no arbitrary code
execution. Hashes identify artifacts, not authenticity. The current trace format
has no cryptographic provenance verification.

## Pending before an autonomous analyst can be trusted

1. A model/provider that passes grounded diagnosis and repair tests on held-out
   cases; the installed Qwen configuration failed this development case.
2. General patch/test execution in an OS-enforced isolated worker. The current
   Python-level loopback guard is not sufficient for untrusted executable code.
3. Semantic citation/evidence validation beyond checking IDs and the one bounded
   repair recipe; signed trace ingestion when a relevant attested source exists.
4. Broader source retrieval and incident details with safe review/redaction,
   representative adversarial injection cases and a preregistered equal-budget,
   held-out memory study. No indiscriminate imports into personal memory.
5. Optional reviewed public presentation. This PR adds no public analyst endpoint
   and does not publish internal reports automatically.
