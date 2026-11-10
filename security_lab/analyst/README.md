# Arena analyst MVP

Local, read-only LLM analysis of synthetic attack traces. The public `/security/arena/`
remains a static recording viewer. This package adds no HTTP route, credential,
service, production patch, scheduled job or deployment hook.

## Reproduce the demonstration

Use a fresh checkout and an isolated Python environment. Install the repository's
server test dependencies (`pip install -e ".[mcp,vault,dev]" uvicorn`). Install/start
Ollama with a Qwen-compatible local model separately; the runner never downloads
models or connects to a cloud endpoint. It uses numeric loopback, disables proxies
and refuses redirects. The model needs Qwen chat framing (the local demonstration
model has a raw template).

```sh
python -m pytest tests/test_security_arena_v3.py tests/test_security_analyst_v3.py -q
python -m security_lab.analyst --out work/analyst-demo --model lain-qwen7b:latest --synthetic --demo-levels
```

`--out` must not exist. `--synthetic` is an operator attestation that the input is
synthetic/public, not a secret detector. Never feed real account logs to this MVP.
The default trace is the existing Arena `run.json`. `--demo-levels` requires that
exact bundled recording and the `levels-off` round. It removes the planted-fault
description, outcome and replay from the model input. It replaces `allowed()` in
an **in-memory snapshot** with `return True`; source files stay untouched.

Two calls use the same model, seed, generation cap, trace, code and documentation.
One receives no incident memory. The other receives a fixed prior synthetic
observation retrieved through INFINITO `AgentMemory`, frozen before either call.
Input token lengths differ. There is no training or scientific improvement claim.

Outputs include exact prompts/responses, model digest inventory, input hashes,
token counts, source citations, proposed diff, independent verification and
comparison JSON. They are private local lab artifacts unless separately reviewed
for publication. Output files are created outside the public viewer.

## Boundaries and verification

The model gets strings only: no shell, filesystem, browser, memory writes or
deployment tools. Read-only code access is two host-allowlisted modules (the demo
supplies only `remote_apps.py`), with size limits and resolved-path containment.
Traces, repository content, source summaries and memories are untrusted data.
Instructions in any of them have no execution path. Raw model special tokens are
neutralized, output schema is closed, and evidence/source IDs must exist in the
input. These controls do **not** prove the model cannot be deceived semantically.

The independent verifier is deterministic, not a second LLM. It accepts only an
AST-identical copy of the reviewed `allowed()` implementation, including the
signature. Arbitrary Python, decorators, extra statements and alternative patches
are rejected without execution. The accepted function is equivalent to the trusted
restoration recipe, so the verifier invokes the trusted original, never `eval` or
`exec` of model output. This is **not** a general-purpose patch sandbox.

It starts fresh Arena clones with synthetic accounts, temporary storage and the
existing loopback guard. It requires the deliberately faulty clone to admit
A10/A11/A13/A14, the restored clone to stop all attacks, and positive/negative
permission controls to pass. `cast()` also exercises legitimate setup through the
real server. The guard is Python-level containment, not an OS network sandbox;
unreviewed code cannot be executed here. General patch execution needs an OS-level
isolated worker without host mounts, secrets or network, before extending this MVP.

LLM-proposed regression tests remain reviewable descriptions. The executed tests
are independent repository tests; we never claim generated tests were executed.
Citation existence is checked automatically; causal interpretation and whether a
source actually supports the model's prose still need review. Imported traces are
not cryptographically attested. A SHA-256 identifies bytes, not truth or authorship.

## Other traces

```json
{"format":"structured-v1","events":[{"id":"event-1","who":"synthetic-reader","door":"remember","request":"synthetic action","reply":"synthetic response","rule":"no_write","saw":"synthetic observation","held":false}]}
```

Use `--trace file.json` and omit `--demo-levels`. Imported `held` becomes
`reported_held`; it is never promoted into an independent verdict. The current
adapter is specialized to permission analysis and supplies current `remote_apps.py`.
Other causes may produce unsupported proposals. Imported traces remain unverified
even if the model proposes the known restoration. Exit 0 means both demo arms were
verified; exit 2 means at least one arm was rejected, failed or unverified. Operational
errors (unavailable model, malformed input, missing dependencies) are errors, not results.

## Catalog and memory

`catalog.json` is a versioned, small reviewed catalog, not an encyclopedia. Every
entry has a primary source URL, authority, source version, review date, next-review
date, tags and a short editorial summary. Retrieval ranks relevant tags, excludes
stale entries and hashes summaries. The hash covers the local summary, **not** the
remote page. Dates express a local review policy, not a guarantee that advice stays
valid. Refresh through review/PR; runtime never follows URLs found in logs or memory.

Included: OWASP Authorization Cheat Sheet, ASVS 5.0 pointer, LLM01:2025, MITRE
CWE-862, NIST CSF 2.0 pointer. ATT&CK, D3FEND and CVEs are not imported without a
relevant technique/defence/affected version. No CVE claim is made for a planted bug.

INFINITO uses a **new explicit local database**, ignoring personal-memory defaults.
It records incident, hypothesis, failure and verified-defence categories as separate
facts/assumptions, with report hashes for provenance. Model text and external source
instructions are never automatically persisted. This MVP records only fixed-vocabulary
metadata; detailed evidence stays in the corresponding report. Memory is never an
authority to execute a patch. Remote personal MCP writes are deliberately absent.

## Infrastructure review

Base `664f4554739438b1a7691de8e7c64a27dfaaf9fc`: Caddy explicitly serves only the
Arena index and `run.json`; other routes go to the existing loopback service on
8500. CI `server-tests.yml` installs MCP, cryptography and KeePass and collects
`tests/test_*v3.py`; the new test file follows this naming convention. No server
configuration or product files are changed by this PR.

See `docs/ARENA_ANALYST_PROTOCOL.md` for gates frozen before inference and
`docs/ARENA_ANALYST_RESULTS.md` for what was actually executed.
