# INFINITO Experimental — P0 runnable pilot (11 November 2026)

This is the **first executable integration preflight** for [issue #165](https://github.com/webtilians/principiodelTodo/issues/165). It is deliberately separate from production and is not a claim that Claude, ChatGPT or Qwen learned anything.

## What was implemented

- Uses the **actual** `SQLiteCognitiveMemoryStore` from the INFINITO library, with an ephemeral in-memory database and a dedicated synthetic `project` marker.
- Executes **two disjoint synthetic tool episodes** against a local, deterministic toolbox. The reporter records observed inputs, outputs and **an independent end-state verification**.
- A third **failed episode** is stored as a negative control and excluded from the verified transitions.
- A deterministic breadth-first **positive control** composes a previously unobserved full path using steps from both successful episodes, and an independent sandbox referee executes and verifies that path.
- Exports four packets for future independent LLM runs (A no memory, B raw episodes, C verified effects, D effects + a subjective reflection); **export does not call a model or show the composed plan**.

This is a *minimal integration harness* inspired by the evidence/experience research, **not** a port of F10-P learned weights, F12-F full validator, E11 selector, or a real MCP integration. Code review of their compatibility is still required.

## Run the completed offline preflight

```bash
python -m experimental_pilot.pilot --preflight
python -m pytest -q tests/test_experimental_pilot_v3.py
```

Expected report identifies itself as `type=deterministic-positive-control`, `model_calls=0`, `real_users=0`, `real_mcp_calls=0`. A PASS means **only** that the recorder, SQLite storage, retrieval, composition control, and referee work in this synthetic exercise.

## Export packets for agent experiments

```bash
python -m experimental_pilot.pilot --export > /tmp/infinito-pilot-packets.json
```

**P1 automation is available below:** the runner executes each variant in a separate clean API request, with the same model/version, tool budget and observation rules, without exposing the hidden sandbox implementation or the referee solution. Model calls and costs require the user's own independently configured services; this repository does not have permission to spend their API credits.

The model's JSON response must look like:

```json
{"steps":["ingest","normalize","tag","deliver","verify"]}
```

**The example is a positive-control test answer**, not something to show to a model during evaluation. To score a local response saved as a JSON file:

```bash
python -m experimental_pilot.pilot --evaluate /tmp/agent-response.json
```

An answer must contain a list of tool names with no more than six actions. The last action must be `verify`. The referee uses the simulated tool state, not model self-assessment.

## Prohibited shortcuts and limitations

1. A valid CI run **is not** evidence of model learning or benefit for humans. We have not invoked any LLM in this stage.
2. No user memory, OAuth accounts, vault secrets, production database, network traffic or deployment. Synthetic and ephemeral only.
3. Do not compare variants unless independent sessions, equal context/tool budgets, identical model settings, logged tokens and blinded adjudication are implemented. D may require a padding control to equalize tokens.
4. The trusted recorder is a local in-process fixture, not cryptographic provenance. Do **not** import external records claiming `source=trusted-local-recorder` and treat them as verified. For real MCP use, sign/attest tool observations outside the model.
5. The deterministic composer is **a positive control**, not a trained planner. Models must produce their own plan, using only their assigned packet.
6. All historical experiment conclusions and reserved worlds/seeds remain unchanged.
7. Before any real writable MCP trial, validate project/account separation and the F10-S SQLite concurrency fix in the exact deployed code; use consent and granular authorization.

## Broader future milestone (outside this minimal P1)

Port the evidence verifier from F12-F with exact episode provenance; attach a read-only MCP tool adapter; run independent model sessions with and without verified evidence and log answers and refusals. Then evaluate combinations F10-P and F12-F before adding reflection. The pilot should only move to real users after P1/P2 safety and benefit checks.

## P1: minimal isolated API runner

P1 now automates the four exported conditions for OpenAI and Anthropic, using
only Python's standard library. P0 remains a deterministic positive control;
**no real model inference has been validated by this implementation**.

```bash
python -m experimental_pilot.runner --dry-run --output-dir work/pilot-p1
python -m pytest -q tests/test_experimental_pilot_v3.py
```

This creates `results.json` and `summary.md` with eight rows (provider/model ×
variant), prompt hashes, input sizes, token usage, parsed actions, errors and local
referee results. Fake clients reproducibly abstain (`steps=[]`), report zero real
model calls and zero token usage, and never use the network. Their scores measure
no model capability. Raw replies, HTTP headers and credentials are never saved.

Each condition gets a fresh client and exactly one user message containing only
the task and its own observations. No previous response, other variant, positive
control answer, referee code or feedback is supplied. Evaluation happens locally
after each reply, with no feedback or retry. Sources are sorted for reproducible
packet hashes. D uses the existing fixed subjective reflection, not a learned or
model-generated reflection.

### Live mode: separate explicit authorization required

Do not run live without the owner's explicit permission and spending cap.
Set these environment variables privately (never commit keys):

- `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- `OPENAI_MODEL`, `ANTHROPIC_MODEL`: explicit model/version identifiers
- `PILOT_MAX_USD`: positive finite maximum for the complete run
- `OPENAI_INPUT_USD_PER_MILLION`, `OPENAI_OUTPUT_USD_PER_MILLION`
- `ANTHROPIC_INPUT_USD_PER_MILLION`, `ANTHROPIC_OUTPUT_USD_PER_MILLION`

Rates must be verified upper bounds for the selected model and account. Only
after approval, use `python -m experimental_pilot.runner --live
--authorize-paid-api --output-dir work/pilot-p1-live` on one line.

All eight attempts are reserved before the first request using, per condition,
4096 input bytes + 1024 tokens of protocol allowance and 256 output tokens, at the
configured rates. Reservations are not recycled after errors. Missing, nonfinite,
nonpositive or insufficient budgets fail closed. This is a conservative local
estimate conditional on correct rates/token accounting, **not a provider billing
hard cap**; also configure a provider-side spending limit. No remote tools or
retry charges are permitted. Fixed HTTPS endpoints reject redirects; timeout 60s.
OpenAI uses Responses with `store=false`; Anthropic uses stateless Messages.
See the official [OpenAI Responses reference](https://developers.openai.com/api/reference/resources/responses/methods/create)
and [Anthropic Messages reference](https://platform.claude.com/docs/en/api/messages).

### Interpretation and limitations

The common budget is one request, at most 4096 prompt bytes, 256 output tokens and
six proposed actions per condition. Actual prompt lengths/token counts differ by
condition and provider; no artificial padding is added. Equal ceilings are not
identical spend, reasoning effort or tokenization. Provider default sampling is
used consistently within each model; results from real models need not repeat.
This single synthetic task, one sample per condition, is a plumbing pilot, not a
statistical study or proof of learning, transfer or memory benefit. Tool names
also give semantic clues. A strict JSON failure, refusal, truncation, HTTP error
or unknown tool is recorded as an error, without rescuing the answer. Transport
failures may still be billed and have unknown usage. Reported live `model_calls`
is the number of attempted conditions, not verified successful inferences.

No changes to production, real users, remote MCP or persistent user memories.
The broader provenance/MCP milestone above remains out of scope for this P1.
