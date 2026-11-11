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

**Next step is not yet automated:** run each variant in a separate clean Claude or ChatGPT session, with the same model/version, tool budget and observation rules, without exposing the hidden sandbox implementation or the referee solution. Model calls and costs require the user's own independently configured services; this repository does not have permission to spend their API credits.

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

## Next executable milestone (P1)

Port the evidence verifier from F12-F with exact episode provenance; attach a read-only MCP tool adapter; run independent model sessions with and without verified evidence and log answers and refusals. Then evaluate combinations F10-P and F12-F before adding reflection. The pilot should only move to real users after P1/P2 safety and benefit checks.
