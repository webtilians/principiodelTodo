# Arena analyst MVP protocol — 2026-11-10

Frozen before implementation/inference. Base: 664f4554739438b1a7691de8e7c64a27dfaaf9fc.

Question: can a local LLM turn synthetic Arena evidence into a grounded hypothesis and a minimal permission repair, with independent deterministic verification?

This is an engineering demonstration on an already known planted fault, NOT a learning experiment or a new vulnerability claim. No production changes, accounts, credentials or remote execution.

Two arms: no incident memory; INFINITO local incident memory. Identical model, seed, maximum generation tokens, trace, code and source retrieval. Memory arm receives a prior synthetic observation; baseline has an empty memory slot. Record actual input/output usage (inputs need not have identical token counts). Do not infer a memory benefit from these two demonstrations or tune prompts after seeing results without reporting it.

Use levels-off, stripped of fault labels, explanations, replay and outcome. Supply a read-only code snapshot with the deliberate permission defect. Retrieve reviewed source summaries from a versioned catalog, never arbitrary live pages. Save hashes of inputs, model identity, prompt, response, source IDs, timings and validation failures. Other structured traces can be analyzed but remain unverified.

Gate: LLM response must cite existing evidence and catalog IDs, identify remote_apps.allowed, propose restoring its read/add checks, and propose regression cases. A separate deterministic verifier rejects all but an AST-identical restoration of the trusted permission function. It never executes model-generated Python. In fresh synthetic Arena clones, the faulty version must admit A10/A11/A13/A14 and the restored version must stop them while legitimate operations remain permitted. Test current Arena separately. Invalid responses are failures, never silently replaced with a canned answer.

Record incident, hypothesis, failure and verified defence separately in a dedicated local INFINITO database with provenance. Do not write raw logs, external instructions, or model text into personal memory. All retrieved text is untrusted, including memory. The LLM has no tools, filesystem writes, shell, deployment or memory-write capability. This contains actions; it does not prove immunity to semantic prompt injection.

If either arm fails, report the raw outcome and keep that arm unverified. If the runtime is unavailable, report the limitation rather than claiming LLM execution. Independent evaluation means deterministic code outside the model, not another research group. Future general patch evaluation requires an OS-enforced sandbox; this MVP accepts one bounded restoration recipe only.
