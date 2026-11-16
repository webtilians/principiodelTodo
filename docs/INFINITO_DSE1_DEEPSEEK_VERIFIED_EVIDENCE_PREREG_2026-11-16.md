# DSE-1 · Pilot preregistration: verified experiences versus raw episodes

Date: 2026-11-16. Scope: **isolated synthetic study**, no production server, no user memories, no paid APIs and no changes to INFINITO cognition roles. Based on DeepSeek V4.1's task/environment/verifier discipline, not a reproduction of DeepSeek training.

## Research question
With one **frozen real Qwen model** and one new composition family, does a compact summary grounded in observed tool outcomes outperform raw episodic logs or no memory? This probes **inference-time use of evidence**, NOT learning of model weights, personal-memory advantage, or continual learning.

## Fixed design (before implementation)
- Training experience generation: execute real deterministic simulator tools; store 7 synthetic public observation traces (2 public-copy, 2 public-invert, 2 hidden-copy, 1 failed hidden-read). Never construct experiences from private oracle state; preserve responses and verification results.
- Tool names are opaque: `op_p`, `op_s`, `op_r`, `op_x`, `op_j` and `verify`. Simulator privately defines copy-public, select hidden, read hidden after select, flip working bit, keep working bit, and verify. The catalogue must not reveal semantics.
- **Heldout composition:** select hidden → read hidden → invert → verify. This complete plan is absent from training. Six target cases, with fixed public/hidden bit pairs: (0,0), (1,1), (0,1), (1,0), (0,0), (1,1); labels and task IDs vary. They repeat ONE task family, not six independent skill families.
- Arms, matched prompts and model: A = no references; B = INFINITO SQLite retrieved raw observation traces; C = the same records, transformed into **mechanically verified observed-effect summaries** with cited source IDs; no answer or target hidden bit in prompts. Engineered extraction is an explicit intervention and is not attributed to the LLM.
- All three arms get the same tool catalogue, user goal, `max_new_tokens=96`, `temperature=0`, 5-action limit, fixed sequence of cases. Fresh chat template invocation per case. No reuse of model conversational history.
- Runtime: Qwen/Qwen2.5-0.5B-Instruct, fixed revision when available, transformers CPU, no paid model API. Record installed versions, exact model commit when available, token counts, generated output, plans, verifier result and per-arm cost=0 API. If download/install fails, status **blocked**; do not substitute fake inference as a positive scientific result.
- Output contract: a JSON object containing provenance, each individual case, every raw plan (synthetic and safe), prompt token count and verifier result; markdown summary.
- Safety: isolate all tests, no secrets, no network calls apart from downloading public model dependencies, no real tools, no modifications to product memory.
- Tests: prohibit private state leakage to records or prompts, reject malformed plans and plans exceeding budget, prove failed hidden read and independently positive verification, verify heldout composition unseen.

## Precommitted gates
1. Integrity: real, identifiable model; all 18 arm×case prompts evaluated; no oracle data in model input, no target composition among training traces, identical model/settings, no use of result for prompt tuning.
2. Effect: report success of A, B, C separately. Exploratory signal if C wins at least **two more** cases than B and A each, with no integrity failure.
3. If a gate fails, report failure without rerunning or adjusting prompts on these same cases.
4. Even a positive exploratory signal is NOT a confirmation; requires independent new task families, multiple seeds, larger comparable models and tightly token-matched conditions.
5. User's local qwen3.8-motor 27B Q4 in Ollama is not available from this GitHub runner. No claim about that model.

Results and implementation are to be committed after this preregistration. Do not merge the draft into master.