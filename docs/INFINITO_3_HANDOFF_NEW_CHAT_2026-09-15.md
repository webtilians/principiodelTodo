# INFINITO 3.0 — New Chat Handoff (2026-09-15)

## Read this first

This file is the canonical handoff for continuing INFINITO 3.0 in a fresh ChatGPT conversation. Do **not** rely on conversational memory alone. Inspect the exact branches/commits below before changing code.

Repository: `webtilians/principiodelTodo`

## Non-negotiable rules

1. **Never merge to `master` unless the user explicitly asks.**
2. `master` is intentionally untouched at `9a2da8afe9d94f526e316b631344af3d82b45ed4`.
3. `infinito-3.0` is also intentionally untouched by the experimental temporal/semantic work and remains at `3e56a0e2579f2e8a1af70e2fa2a7d0e87faf6e2a`.
4. Frozen held-out banks must never be edited after their first live run. Fix algorithms, not exams.
5. Do not overclaim AGI, consciousness, or statistical significance. Report only what the frozen benchmarks support.
6. Keep expensive live workflows explicit/manual. Prefer deterministic tests before any paid run.

## Current best experimental implementation

The implementation used by the clean V5 held-out is:

`3309095c2e0df436ce1772ecbf77601189c50fe8`

It includes the event-sourced temporal architecture plus later preference-state/history work. The V5 runner reports this configuration as:

- `PreferenceStateContextBuilder`
- `SemanticCognitiveEventExtractor` (`hybrid_semantic_v1_1_gated`)
- `SemanticTemporalCognitiveState` with retraction tombstones
- `SemanticTemporalMemoryStore`
- `TemporalGoalEngine`
- uncertainty-gated semantic reranker
- OpenAI embeddings (`text-embedding-3-small`) in the live benchmark

Important branches around this implementation:

- `baseline/preference-history-resolution-v1-20260914`
- `baseline/preference-state-context-v1-20260914`
- `feat/preference-history-resolution-20260914`
- `feat/preference-state-retrieval-20260914`
- `eval/v4-preference-state-20260914`

Before making new changes, inspect these branches and the implementation commit rather than reconstructing the architecture from old chat prose.

## Architectural path that led here

The project evolved from text memory + semantic retrieval into an explicit cognitive state pipeline:

```text
User turn
  -> Safety
  -> CognitiveEventExtractor / SemanticCognitiveEventExtractor
  -> typed event schema
       ASSERT_FACT / REPLACE_FACT / RETRACT_FACT
       ASSERT_PREFERENCE / RETRACT_PREFERENCE
       CREATE_GOAL / COMPLETE_GOAL / CANCEL_GOAL / RESCHEDULE_GOAL
       STORE_NOTE
  -> SemanticTemporalCognitiveState
       versioning, valid_from/valid_to, lineage, tombstones
  -> SemanticTemporalMemoryStore + TemporalGoalEngine
  -> PreferenceStateContextBuilder / structured state retrieval
  -> semantic cohort + uncertainty-gated reranker when needed
  -> LLM
```

The important conceptual result is that long-horizon failures were often caused **before retrieval**: free-form language was not being converted into state mutations, or historical/current state was not explicitly represented. The current architecture treats current state, historical lineage, preferences, and goals as structured state rather than hoping embeddings infer everything later.

## Benchmark chronology

### Original generalized-context banks

Across the earlier standard/precision/validation banks, generalized context reached perfect deterministic context criteria on 40 unique cases. These were useful development banks, not independent evidence after later tuning.

### Long-horizon first bank

After goal/calendar fixes:

- 3 trajectories
- 79 user turns
- 12 probes
- 7 wins / 5 ties / 0 losses
- cognitive answer score = `1.0`
- context score = `1.0`

This bank was subsequently used to fix bugs, so do not use it as independent evidence of generalization.

### Frozen held-out V2

The second bank exposed major temporal/event failures. Adding `CognitiveEventExtractor + Temporal Cognitive State` improved the same frozen bank from roughly:

- `5 / 16 / 5`, cognitive ~`0.457`, context ~`0.599`

to:

- `14 / 10 / 2`, cognitive ~`0.755`, context ~`0.888`

This was strong evidence for the architectural direction, but the component was then developed using V2 evidence, so a new bank was required.

### Frozen V3

Fresh V3 (141 turns / 28 probes) with the temporal architecture initially produced:

- `13 / 12 / 3`
- cognitive `0.573`
- context `0.743`
- lift `+0.180`

This failed preregistered integration criteria and exposed language-coverage, goal-intent, and history-representation issues.

A hybrid Semantic Cognitive Event Extractor improved the same V3 to approximately:

- cognitive `0.644`
- context `0.808`
- lift `+0.234`

but cost was high and losses did not cleanly disappear. This motivated structured state retrieval and cheaper gating rather than more phrase-specific rules.

### Frozen V4

V4 was frozen before live execution and used new entities/phrasings. Initial semantic-extractor run:

- 4 trajectories
- 130 user turns
- 29 probes
- `11 / 17 / 1`
- cognitive `0.6054597701`
- context `0.7545977011`
- lift `+0.2316091954`
- semantic event calls `74`
- semantic event tokens `34,976`
- final open goals `1`

V4 showed the semantic extractor generalized somewhat, but the next bottlenecks were structured state query intent, historical relation rendering, preference-state retrieval, and cost. Subsequent experimental branches addressed those areas.

## Frozen V5 — clean independent test

### Freeze protocol

The V5 bank was frozen **before** its test harness, runner, workflow and trigger were added.

Frozen case commit:

`6761557ab7404ba02472bf1c86963de7d0429ece`

Frozen suite SHA256:

`8a0a4600d9545e63ffa5da77006ed324c0a98c50c41d41d57ed11f529ecfd222`

Bank shape:

- 4 independent trajectories
- 130 user turns
- 29 probes
- all probe labels begin with `v5 `
- history limits <= 5
- fresh entities, domains and paraphrases relative to V4

Freeze/test/runner/workflow order was:

1. `6761557a...` — **freeze unseen long-horizon V5 bank**
2. `1151adf2...` — validate frozen V5 bank shape/hash
3. `16bcfb97...` — add frozen V5 runner
4. `acf83f16...` — add frozen V5 workflow
5. `08d74962...` — trigger first frozen V5 held-out

Therefore V5 is a legitimate clean held-out with respect to the implementation used for its first run.

### First live V5 run

Workflow run: `34900050786`
Job: `104163574660`
Artifact ID: `10370866495`
Artifact SHA256: `ed89060d370cf04231c50ae591c1e8a33e600d629395cde7c3fa74aee1eab926`

Implementation commit evaluated:

`3309095c2e0df436ce1772ecbf77601189c50fe8`

Model configuration:

- answer model: `gpt-5.6-luna`
- semantic event model: `gpt-5.6-luna`
- embedding: `text-embedding-3-small`
- semantic reranker: `gpt-5.6-luna`

Result:

- trajectories: `4`
- user turns: `130`
- probes: `29`
- **wins/ties/losses = `20 / 9 / 0`**
- baseline score = `0.3333333333`
- **cognitive score = `0.8402298851`**
- **answer lift = `+0.5068965517`**
- **context score = `0.8310344828`**
- semantic event calls = `60`
- semantic event successes = `60`
- semantic event failures = `0`
- semantic events emitted = `45`
- non-mutating requests skipped locally = `60`
- semantic reviews = `0`
- semantic event input tokens = `22,958`
- semantic event output tokens = `4,150`
- semantic event total tokens = `27,108`
- reranker calls = `6`
- reranker tokens = `1,590`
- cognitive effective total tokens including event extractor = `73,548`
- effective delta vs baseline = `36,644`
- final active memories = `26`
- final open goals = `1`

The important scientific interpretation is: **V5 is the strongest independent evidence so far that the architecture generalizes beyond banks used for development.** It achieved zero baseline wins/losses against the cognitive arm and a large lift, but context quality is still ~0.83 rather than near-perfect and the semantic event extractor remains expensive.

Do not claim this proves general intelligence. It proves robust performance on this specific frozen long-horizon bank.

## What to do next

The next model should **not immediately create V6 and not immediately merge anything**. First:

1. Download/read the V5 artifact (`10370866495`) probe by probe.
2. Classify all 9 ties and any imperfect cognitive/context scores into:
   - scorer lexical false negative/positive
   - event extraction issue
   - state mutation / tombstone issue
   - goal lifecycle issue
   - structured state query-intent issue
   - historical-lineage rendering issue
   - preference-state retrieval issue
   - semantic cohort/reranker issue
   - answer-generation issue despite correct context
3. Inspect the single final open goal and determine whether it is legitimate or stale.
4. Compute per-trajectory V5 scores and identify which architecture family still dominates residual error.
5. Separately evaluate cost. 60 semantic-event calls / 27,108 tokens over 130 turns is still substantial. Find cheap deterministic/gated opportunities **without using V5 phrases as production special cases**.
6. Only after diagnosis, decide whether the current candidate is mature enough to integrate into `infinito-3.0` or whether one more architecture fix should be made on an experimental branch.
7. If code is changed using V5 evidence, V5 becomes a development bank. Any claim of new generalization then requires a brand-new V6 frozen before first live execution.

## Integration policy

Do not move `infinito-3.0` or `master` casually.

Current branch heads verified on 2026-09-15:

- `master`: `9a2da8afe9d94f526e316b631344af3d82b45ed4`
- `infinito-3.0`: `3e56a0e2579f2e8a1af70e2fa2a7d0e87faf6e2a`

The current best experimental evidence lives outside those branches. If integration is later justified, create/inspect a clean compare against `infinito-3.0`, run all deterministic tests, and keep the PR draft until explicitly approved.

## Suggested first message in a new chat

Paste this to the new model:

> Continue the INFINITO 3.0 research in `webtilians/principiodelTodo`. Read `docs/INFINITO_3_HANDOFF_NEW_CHAT_2026-09-15.md` from branch `handoff/infinito3-20260915` first and treat it as the canonical project state. Do not merge to `master` or `infinito-3.0`. Start by downloading and analyzing the first frozen V5 artifact probe-by-probe (workflow run `34900050786`, artifact `10370866495`) and classify the 9 ties / residual context errors before changing code.

## Safety / secrets

- Never print or commit API keys.
- GitHub Actions has a masked `OPENAI_API_KEY`; use it only through workflows.
- Keep paid runs explicit and minimize unnecessary baseline/model calls.
