# INFINITO 3.0 — Frozen V4 structured-state ablation

Frozen V4 cases: `ef689e8adece7aa265c236ac9f50db0ebac45469`

Suite SHA256: `8825f79527a36da808915f6b085fcee349e808da98879c8947ed807df5390979`

Semantic-extractor baseline: `9be85131cbabf257036bfb6742493ffe9f8d66d2`

Structured-state implementation under test: `42129d7c1926f182cb685cf637a2a3e9e2d4813a`

Artifact SHA256: `dc7e764ee07daabcc0c183adc51e69b42451d802fd866659df9463c9be97678e`

## Aggregate result

- 4 trajectories
- 130 user turns
- 29 probes
- W/T/L: **16 / 13 / 0**
- baseline answer score: **0.3764**
- cognitive answer score: **0.7624**
- answer lift: **+0.3859**
- context score: **0.8370**
- semantic event calls: **50**
- semantic event tokens: **23,609**
- skipped non-mutating requests: **73**
- semantic reviews: **1**
- reranker calls / tokens: **3 / 934**
- final active memories: **25**
- final open goals: **1**
- deterministic tests in paid workflow: **131 passed**

Compared with the first V4 semantic-extractor run, structured retrieval raises cognitive score from about 0.605 to 0.762, context from about 0.755 to 0.837, lift from about +0.232 to +0.386, and removes the only baseline win. Semantic extractor calls fall from 74 to 50 and extractor tokens from 34,976 to 23,609.

## Interpretation

The experiment supports three architectural changes:

1. structured profile retrieval by fact predicate rather than vector top-k alone;
2. explicit temporal rendering of predecessor evidence (`relation=immediately_previous`);
3. goal-query intent and calendar-range filtering independent of goal-domain vocabulary.

The remaining major weakness is now concentrated in preference state retrieval. Narrow or faceted preference questions still depend too heavily on ordinary semantic top-k, and historical preference retractions are not rendered as explicit state transitions.

Known scorer limitations remain: `Ghent` vs `Gante` and `informe anual` vs `annual report` can score as failures despite semantically correct answers. A five-item goal answer was also truncated by the probe output limit despite perfect goal context.

Decision: freeze this exact implementation as `baseline/structured-state-retrieval-v1-20260914` and improve preference-state retrieval before creating a new unseen held-out bank.
