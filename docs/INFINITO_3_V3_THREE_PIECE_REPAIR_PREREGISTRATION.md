# INFINITO 3.0 — V3 three-piece repair preregistration

This evaluation reuses the **exact frozen V3 bank** from commit `cbb05b9a803a6d0ed021f5790d43d6a87e45f426` and blob `44262e3a1da91806d09352a9d84bf69db76178f8`.

The V3 cases and expectations are not modified for this repair run.

## Baseline to beat

Original Temporal Cognition V3 result:

- 141 user turns
- 28 probes
- W/T/L: 13 / 12 / 3
- mean cognitive answer score: 0.5729
- mean context score: 0.7426
- mean answer lift: +0.1801
- final open goals: 4

## Changes under test

1. `SemanticCognitiveEventExtractor`: rule fast path plus a closed-schema semantic fallback for unseen durable-state phrasing and normalized lifecycle targets.
2. `StateAwareGoalResolver`: lifecycle matching from semantic identity, lexical/entity overlap, temporal evidence and current open-goal state.
3. `StructuredStateRetriever` + `StructuredTemporalContextBuilder`: authoritative current slots and lineage are retrieved before vector ranking, and historical relations are rendered explicitly.

## Repair criteria

Call the V3 repair **successful enough to justify a fresh unseen V4** if all are true:

- mean cognitive answer score >= 0.70;
- mean context score >= 0.85;
- mean answer lift > +0.20;
- baseline wins <= 3;
- final open goals <= 2, with manual inspection distinguishing intentional from stale goals.

Call it **strong V3 repair** if, in addition:

- mean cognitive answer score >= 0.78;
- mean context score >= 0.90;
- baseline wins <= 1;
- the three original V3 loss families no longer reproduce.

Do not integrate into `infinito-3.0` from V3 alone even if the repair passes. V3 has informed the design of these changes. A new frozen V4 is required before integration.

## Cost accounting

The report must expose separately:

- main answer-model tokens;
- semantic reranker calls/tokens;
- semantic event-extractor calls/tokens;
- structured query-planner calls/tokens;
- combined effective cognitive token overhead versus baseline.

A quality gain is not treated as free if it comes from hidden auxiliary model calls.
