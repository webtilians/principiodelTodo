# INFINITO 3.0 — Long-horizon held-out evaluation

## Why this benchmark exists

The original INFINITO 3.0 evaluation banks use paired probes with controlled visible history. That is useful for isolating memory/context effects, but it does not test a conversation that evolves independently for many turns.

This document records a separate **frozen long-horizon held-out bank**. The bank was authored before the fixes described below and was not edited to make those fixes pass.

Configuration used for the final runs:

- model: `gpt-5.6-luna`
- reasoning effort: `none`
- semantic embeddings: `text-embedding-3-small`
- three independent trajectories
- 58 user turns total
- seven scored probes
- short-term history limit: six turns
- isolated in-memory SQLite cognitive state per trajectory

The held-out bank covers:

- dense, semantically interleaved preferences
- creative/artistic semantic facets
- water-related semantic facets
- weekday language
- weekend queries
- explicit Spanish calendar dates
- goal cancellation and stale-memory suppression
- preference revocation after noise

The suite lives in `src/infinito3/trajectory_holdout_cases.py` and should remain frozen when evaluating changes against these results.

## First frozen held-out result

The first run was useful specifically because it was **not perfect**.

- cognitive wins: 7 / 7 probes
- mean cognitive answer score: about **0.905**
- mean cognitive context score: about **0.778**

It exposed three architectural defects:

1. semantic embeddings alone could not reliably separate dense preference facets;
2. human calendar language such as `viernes`, `sábado`, `este fin de semana` and `25 de septiembre` was not represented consistently between goal creation and retrieval;
3. a cancelled goal could disappear from the goal engine while its original episodic memory leaked back into context.

The preference-revocation trajectory already behaved correctly and was not changed merely because other parts of the bank failed.

## Calendar and lifecycle fixes

A shared deterministic temporal layer was introduced so goal creation and later retrieval interpret calendar language consistently.

The Context Builder now distinguishes temporal **framing** from the actual date being queried. For example:

- `Hoy es 18 de septiembre. ¿Qué tarea futura tengo programada?` uses September 18 as framing of the simulated present, not as the requested target date;
- `¿Qué tengo programado para el 25 de septiembre?` does request September 25 explicitly.

Goal state is authoritative for commitment queries, preventing a cancelled or completed goal from being resurrected from its duplicate episodic memory.

These behaviors are covered by deterministic regression tests.

## Why an LLM reranker was added

The embedding-only Context Builder still failed in one important way: similarity is useful for retrieval, but it is not always a good final classifier of membership in a semantic facet.

Two held-out examples made that distinction visible.

### Creative/artistic query

Stored preferences included:

- painting with watercolors
- ceramics
- night photography
- chess
- curry
- running
- kayak
- other unrelated preferences

For the query asking for **all creative or artistic hobbies**, embedding-only selection returned watercolors and ceramics but omitted night photography.

### Water-related query

For the query asking for activities related to water, embedding-only retrieval found the three true members — swimming, diving and kayak — but also admitted watercolors and ceramics into the final context.

A second-stage semantic membership reranker was therefore added behind an injectable interface. It only receives already-retrieved candidate facts; candidate text is treated as untrusted data, not instructions.

The first implementation returned full UUIDs and occasionally exhausted its output budget. It was replaced with compact integer indices such as:

```json
{"selected":[0,2,5]}
```

A three-query live diagnostic then classified creative, water and sports candidate sets correctly in **3 / 3** cases using 592 total reranker tokens.

## Frozen held-out ablation

The decisive comparison used the same frozen 58-turn / 7-probe bank.

| Configuration | Cognitive answer | Context score | Reranker calls | Reranker tokens | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Reranker OFF | 0.976 | 0.968 | 0 | 0 | Creative recall incomplete; water context noisy |
| Reranker always ON for eligible multi-value facets | 1.000 | 1.000 | 3 | 729 | Perfect held-out result, but one call was unnecessary |
| **Uncertainty-gated reranker** | **1.000** | **1.000** | **2** | **535** | Same quality with fewer semantic calls |

The gated run also produced **7 cognitive wins / 0 ties / 0 baseline wins** with all seven probes scoring 1.0 for both answer and context.

### Probe-level difference with reranker OFF

The two measurable degradations were concentrated in the dense semantic-facet trajectory.

**Creative facet**

- embedding-only answer score: **0.833**
- embedding-only context score: **0.889**
- selected context: watercolors + ceramics
- missing true member: night photography
- gated reranker result: watercolors + ceramics + night photography

**Water facet**

- embedding-only answer score: **1.000**
- embedding-only context score: **0.889**
- embedding-only context contained swimming + diving + kayak **plus watercolors + ceramics**
- gated reranker result: only swimming + diving + kayak

The sports preference probe was already perfect without reranking. The gated policy therefore correctly skipped it.

## Uncertainty gate

The reranker is no longer called for every multi-value semantic query.

The embedding stage runs first. Escalation occurs only when its own candidate geometry signals uncertainty, without using a domain dictionary:

1. **dense competition:** the number of rejected same-predicate candidates is greater than the number selected by embeddings; or
2. **flat embedding evidence:** at least five candidates are all retained and their semantic scores are nearly flat, so embeddings did not provide a useful membership boundary.

This rule is covered by synthetic deterministic tests for dense competition, balanced small cohorts, flat embeddings and structured full cohorts.

On the frozen held-out bank:

- creative: 11 candidates, embeddings selected 2 → reranker called → selected 3 correct facts;
- water: 11 candidates, embeddings selected 5 → reranker called → selected 3 correct facts;
- sports: four candidates, embeddings selected the two correct facts → reranker skipped.

## Cost interpretation

The explicit semantic reranker overhead is the cleanest cost signal because the main response model is stochastic and its token totals vary slightly between independent runs.

- always-on eligible reranking: **729 reranker tokens / 3 calls**
- uncertainty-gated reranking: **535 reranker tokens / 2 calls**
- reduction: **194 tokens, 26.6% fewer reranker tokens and 33.3% fewer reranker calls**

The final gated run used 18,696 main cognitive-provider tokens plus 535 reranker tokens, for 19,231 effective cognitive tokens. The exact difference versus other runs should not be interpreted as a deterministic cost delta because assistant generations differ stochastically; the explicit 535-token reranker counter is the auditable added cost.

## Current deterministic validation

After the calendar/lifecycle, compact-reranker and uncertainty-gate work, the branch passes **101 deterministic INFINITO 3.0 tests**.

The new live result is evidence that the current architecture can maintain and retrieve state across dozens of independent turns while handling semantic facets, calendar language, cancellation and preference revision. It is **not** evidence of AGI, consciousness, statistical significance or broad real-world reliability.

## Remaining scientific limitations

- only three long-horizon held-out trajectories and seven scored probes;
- one/few stochastic repetitions per configuration;
- normalized substring scoring is intentionally limited;
- no independent semantic/human judge yet;
- embedding and response API costs should be reported separately;
- the uncertainty gate was designed after observing the first held-out failures, so it requires a new unseen trajectory bank before being considered broadly validated;
- restart persistence, large real databases and much longer trajectories still need dedicated experiments.

## Next experiment

Do **not** tune the current frozen bank further.

The next useful experiment is a second unseen long-horizon bank, authored before inspecting outputs, followed by repeated runs. It should test new semantic facets, several simultaneous goals, cancellation/completion ambiguity, cross-language updates, contradictions separated by many turns, and substantially larger noisy memory populations.

After that, component ablations should compare:

- LLM only
- naive semantic RAG
- persistent memory only
- memory + goals
- full INFINITO with uncertainty-gated semantic reranking

The target metric should be quality per effective provider token, not accuracy alone.
