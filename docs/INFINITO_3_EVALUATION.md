# INFINITO 3.0 — Evaluation Harness

## Purpose

Milestone 5 turns the cognitive loop into something measurable.

The harness runs fixed scenarios against the same LLM adapter/model in two conditions:

- **baseline**: normal short-term conversation, no cognitive retrieval
- **cognitive**: the same visible pre-probe history plus INFINITO memory, goals and Context Builder

Every scenario receives a fresh `CognitiveLoop` from a factory, so memory and goals cannot leak between tests.

## Evaluation design

A scenario has two phases:

1. **setup turns** build the hidden cognitive state.
2. a **probe** is evaluated through `CognitiveLoop.compare()`.

The final A/B probe therefore has the same visible short-term history in both variants. The difference being measured is the hidden cognitive state supplied by INFINITO.

This is a paired-probe benchmark. It is deliberately different from a full independent-trajectory benchmark where baseline and cognitive agents generate different conversations for many turns. Both are useful, but they answer different questions.

## Deterministic metrics

`DeterministicEvaluator` uses normalized substring assertions. This is not presented as a complete semantic quality metric. Its purpose is to make the first benchmark cheap, stable and auditable.

Each scenario may define:

- required phrases in the answer
- forbidden phrases in the answer
- required phrases in the ContextPacket
- forbidden phrases in the ContextPacket
- required context sources such as `goal` or `user_model`

The harness reports answer recall, forbidden-answer rate, answer score, context recall, forbidden-context rate, source coverage, context score, context-token cost, latency, provider token usage when available, baseline-vs-cognitive lift, wins/ties/losses, and aggregates by scenario tag.

Forbidden-answer checks are useful for known contradiction leakage. They are not a general hallucination detector.

## Optional judge

`PairwiseJudge` is a plug-in boundary for richer evaluation. A judge may be a human annotation service, deterministic domain-specific scorer, separate LLM judge, or judge ensemble. The core harness does not require a judge, and CI never depends on an external model.

## Scenario banks

`standard_evaluation_suite()` contains four fixed smoke-test scenarios: long-term identity, contradiction resolution, goal continuity and relevance filtering.

`extended_evaluation_suite()` contains 20 scenarios covering long-term user facts, reinforcement, exclusive-fact supersession, multiple preferences, goals, tight budgets, controls, semantic paraphrases, cross-lingual retrieval, prompt hygiene and contradiction under noise.

`precision_evaluation_suite()` is a separate frozen 12-case bank authored after the original 20-case benchmark. It deliberately probes multi-value requests, multiple requested predicates, irrelevant urgent goals, near distractors, cross-language queries, arithmetic negative controls and small budgets.

`precision_validation_suite()` contains eight additional cases authored after the first precision fix, including name+age, city+bike, unusual preferences, profile+goal combinations and short English profile queries. It exists specifically to reduce the risk of tuning only to the first precision bank.

## Live experiment history

All live runs below used `gpt-5.6-luna`, reasoning effort `none`, isolated in-memory SQLite per scenario and the same paired-probe methodology.

| Run | Suite | Retrieval | Wins / ties / losses | Baseline score | Cognitive score | Mean lift | Context score | Context tokens | Provider token delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial smoke test | 4 standard | hash | 4 / 0 / 0 | 0.125 | 1.000 | +0.875 | 1.000 | 50.75 | +61 |
| Extended baseline | 20 extended | hash | 14 / 6 / 0 | 0.175 | 0.800 | +0.625 | 0.778 | 51.25 | +54.45 |
| Semantic retrieval | 20 extended | `text-embedding-3-small` | 15 / 5 / 0 | 0.175 | 0.850 | +0.675 | 0.838 | 56.25 | +67 |
| After benchmark-derived fixes | 20 extended | `text-embedding-3-small` | 18 / 2 / 0 | 0.175 | 1.000 | +0.825 | 0.938 | 56.0 | +68 |
| Adaptive Context Builder precision | 20 extended | `text-embedding-3-small` | 18 / 2 / 0 | 0.175 | 1.000 | +0.825 | 1.000 | 46.95 | +55.6 |
| Generalized Context Builder | 20 extended | `text-embedding-3-small` | **18 / 2 / 0** | **0.175** | **1.000** | **+0.825** | **1.000** | **46.95** | **+55.35** |
| Generalized Context Builder | 12 precision | `text-embedding-3-small` | **11 / 1 / 0** | **0.083** | **1.000** | **+0.917** | **1.000** | **47.83** | **+54.17** |
| Generalized Context Builder | 8 validation | `text-embedding-3-small` | **7 / 1 / 0** | **0.125** | **1.000** | **+0.875** | **1.000** | **49.25** | **+62.75** |

Across the three non-overlapping current banks (40 scenarios), the generalized policy produced **36 wins / 4 ties / 0 losses**, with cognitive answer score **1.000** and context score **1.000** in every bank. The four ties are controls where cognitive state should not create an artificial advantage.

## What the experiments exposed

### 1. Semantic retrieval matters

The hash embedding baseline failed a paraphrase where the stored memory was `Me gusta el ciclismo de montaña.` and the probe was `¿Qué deporte practico?`. Replacing only the embedding backend with `text-embedding-3-small` recovered the memory and the model answered correctly.

The same semantic backend also retrieved a Spanish memory for an English probe. The first deterministic oracle incorrectly marked `You like downhill mountain biking.` as a failure because it required the literal Spanish word `descenso`. The oracle was corrected without relaxing the requirement that the original Spanish evidence be present in context.

### 2. The memory gate had observable blind spots

Direct structured facts such as `Mi bici es...`, `Mi color favorito es...` and explicit age were not originally persisted. The transparent rule baseline now recognizes these as structured `USER_MODEL` facts.

### 3. Interrogative probes could create fake goals

A probe such as `¿Qué tengo que hacer mañana?` originally matched `tengo que` and created an artificial goal from the question itself. `SimpleGoalEngine` now rejects information-seeking interrogatives while still accepting genuine requests such as `¿Puedes recordarme mañana llamar al banco?`.

### 4. Context precision required a separate experiment

The first 20-case precision optimization raised context score from **0.938 to 1.000** and reduced mean context from **56.0 to 46.95 tokens** without changing the answer result of **18 / 2 / 0**. It also deduplicated identical goal/memory evidence.

Because that policy was designed after inspecting the 20-case suite, a new 12-case precision bank was frozen before further tuning. That bank successfully broke assumptions in the first policy: some queries needed multiple facts simultaneously, some multi-value requests were phrased differently, urgent goals could be irrelevant, and self-contained questions needed no personal memory.

### 5. Domain dictionaries were not necessary

An intermediate precision fix used explicit topic vocabularies for examples such as music and food. It passed the observed cases but would not scale: a cognitive layer should not require a hand-maintained list for every possible topic.

`GeneralizedContextBuilder` replaces that dependency in the default `CognitiveEngine`. Its active selection policy uses:

- structured fact predicates for explicit profile facets such as name, location, age and bike
- semantic retrieval ordering for open-ended memories and preferences
- generic quantifier intent (`todos`, `todas`, `all`, `everything`, list/enumerate forms) to decide whether several values of a predicate are requested
- relevance-based goal filtering rather than urgency alone
- duplicate goal/memory removal
- negative-control suppression for self-contained arithmetic

The older `BalancedContextBuilder` remains available as a replaceable/reference implementation, but the default engine now uses `GeneralizedContextBuilder`.

The first generalized held-out run exposed one remaining generic intent bug: `todas mis preferencias` was not recognized as plural. The fix broadened the quantifier rule generically rather than adding a music-specific exception. A full rerun then restored answer and context score to **1.000** on extended, frozen precision and validation banks.

## Current validation state

The generalized branch was validated with **75 deterministic tests passing** plus the three live semantic-evaluation banks above. This is evidence for the current memory/context behaviors, not proof of general intelligence or broad real-world reliability.

Important limitations remain:

- the benchmark contains only 40 current live scenarios
- deterministic answer assertions are intentionally simple
- most configurations still have few stochastic repetitions
- paired probes do not model long independent agent trajectories
- there is not yet a semantic/human judge for nuanced answer quality
- real long-lived databases, process restarts and days-long user interaction need broader testing

## Example

```python
from openai import OpenAI

from src.infinito3 import (
    CognitiveEngine,
    CognitiveLoop,
    EvaluationHarness,
    OpenAIEmbeddingProvider,
    OpenAIResponsesAdapter,
    extended_evaluation_suite,
)

client = OpenAI()


def make_loop():
    engine = CognitiveEngine.persistent(
        ":memory:",
        embedding_provider=OpenAIEmbeddingProvider(client, model="text-embedding-3-small"),
    )
    adapter = OpenAIResponsesAdapter(client, model="gpt-5.6-luna")
    return CognitiveLoop(engine, adapter, history_limit=4)


report = EvaluationHarness(make_loop).run(extended_evaluation_suite())
print(report.to_markdown())
print(report.to_json())
```

## Next experiments

The next evaluation work should stop optimizing against these same 40 cases and move to broader held-out behavior: independent multi-turn trajectories, repeated runs with confidence intervals, larger noisy memory stores, persistence/restart tests, semantic/human pairwise judging, grounded hallucination, quality-per-token normalization and explicit component ablations for memory, goals and Context Builder.
