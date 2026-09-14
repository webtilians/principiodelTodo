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

`extended_evaluation_suite()` contains 20 scenarios covering:

- long-term user facts
- fact reinforcement
- exclusive-fact supersession
- multiple simultaneous preferences
- temporal goals
- multiple goals
- tight context budgets
- current-turn self-retrieval prevention
- empty-memory controls
- semantic paraphrases
- cross-lingual retrieval
- structured memory-gate facts
- instruction-like remembered text
- contradiction under irrelevant noise

The extended suite deliberately includes controls and cases expected to expose architectural gaps.

## Live experiment history

All live runs below used `gpt-5.6-luna`, reasoning effort `none`, isolated in-memory SQLite per scenario and the same paired-probe methodology.

| Run | Suite | Retrieval | Wins / ties / losses | Baseline score | Cognitive score | Mean lift | Context score | Context tokens | Provider token delta |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Initial smoke test | 4 standard | hash | 4 / 0 / 0 | 0.125 | 1.000 | +0.875 | 1.000 | 50.75 | +61 |
| Extended baseline | 20 extended | hash | 14 / 6 / 0 | 0.175 | 0.800 | +0.625 | 0.778 | 51.25 | +54.45 |
| Semantic retrieval | 20 extended | `text-embedding-3-small` | 15 / 5 / 0 | 0.175 | 0.850 | +0.675 | 0.838 | 56.25 | +67 |
| After benchmark-derived fixes | 20 extended | `text-embedding-3-small` | 18 / 2 / 0 | 0.175 | 1.000 | +0.825 | 0.938 | 56.0 | +68 |
| Adaptive Context Builder precision | 20 extended | `text-embedding-3-small` | **18 / 2 / 0** | **0.175** | **1.000** | **+0.825** | **1.000** | **46.95** | **+55.6** |

The final two ties are intentional controls: current-turn information that both arms can answer and an empty-memory negative control. No evaluated scenario is currently won by the baseline.

## What the extended benchmark exposed

### 1. Semantic retrieval matters

The hash embedding baseline failed a paraphrase where the stored memory was `Me gusta el ciclismo de montaña.` and the probe was `¿Qué deporte practico?`. Replacing only the embedding backend with `text-embedding-3-small` recovered the memory and the model answered correctly.

The same semantic backend also retrieved a Spanish memory for an English probe. The first deterministic oracle incorrectly marked the correct answer `You like downhill mountain biking.` as a failure because it required the literal Spanish word `descenso`. The benchmark oracle was corrected to score the English response while continuing to require the original Spanish evidence in the ContextPacket.

### 2. The memory gate had observable blind spots

Two direct structured facts were not originally persisted:

- `Mi bici es una Santa Cruz V10.`
- `Mi color favorito es azul petróleo.`

The transparent rule baseline was extended to recognize these structured user facts, plus explicit age facts. After the fix both cases score 1.0 and appear as `USER_MODEL` context.

### 3. Interrogative probes could create fake goals

A probe such as `¿Qué tengo que hacer mañana?` originally matched the phrase `tengo que` and created a second artificial goal from the question itself. This did not stop the model from finding the real goal, but it polluted cognitive state.

`SimpleGoalEngine` now rejects information-seeking interrogatives before goal creation while still accepting request forms such as `¿Puedes recordarme mañana llamar al banco?`. The live rerun confirms that only the real seeded goal appears in the final ContextPacket.

### 4. Context precision experiment

The previous 20-scenario run had perfect answer score but a context score of 0.938 because four scenarios selected irrelevant user-model memories. The precision experiment changed only `BalancedContextBuilder`; memory storage, gate, goal engine, embeddings and LLM configuration were held fixed.

The new selection policy is adaptive rather than a single global cutoff:

- singular queries normally keep the dominant memory candidate and only allow a very close high-score tie
- plural queries can keep multiple candidates when they represent the same structured predicate, preserving cases such as jazz + punk
- stable core facts such as name/location/bike are used as fallback only when the query actually asks for that fact
- a goal and an ordinary memory containing the same evidence are deduplicated before prompt construction
- diagnostics expose how many candidates were removed by the precision policy

Observed result:

- answer result stayed **18 / 2 / 0**
- cognitive answer score stayed **1.000**
- context score improved **0.938 → 1.000**
- mean ContextPacket size fell **56.0 → 46.95 tokens** (about **16.2% less context**)
- mean provider token delta fell **68.0 → 55.6** (about **18.2% less overhead**)

The four previously noisy scenarios now contain only the required evidence. The multi-value music case still retains both `jazz` and `punk`, so the improvement is not a trivial top-1 policy.

The goal scenarios also became smaller because duplicate goal/memory evidence was removed:

- `goal_tomorrow_time`: 2 context items → 1
- `goal_day_after_tomorrow`: 2 → 1
- `two_goals_recall`: 4 → 2

## Interpretation

The current result is an encouraging test of the architecture, not evidence that INFINITO improves general intelligence. The live benchmark is still small, its exact-answer evaluator is intentionally simple, and each configuration has only been run once. Model sampling can vary between runs.

What the experiments do show is narrower and useful: under controlled paired probes, a small amount of selected persistent state repeatedly restores information that the same model cannot answer from its visible short-term history alone. The experiments also successfully exposed concrete implementation defects, and isolated fixes improved both answer quality and context efficiency without creating baseline losses.

The latest precision result should also be treated cautiously because the policy was developed after inspecting this 20-case suite. A new held-out context-precision bank is required before claiming that the pruning rule generalizes rather than merely fitting these cases.

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

The next evaluation work should create a **held-out Context Builder precision suite** that was not used to design the adaptive pruning rule. It should include ambiguous singular queries, genuinely multi-fact answers, unrelated urgent goals, semantically close distractors and queries in Spanish/English. After that: independent multi-turn trajectories, repeated runs for confidence intervals, semantic/human pairwise judging, grounded hallucination, quality-per-token normalization and component ablations.
