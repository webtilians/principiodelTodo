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

## Standard suite

`standard_evaluation_suite()` currently contains four fixed scenarios:

1. `long_term_identity`: a user fact leaves short-term history and the probe checks whether INFINITO recovers it.
2. `contradiction_resolution`: the user changes an exclusive fact; the new value must be used and the superseded value suppressed.
3. `goal_continuity`: an active goal is no longer visible in chat history and must survive through GoalEngine.
4. `relevance_filter`: multiple user-model facts are stored but only the task-relevant one should enter the ContextPacket.

## Example

```python
from openai import OpenAI

from src.infinito3 import (
    CognitiveEngine,
    CognitiveLoop,
    EvaluationHarness,
    OpenAIResponsesAdapter,
    standard_evaluation_suite,
)


def make_loop():
    client = OpenAI()
    engine = CognitiveEngine.persistent(":memory:")
    adapter = OpenAIResponsesAdapter(client, model="<api-model-id>")
    return CognitiveLoop(engine, adapter, history_limit=4)


report = EvaluationHarness(make_loop).run(
    standard_evaluation_suite()
)

print(report.to_markdown())
print(report.to_json())
```

For serious model comparisons, pin provider/model configuration, run each scenario multiple times when the model is stochastic, save the raw `ABComparison` objects, and compare confidence intervals rather than one-off scores.

## Real-model run in GitHub Actions

The repository includes `scripts/run_infinito3_real_eval.py` and a manual workflow named **INFINITO 3.0 Real Model Evaluation**.

Before the first cloud run, create a repository Actions secret named:

```text
OPENAI_API_KEY
```

The secret is injected into the job environment and is never committed to the repository. The model id is selected when launching the workflow rather than hard-coded in INFINITO.

The runner uses a fresh SQLite `:memory:` store for every scenario and local deterministic hash embeddings. That keeps scenarios isolated and avoids extra embedding API cost in the first real-model benchmark.

To run it from GitHub:

1. Open **Actions** in the repository.
2. Select **INFINITO 3.0 Real Model Evaluation**.
3. Choose **Run workflow**.
4. Enter the OpenAI API model id to evaluate.
5. Optionally set reasoning effort and short-term history limit.
6. Run the workflow.

GitHub publishes the Markdown result in the Actions summary and uploads both JSON and Markdown reports as an artifact retained for 30 days.

The same runner can be executed locally:

```bash
export OPENAI_API_KEY="..."
python scripts/run_infinito3_real_eval.py --model "<api-model-id>"
```

A local run needs the computer only while the script is running. A GitHub Actions run does **not** require the user's computer to remain powered on.

## What this milestone does not claim

A positive score does not prove that INFINITO improves general intelligence. The harness only demonstrates performance on explicitly defined cognitive tasks. The benchmark must grow before broad claims are justified.

The next evaluation work should add independent multi-turn trajectories, larger held-out scenario sets, semantic/human judging, grounded hallucination checks, token/cost-normalized utility, statistical confidence intervals, and ablations for memory, goals and Context Builder separately.
