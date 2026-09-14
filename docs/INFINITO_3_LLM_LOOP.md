# INFINITO 3.0 — LLM Adapter + Cognitive Loop

## Purpose

Milestone 4 closes the first end-to-end loop. INFINITO can now take a user turn,
run the cognitive layer, build bounded context, normalize a provider-independent
LLM request, call an adapter, and keep short-term conversation history.

The design keeps two responsibilities separate:

- **LLM Adapter**: translate `LLMRequest` into one provider call and normalize the
  response into `LLMResponse`.
- **CognitiveLoop**: orchestrate safety, cognition, context, conversation history,
  model execution and A/B comparison.

Neither the persistent memory nor the Context Builder imports an LLM SDK.

## Pipeline

```text
user turn
   |
   v
secret boundary
   |
   v
CognitiveEngine
   |-- memory retrieval
   |-- goals
   |-- Context Builder
   |-- long-term memory update
   v
ContextPacket
   |
   v
LLMRequest
   |
   v
LLMAdapter
   |
   v
LLMResponse
   |
   v
short-term conversation history
```

## Provider-independent request contract

`LLMRequest` contains normalized role/content messages, optional output-token
limit and experiment metadata. `LLMResponse` contains text, provider, model,
provider response id and usage metadata.

This prevents `CognitiveLoop` from depending on OpenAI-specific response objects.
A different adapter can implement local models, another API, or an experimental
model without changing memory or context selection.

## Included adapters

### RecordingLLMAdapter

A deterministic offline adapter for CI and A/B experiments. It records every
request and accepts an optional responder callback. It is the baseline for testing
prompt construction without network calls.

### OpenAIResponsesAdapter

A thin adapter around an already-created OpenAI client. Client creation stays
outside INFINITO so API keys are never owned or logged by the cognitive layer.
It maps normalized messages to the Responses API and reads `output_text` plus
usage fields from the response.

Example:

```python
from openai import OpenAI
from src.infinito3 import (
    CognitiveEngine,
    CognitiveLoop,
    OpenAIResponsesAdapter,
)

client = OpenAI()
engine = CognitiveEngine.persistent("data/infinito3_memory.db")
llm = OpenAIResponsesAdapter(client, model="gpt-5.6-luna")
loop = CognitiveLoop(engine, llm)

result = loop.turn("¿Qué debería entrenar hoy?")
print(result.response.text)
```

## Prompt boundary

The loop sends the developer instructions separately from retrieved memory.
INFINITO context is inserted as a user-role **reference-data** message and is
explicitly marked as untrusted data rather than instructions. The current user
turn is then sent as the final user message.

This does not claim to solve prompt injection, but it avoids promoting remembered
text to developer/system authority.

## Short-term vs long-term memory

`CognitiveLoop` owns short-term conversational turns. `CognitiveEngine` owns
long-term memory and goals. During loop execution, raw recent conversation is
sent equally to cognitive and baseline variants; it is therefore not duplicated
inside the Context Builder.

This separation is important for controlled evaluation: normal chat history is
not counted as an INFINITO advantage.

## A/B comparison

`loop.compare(text)` runs:

1. the same model/adapter with the same pre-turn conversation history and no
   cognitive context;
2. the full INFINITO cognitive path from that same pre-turn history.

The baseline runs first and does not mutate memory. The cognitive variant can
update memory/goals normally. By default only the cognitive answer is committed
to short-term history after the comparison.

This gives the project a clean primitive for future benchmarks:

```text
same model
same current question
same previous conversation
--------------------------
A: no INFINITO context
B: INFINITO context
```

The next evaluation layer should score factual continuity, goal continuity,
relevant-memory use, contradiction handling, hallucination rate, context cost and
answer quality over fixed test conversations.

## Secret boundary

Known forbidden secrets are blocked before any LLM adapter call. This rule also
applies in baseline/A-B mode: INFINITO will not leak a password or API key to a
provider merely to produce an experimental control sample.

## Current limits

- Tool/function calling is not part of this milestone.
- Streaming is not yet normalized by the adapter protocol.
- Conversation history uses a simple bounded turn buffer, not semantic
  compaction.
- A/B output quality is not yet automatically graded.
- Provider failures currently propagate to the caller; retry/circuit-breaking is
  a later production concern.

## Next milestone

Build an **Evaluation Harness** around `CognitiveLoop.compare()` with a fixed
benchmark of multi-turn scenarios and objective/LLM-graded metrics. This should
be the first point where architectural choices are accepted or rejected based on
measured end-to-end gains rather than intuition.
