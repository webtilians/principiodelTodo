# INFINITO 3.0 — Persistent Cognitive Memory

## Purpose

Milestone 2 replaces the in-memory baseline with a persistent cognitive memory
that can evolve without losing its history. The database is not treated as a
bag of chat messages. Each memory has identity, type, confidence, importance,
access statistics, lifecycle state and optional structured fact fields.

## Storage model

The default persistent backend is SQLite (`SQLiteCognitiveMemoryStore`). Each
row stores:

- stable memory id
- original content and normalized content
- memory kind: working / episodic / semantic / user_model
- importance and confidence
- creation/update/access timestamps
- access count
- optional embedding
- optional structured fact `(subject, predicate, value)`
- lifecycle state: active / superseded / forgotten
- `supersedes_id` lineage pointer
- arbitrary JSON metadata

Nothing is hard-deleted by normal consolidation or forgetting. This makes the
memory auditable and allows restoration.

## Hybrid retrieval

Retrieval combines several signals instead of trusting one score:

```text
42% lexical overlap
38% vector similarity
 8% importance
 5% confidence
 4% recency
 3% prior access
```

The weights are a baseline, not a scientific result. They must eventually be
optimized against a fixed retrieval benchmark.

`HashEmbeddingProvider` is an offline deterministic baseline. It is useful for
CI and local development but is not claimed to provide deep semantic meaning.
For semantic retrieval, inject `OpenAIEmbeddingProvider` or another provider
implementing the same `EmbeddingProvider` interface.

Example:

```python
from openai import OpenAI
from src.infinito3 import CognitiveEngine, OpenAIEmbeddingProvider

client = OpenAI()
engine = CognitiveEngine.persistent(
    "data/infinito3_memory.db",
    embedding_provider=OpenAIEmbeddingProvider(client),
)
```

## Reinforcement instead of duplication

If the same text or the same structured fact is encountered repeatedly, the
existing memory is reinforced instead of creating another row. Its confidence
increases and a reinforcement counter is stored in metadata.

Example:

```text
"Me gusta descenso"
"Me encanta descenso"
```

Both map to the same non-exclusive fact:

```text
(user, likes, descenso)
```

The second observation strengthens the first.

## Contradictions and memory lineage

Some user facts are exclusive and should have only one active value at a time:

- name
- location
- age
- favorite color
- primary bike (baseline rule)

If the user first says:

```text
"Vivo en Málaga"
```

and later:

```text
"Vivo en Granada"
```

the old row is marked `superseded`; the new row becomes active and points to the
old memory through `supersedes_id`. The previous statement remains queryable for
audit/history but is excluded from normal retrieval.

Preferences such as `likes` are multi-valued and do not overwrite each other.

## Forgetting

`forget()` implements soft forgetting. A retention score combines:

- importance
- confidence
- recency
- access frequency
- memory-kind durability

Low-value memories can move to `forgotten` and disappear from normal retrieval.
They remain in SQLite and can be restored. Memories above the configured
importance protection threshold are preserved by default.

This is deliberately safer than destructive deletion while the forgetting
policy is still experimental.

## Consolidation

`consolidate()` merges duplicate/equivalent active records, retaining the
strongest representative and marking redundant rows as superseded. In normal
v3 ingestion most duplicates are already prevented at write time; consolidation
is mainly useful for imported legacy data and future batch/sleep processing.

## Current limits

The structured fact extractor is intentionally conservative and rule-based. It
does not attempt general natural-language understanding. A future extractor may
use an LLM to propose structured facts, but any such proposal should pass schema,
confidence and contradiction checks before being committed.

The current SQLite implementation scores candidate rows in Python. That is fine
for the present experimental scale. If memory grows substantially, retrieval
should move to SQLite FTS plus a dedicated vector index while preserving the
same `MemoryStore` contract.

## Next milestone

Milestone 3 should build the **Context Builder** above this memory. Its job will
be to decide not merely which memories are similar, but which combination of
memories, goals, recent context and contradictions should enter the LLM context
window for a particular task.
