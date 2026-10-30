# INFINITO 3.0 — Cognitive Layer Architecture

## Goal

INFINITO 3.0 is a persistent cognitive layer that sits between a user and any
LLM. Its job is not to generate language. Its job is to decide what deserves
attention, what should persist, what should be retrieved, and which goals remain
active across interactions.

The architecture deliberately separates the cognitive layer from Streamlit,
OpenAI, and the legacy IIT experiments.

## Core pipeline

```text
User input
   |
   v
SensitiveInformationFilter
   |
   +-- forbidden secret --------> NEVER STORE
   |
   v
Memory retrieval (existing memories only)
   |
   v
MemoryGate
   |
   +-- store? ---> MemoryStore
   |
   v
GoalEngine
   |
   v
CognitiveDecision
   |
   v
Context Builder / LLM adapter (next milestone)
```

## Design rules

1. **Provider agnostic**: `CognitiveEngine` must not import an LLM SDK.
2. **UI agnostic**: the engine must run in tests without Streamlit.
3. **Safety before memory**: secrets are inspected before any persistence path.
4. **Conservative persistence**: sensitive PII is not persisted by default.
5. **Replaceable components**: gate, memory backend, goal engine and safety
   policy are interfaces, not hard-coded dependencies.
6. **Baselines before neural complexity**: learned gates, LoRA and IIT-inspired
   metrics must beat deterministic baselines on held-out tests before becoming
   defaults.
7. **IIT is experimental**: IIT/PHI research remains valuable, but is not a
   required dependency of the production cognitive core.

## Memory model

The first milestone defines four memory classes:

- `WORKING`: short-lived information used inside the current interaction.
- `EPISODIC`: events and experiences tied to a moment or situation.
- `SEMANTIC`: durable facts and concepts.
- `USER_MODEL`: durable preferences and user-specific facts.

`InMemoryMemoryStore` is only a deterministic baseline. The next persistence
backend should implement the same `MemoryStore` protocol with SQLite plus vector
retrieval, allowing lexical, embedding and hybrid retrieval to be compared.

## Memory Gate

`RuleBasedMemoryGate` is intentionally simple and explainable. It is not the end
state. It establishes a baseline that future neural gates must outperform.

A replacement gate should be evaluated with a fixed train/validation/test split
and report at least precision, recall, F1 and calibration. Accuracy on the
training set is not considered evidence of generalization.

## Goals

`SimpleGoalEngine` is a deterministic baseline. It fixes the legacy substring
ordering bug where `pasado mañana` could be interpreted as `mañana`, and parses
simple clock expressions such as `a las 10:30`.

A production goal engine should eventually represent timezone, recurrence,
confidence and source span explicitly.

## Safety policy

The first policy distinguishes:

- `SAFE`: eligible for persistence.
- `SENSITIVE`: usable during the current interaction, but not persisted by
  default.
- `FORBIDDEN`: secrets such as passwords, PINs, private keys and API keys. They
  are blocked before the memory gate.

The old behaviour that gave credentials a positive memory bonus must never be
reintroduced.

## Legacy compatibility

No legacy module is deleted in this milestone. Existing Streamlit, IIT, PHI,
LoRA and research code remains untouched on the branch so results can be
reproduced and selectively migrated.

## Next milestones

1. Persistent SQLite memory repository with explicit schema and migrations.
2. Embedding adapter and hybrid lexical/vector retrieval.
3. Context Builder with token-budget-aware memory selection.
4. LLM adapter interface and a baseline experiment: LLM alone vs LLM + INFINITO.
5. Evaluation dataset for memory write/retrieve/forget decisions.
6. Learned MemoryGate trained only after the baseline suite exists.
7. Re-evaluate Neural Memory/LoRA as an optional experiment, not a dependency.
