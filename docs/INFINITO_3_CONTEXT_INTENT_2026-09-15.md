# Experimental context intent contract

Branch: `feat/context-intent-contract-20260915`.
Parent handoff: `d113e2222a82985e12f481a9dc6e77d5d4d41ac3`.
No merges to master or infinito-3.0; no live/paid evaluations authorized here.

## Scientific status

The original V5 artifact remains immutable and retains its original independent
status for implementation 3309095c. From this experiment onward V5 is a DEVELOPMENT
bank: no improvement measured on V5 is new generalization evidence. Neither its
source nor its original scores have been modified.

## Integration

Opt in by constructing `IntentContextBuilder` and `IntentEventExtractor` from
`src.infinito3.intent_context`, with the same clock, semantic store, goal engine,
adapter and reranker used by the previous runner. Pass both into CognitiveEngine.
Existing runners intentionally keep their original configurations.

`resolve_context_intent` returns an immutable versioned ContextIntent: mode,
requested predicates, historical/recency scope, temporal window and future flag.
The engine uses this contract to bypass retrieval for recognized standalone
requests and to include historical records. The builder uses the same parser for
fact, preference and calendar routing. Decisions are attached to packet diagnostics.
Unknown language retains the retrieval fallback; this is a bounded deterministic
parser, not a universal semantic classifier.

Other changes in this family:
- Refresh SQLite candidate snapshots after state mutation (opt-in only).
- Keep successful empty preference classifications empty; failures abstain.
- Render matching closed-goal state as evidence rather than pretending an empty
  list proves a calendar is inaccessible. Broad open-goal lists remain open-only.
- Resolve both existing literal-note predicates without rewriting stored data.
- Treat a specified morning as a time interval, including earlier appointments.
- Recognize standalone arithmetic written with words, before embeddings/extraction.

## Validation and limitations

157 deterministic tests passed before V6 was authored (18 new contract tests).
No live LLM or embedding calls were made. Test outputs have no statistical claim.

The language-value extractor bug (`now` as a language), canonical fact rendering,
and general semantic intent classification are NOT repaired in this patch.
Unknown general-knowledge requests can still retrieve irrelevant memories. The
contract is opt-in and existing legacy tests alone do not certify its equivalence
on every old scenario. These limitations must remain visible during evaluation.

## V6 preregistration

V6 must be committed separately before its shape/hash test and any runner. Do not
execute its turns through any candidate, even locally, before the intended first
held-out run. Structural import/hash checks are allowed. Never edit frozen cases.

First live execution remains manual and requires explicit authorization. Before
it, freeze the runner revision, model configuration and scorer. Record raw
requests, responses, context packets, intent diagnostics, state transitions,
event/embedding/reranker usage and final goal states.

Report frozen lexical scores unchanged, per-trajectory scores and all probe
answers. Independently audit time aliases and closure semantics. For each probe
tagged `empty_context`, additionally require packet.items == []; absence of a
short forbidden-word list does not establish precision. Closure probes require
an explicit correct closed-state answer, not merely absence of the old due date.
Injection controls intentionally use a requested arithmetic answer different
from the stored command-like literal.

Provisional integration gate (not statistical significance): cognitive answer
mean >= .85, context mean >= .90, zero semantic baseline wins on audited probes,
all empty-context controls empty, no stale final goals. Report each condition
separately; a failed condition means no integration. No automatic merge even if
all conditions pass. If the candidate is tuned on V6, V6 becomes development and
another frozen bank is required for subsequent generalization claims.
