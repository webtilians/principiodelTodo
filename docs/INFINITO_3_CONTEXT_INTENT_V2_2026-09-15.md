# INFINITO 3.0 — ContextIntent v2

Experimental branch: `feat/context-intent-v2-20260915`.

Scope is intentionally one architectural family: the centralized context-intent
contract. No answer-generation model, memory storage policy, semantic event
schema, scorer, V6 bank or V6 runner is changed by this milestone.

V2 adds bounded, entity-independent handling for:

- compositional profile requests introduced by verbs such as `recall`, `show` and `list`;
- historical relation wording such as `preceded`, `former`, `prior` and `came before`;
- calendar range queries expressed as commands rather than question forms;
- lifecycle questions that contrast completed/closed state with open/pending state;
- a canonical retrieval alias family for literal/test/verification phrases;
- historical preference operator cues (`lost interest`, `lost appeal`, `no longer enjoy`, etc.) before semantic reranking.

The standalone safety boundary remains conservative: only unambiguous general or
arithmetic requests bypass retrieval. Personal or mutation requests retain it.

V6 R2 is development evidence only for this candidate. The new tests deliberately
use different entities and formulations. After deterministic tests pass, a new V7
bank must be frozen before any live execution. A V7 run requires separate explicit
authorization and must not be described as an independent proof of V2 until that
bank is frozen.
