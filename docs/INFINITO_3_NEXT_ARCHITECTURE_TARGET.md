# INFINITO 3.0 — Next architecture target after frozen V3

Frozen implementation baseline: `9256992d2ac997bde2ab6a5851f29a9211e9eed7`

Frozen V3 result: cognitive `0.5729`, context `0.7426`, lift `+0.1801`, W/T/L `13/12/3`.

The next implementation targets are deliberately limited to the failure classes exposed by V3:

1. Semantic Cognitive Event Extractor with a closed schema and deterministic fallback.
2. Goal lifecycle target resolution using semantic + temporal + current-state evidence.
3. Structured state retrieval for explicit fact slots, plus explicit temporal lineage rendering.

No V3 case will be edited while these changes are developed. Improvements will first be measured against frozen V3, then validated on a new frozen V4 before integration into `infinito-3.0`.
