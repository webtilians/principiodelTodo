# INFINITO 3.0 — Frozen Held-out V3 result

Implementation under test: `9256992d2ac997bde2ab6a5851f29a9211e9eed7`

Frozen V3 cases: `cbb05b9a803a6d0ed021f5790d43d6a87e45f426`

Suite SHA256: `58e9db6adb94fea9a1f98293d1c6a305166f3debea8b9e8d84142c7627f61ccb`

Artifact SHA256: `431c3cc9e77f6ae68425f206fa415cab305b8f10ffd55a4d0a831ef0216fa9ba`

## Result

- 4 trajectories
- 141 user turns
- 28 probes
- W/T/L: **13 / 12 / 3**
- mean baseline answer score: **0.3929**
- mean cognitive answer score: **0.5729**
- mean answer lift: **+0.1801**
- mean context score: **0.7426**
- reranker calls: 3
- reranker tokens: 976
- effective cognitive token delta: +8,289
- final active memories: 25
- final open goals: 4
- deterministic tests before live run: **118 passed**

## Preregistered decision

The run does **not** pass the preregistered integration criteria. In particular:

- cognitive score 0.573 < required 0.70 and also below the explicit 0.65 no-integration boundary;
- context score 0.743 < required 0.85;
- answer lift +0.180 < required +0.20;
- baseline wins = 3, which is within the allowed maximum but does not compensate for the other failures.

Decision: **do not integrate the temporal architecture into `infinito-3.0` yet. Keep it experimental.**

## Scenario breakdown

| Scenario | W/T/L | Cognitive | Context |
| --- | ---: | ---: | ---: |
| profile three-version lineage | 3/4/0 | 0.494 | 0.780 |
| concurrent goal lifecycle | 2/2/2 | 0.521 | 0.708 |
| dense preferences/retractions | 4/1/1 | 0.764 | 0.847 |
| mixed state/notes/goals/revisions | 4/5/0 | 0.542 | 0.667 |

## Three losses

### 1. Six concurrent goals

The query asking for all commitments received no goal context. One initial commitment (`clase de guitarra`) was not extracted as a goal, and the goal-query intent path did not generalize sufficiently to this wording. This is a **goal creation + query-intent/context-selection failure**.

### 2. Rescheduled veterinarian appointment

The reschedule event was extracted, but target resolution returned `goal_lifecycle_target_not_found`. The old Tuesday 18:00 appointment stayed authoritative, so Thursday's probe returned the stale schedule. This is a **goal lifecycle target-resolution failure**.

### 3. Current beverages after retraction

`I no longer drink kombucha.` was not converted into a retraction event. `pu-erh` was stored, but the beverage query selected kombucha + matcha rather than matcha + pu-erh. This combines an **event-extraction retraction gap** with a **semantic/context-selection gap**.

## Other important failure families

### Structured-event coverage is still phrase-sensitive

Examples not handled reliably include:

- `He cambiado otra vez de bici: ahora uso una Yeti SB160.`
- `I no longer drink kombucha.`
- `I have started enjoying bouldering.`
- `Mi gata se llama Nube.`
- `Mi frase de verificación es: ...`
- `tengo clase de guitarra`
- `tengo revisión del coche`

The current rule-based extractor therefore remains a bottleneck.

### Historical evidence lacks explicit temporal semantics in the prompt

For several probes the correct predecessor was present in the ContextPacket (`Lyon`, `Irene`, `Zaragoza`), yet the LLM answered that it did not know. The memory text only says the historical fact itself; it does not explicitly tell the model `this was the immediately previous value before X`. This is a **historical-context representation/provenance problem**, not retrieval alone.

### Heterogeneous structured queries are incomplete

Current-profile queries sometimes returned only a subset of name/city/bike/language/job, even though those facts existed in state. Candidate generation still relies too much on text retrieval before structured predicate recovery.

### Goal lifecycle remains the largest operational weakness

V3 finishes with four open goals. Only the newly-created repaired-laptop goal is intentionally open. The veterinarian, glasses/prototype-related stale states account for the remaining unwanted open goals because creation or lifecycle matching failed.

### Self-contained prompt hygiene is improved but not sufficient as a general guarantee

The dedicated V3 math controls pass, but V2 already showed that unrelated memories can still leak into self-contained requests under some phrasings. This remains a regression target rather than a solved invariant.

## Conclusion

V2 showed that event-sourced temporal cognition can produce a large improvement on a bank that exposed the previous architecture's weaknesses. V3 shows that the current implementation does **not yet generalize robustly** to unseen phrasing and mixed-state pressure.

The next development target should not be another Context Builder threshold. The highest-value work is to reduce phrase sensitivity in event extraction and goal lifecycle resolution, and to make historical/structured state explicit to context selection. Any fixes must be evaluated against the already-frozen V3 bank first and then against a new V4 held-out before integration.