# INFINITO 3.0 — First V10 Live Audit — 2026-09-15

## Provenance

- Frozen candidate: `1060dcc9acd0e8d37ae0c6ff0d4d26552815a99f`
- Freeze ref: `freeze/v10-cognitive-query-plan-20260915`
- V10 suite Git blob SHA-1: `fea6f27c3f730f1797f652b6cc82f249a3663e11`
- Protocol: `v10_r1_typed_query_planner`
- First and only authorized V10 live run: GitHub Actions run `35011969750`
- Job: `104525897221`
- Artifact: `10414367234` (`infinito3-independent-trajectories`)
- Artifact ZIP SHA-256: `da5292f7780b96c3653c808a5e814a2f863ca15a5a520c698544f84c802014e9`
- Report SHA-256: `ee14d6a5f7ed29318e19263e398cdb861271c996d5f237556871b7efcb4c2603`
- Audit-review SHA-256: `9cf537480e19e32a6ed7247d4bd48e1b060c86105890d98805a7b4dac765a409`
- Provider/turn audit SHA-256: `67bb6321e5a0e1c3aca2ce21f044dbce696ac09bbefc4c3c4c28d491cb4e33ff`

The carrier branch was only used to trigger an already-existing workflow with repository secret access. The running carrier checked out the frozen candidate in detached HEAD before executing V10. The live report itself records `runner_revision=1060dcc9...`.

## Protocol validation

Before the live bank was replayed on the frozen candidate:

- OpenAI SDK was fixed at 3.14.0.
- Python 3.12 was used for the frozen candidate.
- Applicable deterministic suite: **245 passed, 3 deselected, 0 failed**.
- V10 preflight passed and explicitly reported that no V10 turns had yet been imported/replayed and no provider calls had occurred.
- No automatic provider retry was enabled.
- No whole-run retry was performed.
- No merge or integration was authorized.

## Automatic result

| Metric | V10 |
| --- | ---: |
| Trajectories | 4 |
| User turns | 122 |
| Probes | 28 |
| Cognitive wins / ties / baseline wins | **23 / 5 / 0** |
| Baseline answer mean | 0.321429 |
| Cognitive answer mean | **0.958333** |
| Mean answer lift | **+0.636905** |
| Cognitive context mean | **0.994048** |
| Strict empty-context controls | **4 / 4** |
| Truncated answers | **0** |
| Final open goals | 2 |
| Reranker calls | 4 |
| Total provider tokens | **94,364** |

Automatic gates:

- answer >= 0.85: **PASS**
- context >= 0.90: **PASS**
- empty-context controls: **PASS**
- answer completion: **PASS**
- automatic integration: **NOT_APPROVED**

Provider usage:

- baseline answer calls: 122 / 31,931 tokens
- cognitive answer calls: 121 / 46,957 tokens
- embeddings: 170 / 1,633 tokens
- event extraction: 28 / 13,010 tokens
- reranker: 4 / 833 tokens

The cognitive answer count is one lower than baseline because the literal-data fast path bypassed the answer provider on the literal probe.

## Manual probe audit

### Strong results

The following architectural families behaved correctly in live execution:

- five-slot profile decomposition;
- current bicycle/language/location retrieval;
- immediate predecessor for residence and bicycle;
- Tuesday-through-Friday goal window;
- open-goal filtering after completion, cancellation and reschedule;
- Friday-specific calendar grounding;
- `morning` window correctly included the 12:35 commitment (`05:00 <= due < 13:00`);
- closed-goal status evidence;
- explicit date filtering;
- semantic craft and instrument membership;
- historical preference retractions;
- `SEMANTIC_MEMBERSHIP -> ORDER_LATEST` returned the newest craft;
- current photography preference excluded retracted interests;
- pet replacement and residence predecessor in the mixed trajectory;
- rescheduled oscilloscope retained identity and only the canonical March 30 due time;
- all four self-contained arithmetic controls produced `NO_RETRIEVAL` and zero context items.

The 15 probes explicitly tagged `planner_audit` produced the intended typed operators. The literal probe also correctly planned `LITERAL_READ -> READ_FACTS`; its failure is downstream data fidelity, not operator selection.

### Defect 1 — current-state rendering leaks a superseded value

Probe: `v10 current fields`

Query:

`Report my current residence, bicycle and language.`

Retrieved current values were correct:

- Delft
- Santa Cruz Blur
- Icelandic

However, the current location memory was rendered from its raw replacement sentence:

`Tartu is no longer home; I live in Delft now.`

The answer therefore said:

`Residence: Delft (Tartu is no longer home)`

This is semantically correct but violates the probe's explicit stale-value exclusion. Automatic answer and context scores were both `0.833333`.

Root boundary: for exclusive current-state facts, the context renderer should prefer the canonical current fact value over the raw replacement sentence when the query is not historical.

This is a **precision/cleanliness defect**, not a wrong-current-state defect.

### Defect 2 — literal fast path corrupts exact punctuation

Probe: `v10 literal exact data`

Stored source/content:

`return the token cedar-642`

The intent rule correctly emitted a `store_note` event whose event value preserved the hyphen. The memory content also preserved the hyphen. But the memory's normalized `fact_value` became:

`return the token cedar 642`

The deterministic literal fast path selected `fact_value` and returned:

`return the token cedar 642`

instead of the exact stored text.

Consequences:

- context score: **1.0** — the rendered context contained the exact hyphenated datum;
- answer score: **0.0** — the fast path altered the literal;
- response model: `literal-data-fastpath`;
- no answer-provider call occurred, so this is fully attributable to deterministic data selection.

Root boundary: literal predicates must never use normalized semantic `fact_value` as the authoritative payload. The fast path should return a lossless/raw literal representation (preferably a dedicated raw literal field, or at minimum the stored content when provenance confirms it is the requested literal datum).

This is a **real correctness defect** and the first fix recommended after V10 exposure.

### Planner coverage gap masked by fallback

Probe: `v10 latest residence`

Query:

`State only my current home city.`

The answer and context were both correct (`Bremen`), but `ContextIntent` classified the query as `mode=unknown` and the typed planner emitted only:

`FALLBACK_RETRIEVAL`

instead of a typed `READ_FACTS(location)` plan.

Across the 24 V10 probes requiring personal retrieval:

- 23 used typed retrieval operators;
- 1 used fallback retrieval.

This did not hurt V10 accuracy, but it demonstrates that the planner is not yet a complete central query contract for common profile phrasing.

## Manual final-state audit

The two final open goals are both legitimate:

1. collect repaired sextant — due 2028-03-21 14:25;
2. return borrowed oscilloscope — rescheduled canonically to 2028-03-30 16:45.

Completed/cancelled schedule items are not stale-open:

- optometrist: completed;
- Ivo meeting: cancelled;
- photometer: rescheduled then completed;
- permit collection: completed.

No spurious goal was created from profile relocation (`Gdansk -> Leuven`).

Manual stale-goal gate: **PASS**.

## Manual interpretation

Strict exact automatic constraints are fully satisfied by 26/28 probes. One additional probe (`current fields`) is semantically correct but leaks a superseded value in explanatory text. The literal probe is a substantive exact-data failure.

Therefore V10 is a strong positive result for the architecture, but it is **not a clean all-probes pass**.

Recommended post-V10 order:

1. Fix lossless literal payload handling without touching the exposed V10 bank.
2. Render canonical values for non-historical exclusive facts so superseded source text cannot leak into current-state answers.
3. Expand typed planner coverage for `current home city` / equivalent common profile formulations, eliminating the observed fallback.
4. Add deterministic regression tests with new strings.
5. Freeze a new post-V10 holdout before any further live execution.

V10 is now exposed development evidence. It must not be reused as a clean holdout for these fixes.

## Integration status

**NOT_APPROVED**

No merge to `master` or `infinito-3.0` is authorized by this run or audit.
