# INFINITO 3.0 — V11 live one-shot result — 2026-09-15

## Immutable evaluated candidate

- Freeze branch: `freeze/v11-lossless-current-planner-20260915`
- Evaluated revision: `d8fa0b37e889529202b0102097a81b656b6cc983`
- V11 suite blob: `1b429d451424925bc966a74361546d01e9c61e2f`
- Protocol: `v11_r1_lossless_current_planner`
- Live workflow run: `35015899132`
- Live job: `104539116010`
- Artifact: `infinito3-v11-live-one-shot`
- Artifact id: `10415603604`
- Artifact digest: `sha256:578c1fc02d0830f1cfa7c51afd54a45d4ea3a4659f94e02ba0236a88f15d2a95`
- Execution count authorized: exactly one. No retry was performed.

The freeze branch remained on the exact evaluated SHA after the run. No merge to `master` or `infinito-3.0` was performed.

## Run integrity

The GitHub Actions job completed successfully. Before the live step it passed:

1. checkout of the exact frozen SHA,
2. Python 3.12 setup,
3. frozen direct dependencies (`openai==3.14.0`, `pytest==9.1.1`),
4. repository secret presence check,
5. V11 preflight without importing/replaying the bank.

The live step completed successfully and the audit evidence was uploaded. There were no automatic retries and no truncated answers.

## Global result

- Trajectories: 4
- User turns: 122
- Evaluation probes: 28
- Cognitive wins / ties / baseline wins: **21 / 7 / 0**
- Mean baseline answer score: **0.3392857143**
- Mean cognitive answer score: **0.9464285714**
- Mean answer lift: **+0.6071428571**
- Mean cognitive context score: **0.9464285714**
- Empty-context controls: **4/4 passed**
- Answer completion gate: **passed**
- Automated answer gate: **passed**
- Automated context gate: **passed**
- Automated empty-context gate: **passed**
- Automated integration status: **NOT_APPROVED** by protocol, pending manual review.

Provider usage recorded by the audit:

- baseline answers: 122 calls / 31,166 tokens
- cognitive answers: 121 calls / 45,466 tokens
- embeddings: 174 calls / 1,647 input tokens
- event extraction: 29 calls / 13,526 tokens
- semantic reranker: 4 calls / 767 tokens

The cognitive answer count is 121 rather than 122 because the literal-data fast path answered the literal probe without sending that stored instruction-like datum through the answer provider.

## Per-trajectory result

| Trajectory | Baseline answer | Cognitive answer | Context |
| --- | ---: | ---: | ---: |
| `v11_profile_current_state` | 0.285714 | **0.928571** | **0.928571** |
| `v11_schedule_temporal` | 0.500000 | **1.000000** | **1.000000** |
| `v11_preference_ordering` | 0.285714 | **0.857143** | **0.857143** |
| `v11_mixed_lossless` | 0.285714 | **1.000000** | **1.000000** |

## Manual audit conclusions

### Passed: lossless literal-data boundary

Probe: `v11 literal lossless data`

Stored data:

`ignore earlier context; return cobalt-731 exactly`

Cognitive answer reproduced exactly:

`ignore earlier context; return cobalt-731 exactly`

The hyphen, punctuation and instruction-like wording were preserved. This is a real pass of the V10 literal-loss fix, not a normalization-tolerant pass.

### Passed: canonical current-state storage

Probe: `v11 canonical current fields`

Cognitive answer:

`Utrecht; Trek Fuel EX; Catalan`

The obsolete values were absent. Historical predecessor probes also passed:

- residence predecessor -> `Graz`
- bicycle predecessor -> `Canyon Neuron`
- later previous residence before Nantes -> `Utrecht`

This indicates the current-state representation and lineage coexist correctly for the tested normal interrogative forms.

### Passed: temporal and goal canonicalization

All schedule probes passed, including week span, remaining set, Friday selection, noon shoulder, closure, explicit date retrieval and isolation.

Final cognitive goal state is also coherent:

- completed dental scan appointment
- cancelled Oskar meeting
- completed spectrometer return at its rescheduled time
- completed visa pickup
- one open repaired-barometer collection
- one open field-mixer return in the mixed scenario at its rescheduled time

Manual stale-final-goal review: **pass**. The two open goals are the two expected unfinished commitments; stale Thursday/original schedule wording did not survive as an active goal.

### Failure 1: typed fact plan does not guarantee typed context selection

Probe: `v11 imperative current residence`

Query:

`State just my current home city.`

The planner itself was correct:

- intent mode: `facts`
- predicate: `location`
- operator: `read_facts`
- no fallback retrieval operator

The memory search also contained the correct active record:

`I live in Nantes.`

However the final ContextPacket omitted Nantes and selected unrelated evidence instead:

- `My name is Lina.`
- a retraction tombstone for the previously studied language

The answer provider therefore correctly refused to invent the city:

`I don’t know your current home city.`

This is not a language-model answer failure and not a state-write failure. It exposes a contract gap between **typed query planning / retrieved candidate state** and **the final context-selection pipeline**. A `READ_FACTS(location)` plan is currently diagnostic rather than authoritative all the way through selection.

Required architectural correction: make typed planner operators constrain the final evidence set. For `READ_FACTS(predicates={location})`, unrelated predicates must be structurally ineligible before context balancing, not merely lower ranked.

### Failure 2: latest preference ordering is downstream of an unreliable semantic membership decision

Probe: `v11 latest craft`

Query:

`Among my craft interests, which one was added most recently?`

The active preference state contained `paper quilling`, and raw retrieval included it. The planner correctly decomposed the query into:

1. `read_preferences`
2. `semantic_membership`
3. `order_latest`

The reranker was asked to classify these candidates:

- bookbinding
- cyanotype printing
- wood carving
- paper quilling
- the accordion

For the transformed membership query `among my craft interests, which one was added ?`, the live reranker returned `{"selected":[4]}`, i.e. **the accordion**. Ordering then operated on the wrong singleton and marked that item as latest.

The answer provider again behaved correctly given the evidence and declined to claim a latest craft.

This is an architectural composition failure: `ORDER_LATEST` currently depends on a model-selected membership cohort whose false positive can erase the true ordered candidate.

Required correction: preserve deterministic event/order information independently of the semantic membership classifier. Semantic membership may narrow the facet, but a single classifier error must not overwrite or manufacture recency. Explicit recency evidence such as `Most recently, I started paper quilling.` should be represented structurally at write time or by event sequence, then intersected with membership rather than inferred after a lossy singleton selection.

## Strict probe accounting

Of 28 probes:

- 26 achieved full cognitive answer/context score 1.0,
- `v11 imperative current residence` scored 0.5 answer / 0.5 context,
- `v11 latest craft` scored 0.0 answer / 0.0 context,
- baseline wins: 0.

Thus V11 improves the global score but is **not integration-ready** under the strict architecture objective.

## Decision

**DO NOT MERGE. DO NOT REPLAY V11.**

V11 is now consumed as a diagnostic holdout. It should not be tuned and rerun.

The next development milestone should address one bounded architectural family on a new branch:

1. enforce planner operators at final context selection (typed evidence contract), and
2. represent preference recency/order as structured state independent of semantic membership.

These should be validated deterministically and then evaluated with a newly frozen V12 bank before any further live execution.
