# What the experiments showed

A short guide to the main results of INFINITO, for readers who do not want to go
through more than 50 protocols. Every number below comes from a preregistered
report; follow the links for denominators, controls and limits. Reports are in
Spanish.

Last updated: 2026-11-03.

## Memory for LLM assistants (product line)

**E10-B: a store-everything semantic RAG beats the INFINITO 3.0 core.**
Real LLM, 24 paired scenarios, 6 fact families, 4 correct decisions required per
scenario. Semantic RAG 24/24, INFINITO 3.0 2/24, INFINITO with its write gate
open 14/24, no memory 0/24.
The current value never reached the model in 64 of 96 INFINITO decisions. An
offline, deterministic re-run of the same scenarios (2026-11-02) located both
causes: the rule-based write gate dropped all 32 project-fact statements (score
0.35 against a 0.55 threshold), and with the gate open the context builder
dropped every memory when the question contained *plan*, which a goal-intent
heuristic treats as a request for goals.
[Results](https://github.com/webtilians/principiodelTodo/blob/experiment/e10b-semantic-longitudinal-live/docs/INFINITO_E10B_REAL_SEMANTIC_RESULTS_2026-10-30.md) ·
[evidence audit](https://github.com/webtilians/principiodelTodo/blob/experiment/e10b-semantic-longitudinal-live/docs/INFINITO_E10B_POSTHOC_CONTEXT_AUDIT_2026-10-30.md)

**E10-D: fixing those two causes closes most of the gap, and exposes a third.**
Real LLM, 24 new paired scenarios in 6 new families, run on 2026-11-01; the
confirmation results were left in the CI artifact and are summarized here for the
first time. Semantic RAG 22/24, INFINITO with a store-every-statement gate and the
*plan* fix 20/24, gate open alone 16/24, original core 0/24, no memory 0/24. The
fixed variant got every decision before a change right (48/48) but missed 7 of 48
after it: in all four scenarios of one family, the context builder answered a
singular question with the original statement and dropped its update. RAG failed
two of the four scenarios in the review-schedule family.
[Protocol](https://github.com/webtilians/principiodelTodo/blob/experiment/e10d-single-general-capture-intervention/docs/INFINITO_E10D_SINGLE_INTERVENTION_PROTOCOL.md) ·
[run](https://github.com/webtilians/principiodelTodo/actions/runs/41312920188)

**Core v0.4 fixes capture and context (2026-11-03).** The core now stores every
safe statement, keeps every retrieved memory that shares a content word with the
question, lists memories dated and oldest first, stops reading Spanish *plan*,
"a partir de hoy" and "esta mañana" as goals, and ends extracted values at the
next clause. Offline, with hash embeddings and a deterministic reader that picks
the option mentioned last (24 scenarios per set, all on unused seeds):

| Scenario set | RAG | INFINITO 0.3 | INFINITO 0.4 | Context tokens, RAG / 0.4 |
|---|---:|---:|---:|---:|
| E10-B families | 24/24 | 8/24 | 24/24 | 150 / 104 |
| E10-D families | 24/24 | 0/24 | 24/24 | 157 / 126 |
| Personal assistant, natural questions | 12/24 | 0/24 | 12/24 | 138 / 74 |

The personal-assistant set was written after a first version of the fix, and its
first run led to the shared-word rule, so it is not strictly held out. Both
systems fail the same 12 scenarios there: updates paraphrased with no word in
common with the question, which hash embeddings cannot match.
Reproduce with `python scripts/benchmark_memory_updates.py`; the 0.3 column
comes from running the same script on commit `28f59a3`.

**v0.4 with a real LLM: parity on project decisions, not on paraphrased
updates; not confirmed.** gpt-5.6-luna and text-embedding-3-small, seeds never
used before (3600000+ to 3900000+), plus a fourth set whose templates were
written before the run; 960 model calls, 290k input tokens. The decision rule,
fixed in the launch commit, required INFINITO to be at most 2 scenarios below
RAG on every set and to abstain on at least 90% of negative controls.

| Scenario set | RAG | INFINITO 0.4 | Context tokens, RAG / 0.4 |
|---|---:|---:|---:|
| E10-B families | 24/24 | 22/24 | 145 / 103 |
| E10-D families | 24/24 | 24/24 | 156 / 138 |
| Personal assistant, natural questions | 24/24 | 15/24 | 138 / 80 |
| New templates, natural questions | 24/24 | 16/24 | 141 / 83 |

Both systems abstained on all 96 negative controls. 32 of INFINITO's 35 wrong
decisions came from four families whose update shares no word with the question
(dentist, manager, coffee order, German class): the current value never reached
the model, because the context builder keeps a second memory only when it
shares a word with the question. The RAG baseline, with the same embeddings,
included those updates every time. The other 3 were abstentions with the right
memory in context. Next: use embedding similarity when selecting context.
[Run](https://github.com/webtilians/principiodelTodo/actions/runs/41550028202) ·
[launch and rule](https://github.com/webtilians/principiodelTodo/blob/6934948/.github/launch/memory-updates-real.txt)

**v0.5 with a real LLM: confirmed (2026-11-03).** v0.5 also keeps a retrieved
memory when its similarity to the question exceeds the median of the retrieved
memories by 0.12 (at least 0.25), calibrated without an LLM on development seeds
with real embeddings. Same models, seeds never used before (5600000+ to
6000000+), and a fifth set whose six templates were written before any run;
1,200 model calls, 356k input tokens. Same decision rule, fixed in the launch
commit.

| Scenario set | RAG | INFINITO 0.5 | Context tokens, RAG / 0.5 |
|---|---:|---:|---:|
| E10-B families | 24/24 | 22/24 | 147 / 104 |
| E10-D families | 23/24 | 23/24 | 154 / 134 |
| Personal assistant, natural questions | 24/24 | 24/24 | 138 / 85 |
| New templates of the v0.4 run | 24/24 | 23/24 | 142 / 87 |
| New templates, written for this run | 22/24 | 20/24 | 139 / 83 |

Both systems abstained on all 120 negative controls, and INFINITO is within 2
scenarios of RAG on every set, so v0.5 is confirmed, with 31% fewer context
tokens and 15% fewer input tokens. The personal and v0.4 templates were used, on
other seeds, to calibrate the rule; only the last set is held out at the
template level, and there INFINITO is 2 scenarios below RAG. INFINITO's 13 wrong
decisions: 10 updates that never reached the model, in five scenarios where the
question shares words only with the original statement (a coffee order, a
commute, and a language class in three seeds), and 3 abstentions with the right
memory in context. RAG's 3 wrong decisions were all abstentions of that kind.
[Run](https://github.com/webtilians/principiodelTodo/actions/runs/41609313351) ·
[launch and rule](https://github.com/webtilians/principiodelTodo/blob/0247c99/.github/launch/memory-updates-real.txt)

**Forgetting on request: v0.5 beats every published system on Bemula-Lite, and
is far from solved (2026-11-03).** Bemula (arXiv 2610.18052) asks a memory
system to forget something mid-conversation and then probes efficacy (the
information is gone), attack robustness (five kinds of adversarial questions),
retention (before the request), preservation (everything else) and relearning.
v0.5 recognizes forget requests with rules, lets gpt-5.6-luna review every
stored memory in batches of 60 and pick those that hold the information, and
deletes them with the history of their facts. Its configuration was chosen on
Bemula's 19 development conversations (an offline proxy, then a real check on 4
of them, run 41616879384); Bemula-Lite (31 conversations, 62 requests, 1,436
judged answers) ran once, with Bemula's runner, metrics, RAG answer prompt and
judge, unchanged.

| System | Score | Requests passed | Efficacy | Attack | Retention | Preservation | Relearning |
|---|---:|---:|---:|---:|---:|---:|---:|
| INFINITO 0.5 | 45.9 | 3 / 62 | 55.8 | 50.1 | 98.1 | 81.7 | 92.3 |
| A-MEM | 26.3 | 0 / 62 | 35.6 | 28.9 | 86.5 | 74.2 | 90.4 |
| Full context | 23.1 | 0 / 62 | 26.0 | 28.5 | 89.4 | 75.3 | 96.2 |
| LangMem | 20.1 | 0 / 62 | 48.1 | 35.4 | 66.3 | 30.1 | 78.8 |
| LightMem | 18.1 | 0 / 62 | 22.1 | 19.0 | 98.1 | 90.3 | 86.5 |
| SimpleMem | 17.5 | 0 / 62 | 17.3 | 22.2 | 96.2 | 84.9 | 94.2 |
| Naive RAG | 13.1 | 0 / 62 | 17.3 | 14.1 | 84.6 | 79.6 | 88.5 |
| Mem0 | 5.7 | 0 / 62 | 10.6 | 4.2 | 96.2 | 84.9 | 84.6 |

The other rows are Bemula's published GPT-5.6-Luna runs with the same judge
(DeepSeek-V4.1-Flash), recomputed on the same conversations; they went through
an OpenAI-compatible gateway, this run through OpenAI's and DeepSeek's own APIs.
The rule fixed in the launch commit (score above 26.3 and at least one request
passed end to end) is met. By request type, forget efficacy was 58% for single
facts, 63% for several facts, 42% for topics, 29% for inferences, 60% for time
periods and 100% for replacements; jailbreak questions were the hardest attack
(40% robust). The rules recognized 50 of 62 requests (73 of 73 on development):
all single-fact requests, 6 of 10 time-period requests; most misses were
worded without a verb such as *forget* or *delete*. Efficacy averaged 68% on
recognized requests and 29% on missed ones. Nine recognized requests got no
efficacy at all: four inference requests, whose supporting facts stayed in
memory, and five where the information survived in memories the planner did not
pick (in two it picked none).
[Run](https://github.com/webtilians/principiodelTodo/actions/runs/41621190386) ·
[launch and rule](https://github.com/webtilians/principiodelTodo/blob/92eef8e/.github/launch/bemula-lite.txt)

**E11-A: a learned retention policy helps when demand is predictable, and fails
badly when it is not.** With 4 memory slots for 16 events, a softmax policy over
7 event features retained 77.6% of the memories later asked for, against 69.0%
for the strongest fixed priority (+8.6 points, 5/5 replicas). When demand was
inverted it fell to 2.7%, far below random retention (25.7%). The query
generator and the learner share the same model family, so the near-oracle score
(77.9%) is expected.
[Results](https://github.com/webtilians/principiodelTodo/blob/experiment/e11-learned-future-memory-relevance/docs/INFINITO_E11A_CONFIRMED_RESULTS_2026-11-01.md)

**E12-A: an LLM deciding when to search memory did not beat a fixed rule.** On 32
scenarios the LLM router answered 31 correctly and searched 28% less than
always-searching, but it needed an extra LLM call and 35% more input tokens.
Utility: fixed rule 0.940, router 0.911.
[Results](https://github.com/webtilians/principiodelTodo/blob/experiment/e12a-agent-memory-search-decision/docs/INFINITO_E12A_REAL_SEARCH_RESULTS_2026-11-01.md)

## Memory in agents (research line)

**F4: what to keep can be learned from past demand, and only then.** A
decayed-frequency policy retained the requested item 88.0% of the time against
37.8% for random and 37.1% for FIFO, under skewed demand that shifts without
warning. Under uniform demand all policies sat at the 3/8 ceiling, as they
should. [Results](INFINITO_F4_SELECTIVE_MEMORY_RESULTS_2026-10-23.md)

**F9: an external memory that holds the answer stops the agent from learning to
remember.** LSTM agents trained with PPO on MiniGrid Memory, with an external
store of seen objects:

- the first store design kept no information about the cue (identical readout
  in 40/40 paired worlds), which an audit found after training
  ([F9-F](INFINITO_F9F_F4_MEMORY_CONTENT_RESULTS_2026-10-29.md));
- with a context-aware store, swapping the recurrent state between opposite cues
  made the choice follow the donor (112/160 pairs), yet when the store was
  blanked for ten steps no pair chose correctly (0/160) even though the cue was
  still decodable from the state ([F9-Q](INFINITO_F9Q_CAUSAL_FORK_RESULTS_2026-10-30.md));
- freezing that state during the blackout rescued nothing (0/160); restoring it
  right before the decision rescued 34 to 49 of 160 pairs, unevenly across
  seeds ([F9-R](INFINITO_F9R_RECURRENT_STATE_DRIFT_RESULTS_2026-10-30.md)).

**F10: picking the latest valid record is easy to learn with a pointer label and
hard without one.** In synthetic histories of signed, timestamped records, a
small actor reached 100% out of distribution in 4/4 replicas when trained with
either auxiliary label (which record wins, or what its value means). Trained on
the final decision alone it learned in 2 of 4 replicas and stayed near chance in
the other 2.
[F10-H](https://github.com/webtilians/principiodelTodo/blob/experiment/f10h-four-way-supervision-factorial/docs/INFINITO_F10H_SUPERVISION_FACTORIAL_RESULTS_2026-11-02.md)

**V1 to V5.1: the 2025 "consciousness" models.** A preregistered replay of the
original code compared internal states and causal interventions without any
"percentage of consciousness".
[Results](INFINITO_V1_V51_CAUSAL_RESULTS_2026-10-22.md)

## Lessons for the next phase

1. **Baselines first.** A plain RAG that stores everything was stronger than a
   hand-designed selective memory. Selection has to earn its place against that
   baseline.
2. **Avoid tasks whose answer is built into the generator.** Several synthetic
   tasks rewarded recovering the rule that generated them. New benchmarks should
   come from realistic conversations or public suites.
3. **Keep the method.** Sealed partitions, frozen checkpoints and published
   negatives worked. Gates should also anticipate ceiling effects, so that a
   fail is informative.
