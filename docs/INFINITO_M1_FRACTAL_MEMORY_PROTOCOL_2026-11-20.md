# M1: structures the memory discovers and revises, against flat retrieval and a fixed hierarchy (protocol)

Written and committed on 2026-11-20, before any benchmark code exists and before the data is downloaded.
Nothing below is changed after a result is seen. A harness bug fix that does not change an arm, a parameter
or the data is logged in the report. The only decisions left open are listed under "What DEV may decide",
and each is taken on DEV, by a rule written here, before CONFIRM is opened.

This protocol is the P0 deliverable of INFINITO tasks 875c87f8 ("auditar viabilidad de memoria fractal") and
960256f2 (its correction). It covers the audit, the design and the experiment in one document, following the
rule to prefer one larger experiment over several small ones.

## Question

The maintainer's question (INFINITO memory, 2026-11-20): *what if INFINITO could form abstractions over its
own history, and go deeper into them when it needed to reason?* The original idea was "estructuras simples;
profundizas, estructuras complejas". The correction of the same day adds that the system should **learn to
generate, discover and reorganize its own structure** from the experience it accumulates, rather than receive
a hierarchy designed by hand.

Two hypotheses, judged separately:

- **H1, multilevel retrieval.** Going from groups down to the original episodes puts the evidence for an
  answer in front of the model more often than flat retrieval, at the same token budget. This is tested with
  a hierarchy fixed by the designer (A1).
- **H2, discovered and revised structure.** Structure that the memory finds in its own history, without
  categories given in advance (A2), and revises as new episodes arrive (A3), adds value over the fixed
  hierarchy and over flat retrieval.

"Fractal" is a metaphor for a recursive, navigable organization. It is not a mathematical condition, and not
a claim about the brain. A structure that turns out not to be fractal, or not a tree at all, can still pass.

## Corrections taken from the addendum (ADENDA MEMORIA EMERGENTE INFINITO v0.1)

- The dossier's L0–L4 levels are only an illustration and the **fixed-hierarchy control A1**. No taxonomy,
  number of levels, shape or semantic category is imposed on the arms under test.
- An LLM may suggest summaries, but it is neither evidence nor a judge of its own output. **No arm in M1 uses a
  model to build structure.** The judge comes from a different model family than the reader, and a human
  audits it.
- Arms: A0 strong flat retrieval, A1 fixed hierarchy, A2 discovered structure frozen at a cutoff, A3 A2 plus
  revision as new episodes arrive. Added as a negative control: A2 with its members shuffled.
- No points for depth, node count, fractal shape or appearance. Structures are judged only on tasks, source
  links, cost and safety.
- A structure that is unstable or irrelevant gives way to flat retrieval (fallback).
- Every revision is versioned and reversible, and the original sources are never modified.
- Out of scope, as separate lines: AtlasKV, KG2KV and HiKVP (keys and values inside a model), the geometry of
  representations, and any change to weights.

## What exists and what is reused (P0 audit, checked on 2026-11-20)

| Piece | Where | State | Role in M1 |
|---|---|---|---|
| `AgentMemory.remember`, `forget`, `_active` | `src/infinito3/mcp_server.py` | master, production | write and forget path for every arm |
| `MeaningRanker` (potion-multilingual-128M int8) | `src/infinito3/semantic.py` | production (`INFINITO_EMBEDDINGS`) | embeddings for every arm; A0 is its ranking |
| Exact project filter `_is_about` | PR #85, merged | master | scope-isolation check |
| Reflections: must cite sources, deleted with them | `src/infinito3/remote_reflections.py` | production | the contract a product version of derived structures would follow |
| Lineage (`supersedes_id`) and `consolidate()` | `persistent_memory.py`, `docs/INFINITO_3_MEMORY.md` | master | not used: they concern facts that change, and the turns here are not facts of that kind |
| `generalized_context_builder` (budgeted selection) | master | core library, not the MCP `recall` path | not used: assistants receive `recall` |
| Night engine (`cognition/`, `qwen3.8-motor`) | runs on the maintainer's PC | in use | would build structures in a product version; unused here, since no model builds structure |
| E13-A derived view | PR #79, open, not merged | E13A_FAIL | prior evidence; code not reused (written for its generator's grammar) |
| E13-B sets and temporal graph | PR #84, open | blocked, no result | seeds sealed and not reused |
| SQLite concurrency fix | PR #110, draft, not merged | open | the harness writes from one process, in order, so it is unaffected; a product write path must revalidate it first |
| R1 harness | PR #176, open | R1 failed its gate | model loading and latency method reused, not its data |

Earlier results that bear on this experiment:

- **E13-A** (5 November): a derived view with provenance answered 632 to 638 of 640 queries with a
  deterministic reader, against 553 for the core and 547 for RAG. The gain was almost all on two-hop relations
  (44 of 128 for the core, 126 to 128 with the view). Adaptive rebuilding beat periodic rebuilding by only 0.94
  points, below the 2 points required. Its extraction was a program written for the generator's own grammar.
  It predicts that organization can help, and that adapting the organization may add little.
- **R1** (18 November): with 200 memories the meaning ranker puts the answer among the 20 shown for 96% of
  single-fact questions. Structure is not expected to help there.
- **Bemula-Lite** (3 November): forgetting an inference reached 29% efficacy, because the facts it was drawn
  from stayed. Aggregates built from memories carry the same risk, so forgetting is a hard gate here.
- **FINDINGS.md, lesson 2**: avoid tasks whose answer is built into the generator. A synthetic corpus written
  by the author of the arms would also be a hidden ontology, the first risk the addendum names. M1 therefore
  uses a public suite.

## Data

**LoCoMo** (Maharana et al., 2024; `locomo10.json` from the authors' repository): 10 long conversations
between two speakers over many dated sessions, with questions, gold answers and the ids of the turns that hold
the evidence. It is on the README roadmap and has never been run in this repository. The licence is checked
before download, and the data is used for research only. It is downloaded once, after the maintainer approves
the download, and its SHA-256 goes in the report. The file is not committed.

Only dialogue turns are used. LoCoMo's own session summaries and observations are abstractions written by a
model, and no arm sees them. A turn's text is `<speaker> (<session date>): <text>`, plus
` [shares a photo: <caption>]` when the turn has one. The same text reaches every arm.

**Splits.** For each conversation, h = SHA-256 of `INFINITO/M1/v1/conv/<sample_id>`. The two conversations with
the lowest h are DEV and the other eight are CONFIRM. CONFIRM is opened once, after the freeze commit.

**History as a stream.** A conversation with S sessions has a cutoff at session c = ceil(0.6 × S). Sessions 1
to c are the history the structure is discovered from. Sessions c+1 to S arrive one at a time afterwards.
A question is *post-cutoff* if any of its evidence turns comes after c, otherwise *pre-cutoff*.

**Question types.** LoCoMo labels each question with a numeric category: single-hop, multi-hop, temporal,
open-domain or adversarial. Which number means which type is checked on DEV by reading ten questions of each,
and is recorded in the freeze commit. Gates name types, not numbers. In this protocol, *relational* means
multi-hop plus temporal, and *local* means single-hop plus open-domain. Adversarial questions have no answer in
the conversation.

No arm sees a question, an answer or an evidence id while building or revising structure. Questions exist only
at query time.

## Memory build

Each conversation gets a fresh store. Every turn goes through `AgentMemory.remember` (the real write path, with
its filters), in dialogue order, with its session date. A turn the filter refuses is logged and is missing for
every arm alike. Every arm uses the same embeddings, from the same model folder as R1.

## Generic signals (the only input to discovery)

These hold no domain category, entity type or closed list of kinds:

- **Meaning:** each turn's 10 nearest turns by cosine similarity, with that similarity as the weight.
- **Sequence:** consecutive turns in the same session, with weight equal to the median kNN weight of the
  conversation.
- **Shared rare words:** two turns sharing a word whose document frequency is in the rarest 5% of the
  conversation's vocabulary (stop words excluded by frequency, not by a list), with the same weight. This rule
  is fixed and scale-free.

## Arms

L0 is always the turns. Every arm ends in a **packet**: the text handed to the reader, of at most B tokens, with
tokens counted as ceil(characters / 4) over everything in it, including ids, dates and labels.

**Shared walk (A1, A2, A3 and the controls).** A node's score is the highest similarity between the question
and any turn under that node. The walk is best-first over nodes, starting from the root's children. Popping a
node pushes its child groups and the groups it is cross-linked to. A visited set prevents cycles, and at most
50 nodes are popped. When the walk reaches a group with no child groups, it adds that group's up to 6 turns
most similar to the question. It stops after 4 such groups or when B is full. Whatever budget remains is filled
as A0 fills it (the flat fallback). The packet shows turns in dialogue order. A turn appears once, however many
routes lead to it.

| Arm | Structure | Built from |
|---|---|---|
| **A0, baseline** | none: every turn ranked by meaning; turns added in rank order while they fit | — (the store-everything semantic RAG baseline at equal budget) |
| **A1, fixed hierarchy** | designer-imposed time tree: conversation → blocks of 4 consecutive sessions → session → turns | the dates, for all sessions as they arrive |
| **A2, discovered, frozen** | whatever discovery finds in sessions 1..c; later turns are attached, the structure never changes | sessions 1..c |
| **A3, discovered, revised** | A2's structure at the cutoff, revised after every new session (operations below) | sessions 1..c, then each session as it arrives |
| A2-shuffled (control, stage 1 only) | A2's exact shape and group sizes, with the turns randomly permuted among the groups (seed 20261120) | — |
| A2-rebuilt (reference, stage 1 only) | discovery run once on all S sessions | everything, at the end |

A1 is the honest counterpart, without a model, of the dossier's hierarchy. The dossier's L2 and L3 levels,
written by a model, are not reproduced, because of the correction above.

**Discovery (A2, A3, A2-rebuilt).**

1. Build the weighted graph from the generic signals over the turns available.
2. Run Louvain community detection (`networkx.community.louvain_communities`, networkx 3.6.1, already
   installed) at resolutions 0.5, 1, 2 and 4, with 5 seeds each.
3. **Keep a resolution as a level** only if the mean pairwise adjusted Rand index of its 5 partitions is at
   least 0.5 (stability). Its partition is the medoid, the one with the highest mean ARI to the other four.
   Groups with fewer than 2 turns are dropped. A level with more than n/3 groups is dropped (growth bound).
4. **Nesting is found, not forced.** Kept levels are ordered from coarse to fine. A finer group's parent is the
   coarser group holding a majority of its turns. If no group holds a majority, the finer group hangs from the
   root and is cross-linked to every coarser group holding at least 25% of its turns. Turns can therefore
   reach a group through more than one route, and the result may not be a tree.
5. **Cross-links between groups of one level:** weight = the sum of edge weights between their members divided
   by √(size × size), linked if that weight is in the top 5% of the level's pairs.
6. **Fallback:** if no resolution passes step 3, the arm has no structure and behaves exactly as A0. That is
   recorded as a finding: no stable structure was found.

The number of levels (0 to 4), the groups, the links and the shape all come from the data. The procedure is
fixed, so "not specified in advance" here means the content and the shape, not the algorithm.

**Attaching a new turn (A2 and A3, after the cutoff).** The turn joins the finest group whose centroid is most
similar to it, if that similarity reaches the 25th percentile of member-to-own-centroid similarities at the
cutoff. Otherwise it goes to a loose pool, which the walk does not cover and the flat fallback does.

**Revision operations (A3 only, after each new session; at most 10 per session).** Each one is recorded as a
revision with the operation, the nodes and links changed, the turn ids that triggered it and a version number:

- *create:* when the loose pool holds at least 6 turns, discovery runs on the pool and its stable groups become
  new nodes under the root;
- *split:* a group that has grown by 50% since its last version is rediscovered inside, and replaced by its
  stable subgroups if any level passes step 3;
- *merge:* two groups of one level whose centroid similarity exceeds the highest pair similarity at the cutoff;
- *link / unlink:* cross-links recomputed by the rule in step 5;
- *retire:* a group left with fewer than 2 turns; its turns go to the parent, or to the loose pool;
- *revise:* centroids are always recomputed from current members.

Revisions run in shadow: every version is kept, A3 answers from the latest, and loading an earlier version must
reproduce the packets logged under it (the rollback check). Sources are never changed.

All numbers above (k = 10, the 5% rules, resolutions, 0.5, n/3, 6 turns, 4 groups, 50 pops, 25th percentile,
6-turn pool, 50% growth, 10 operations) are fixed now. They are generic choices, not tuned to this data.

**Budget.** B = 1,024 tokens is the decisive budget unless the DEV rule moves it to 512. Both 512 and 2,048 are
reported. Production `recall` shows k = 20 memories rather than a token budget, so A0 with k = 20 is also
reported as a reference, outside the gate.

**Trace (the evidence packet).** For every question and arm the harness logs: question id, arm, structure
version, nodes visited in order with their scores, each included turn id with the route that brought it (the
node, or flat fill), tokens used, and whether the fallback filled anything.

**Replicas.** Discovery's 5 seeds are {1..5}, {6..10} and {11..15} for replicas 1 to 3, so A2, A3, A2-shuffled
and A2-rebuilt each have three replicas. A0 and A1 are deterministic. Stage 2 uses replica 1 only, for cost.

## Models (local, no paid API)

All run through Ollama's OpenAI-compatible endpoint on the maintainer's PC (RTX 4060 Laptop, 8 GB), with
temperature 0, a fixed seed, thinking off and a context of 8,192. Digests go into the freeze commit.

- Reader (every arm): `qwen3.8-motor`. It is a local 27B model, not the assistants people use. S0-I
  (10 November) suggests stronger models make better use of the same memory.
- Judge: a model from a **different family** than the reader, installed through Ollama before DEV with the
  maintainer's approval, chosen by the maintainer. Its tag and digest go into the freeze commit. It never sees
  the arm, and arms are shuffled.

Prompts (only output parsing may be repaired on DEV, and each repair is logged):

- Reader: "Answer the question using only the memory below. If the memory does not contain the answer, reply
  exactly: Not in memory. Use at most 15 words."
- Judge: "Question, reference answer, response. Does the response give the same answer as the reference
  (wording, extra detail and date format may differ)? Reply CORRECT or WRONG."

## Metrics

**Stage 1, no model involved** (every CONFIRM question with evidence ids, every arm and replica):

- **Full-evidence coverage:** the share of questions whose every gold evidence turn is in the packet. Only
  turns count; a group is never evidence.
- Reported as well: partial coverage, packet tokens, build and revision time, structure size in bytes, and
  descriptive counts (levels kept, groups, links, operations by type). Descriptive counts are never rewarded.

**Stage 2, the reader answers** (a sample of A0, A1, A2 and A3, replica 1). In each CONFIRM conversation and
question type, questions are ordered by SHA-256 of `INFINITO/M1/v1/q/<sample_id>/<index>` and the first 8 are
taken (all of them if fewer). That gives at most 320 questions, each read by four arms.

- Accuracy is the judge's CORRECT. Token F1 against the reference is reported too.
- Adversarial questions are scored without a model: correct only if the reply is exactly "Not in memory". A
  wrong reply there is a false memory.

**Forgetting** (a separate copy of each CONFIRM store). In each conversation, take the 5 single-hop questions
with one evidence turn and the lowest SHA-256 of `INFINITO/M1/v1/forget/<sample_id>/<index>`, and forget their
evidence turns through `AgentMemory.forget`. Each structure arm then removes the turn from its groups,
recomputes centroids and links, and retires groups with fewer than 2 turns. The 40 questions are asked of A0,
A1, A2 and A3. Mechanical checks:

1. no packet contains a forgotten turn;
2. no group lists one;
3. every centroid equals the mean of its current members' vectors (within 1e-6), so no aggregate still holds
   the forgotten turn;
4. links were recomputed after the change.

A forgotten answer counts as *revealed* when the judge says the reply gives it. It may survive in other turns,
and A0 is the reference for how often that happens.

**Scope isolation.** CONFIRM conversations are paired by h order (1–2, 3–4, 5–6, 7–8). Each pair goes into one
store, with project tags P and Q. Discovery and revision run per scope. Mechanical checks: no group holds turns
of two scopes, and `recall` with a project returns no turn of the other scope in any arm.

**Human audit.** The maintainer labels 30 stage-2 judgments drawn by SHA-256, blind to arm and to the judge's
verdict.

**Uncertainty.** Paired differences come with a 95% interval from a bootstrap over the 8 CONFIRM conversations
(5,000 resamples, seed 20261120). Every number is also reported per conversation and per replica.

## Harness tests (written and passing before DEV)

- Determinism: same seeds give the same structure and the same packets.
- Budget: no packet exceeds B, and no turn appears twice.
- The walk ends on cyclic link graphs and respects the 50-pop and 10-operation bounds.
- Forgetting: all four mechanical checks, on a small hand-built store.
- Scope isolation on a two-scope store.
- Rollback: reloading a version reproduces its packets.
- No leakage: the build and revision functions receive no question, answer or evidence id (checked by their
  signatures and by a test that fails if they read the question file).

## What DEV may decide (by these rules only, before CONFIRM)

1. The mapping from category number to question type, by reading questions.
2. The decisive budget: B = 1,024 stays unless A0's full-evidence coverage on relational DEV questions at 1,024
   is at least 90%. In that case B = 512, so the gate is not decided at a ceiling. This looks at A0 alone.
3. Output parsing for the reader and the judge, never prompt content, parameters or arms.
4. The time limit. CONFIRM is estimated at about 2,600 local calls (1,280 reader and 1,024 judge calls in
   stage 2, plus 320 for forgetting). If DEV extrapolates to more than 72 hours of compute, the stage-2 sample
   drops from 8 to 6 per type and conversation.

After DEV, a freeze commit records the script, tests, prompts, model digests, the data hash, the mapping, the
budget and DEV outputs. The runner refuses to open CONFIRM unless that commit is in Git.

## Gates (on CONFIRM, at the decisive budget)

Each delta below is the arm's figure minus the comparator's, in percentage points. "CI > 0" means the lower
bound of the 95% interval is above zero.

**H1, fixed hierarchy (A1) passes if:**
1. relational coverage ≥ A0 + 5, CI > 0;
2. no local type more than 2 below A0 in coverage;
3. relational accuracy ≥ A0 + 5, CI > 0;
4. local accuracy no more than 3 below A0, and adversarial accuracy no more than 3 below A0.

**H2 static, discovered structure (A2) passes if:**
1. relational coverage ≥ A0 + 5 with CI > 0, ≥ A1 + 3, and ≥ A2-shuffled + 5, on the mean of the three
   replicas, with no replica below A0;
2. no local type more than 2 below A0 in coverage;
3. relational accuracy ≥ A0 + 5 with CI > 0, and ≥ A1 + 3;
4. local accuracy no more than 3 below A0, and adversarial accuracy no more than 3 below A0.

The shuffled-control condition separates what was discovered from the mere fact of walking through groups.

**H2 adaptive, revised structure (A3) passes if:**
1. post-cutoff relational coverage ≥ A2 + 3, CI > 0, on the mean of the replicas;
2. no pre-cutoff type more than 2 below A2 in coverage;
3. post-cutoff relational accuracy ≥ A2 + 3;
4. local accuracy no more than 3 below A2, and adversarial accuracy no more than 3 below A0.

**Hard gates. If any fails, no arm passes:**
- forgetting: all four mechanical checks find nothing, and no arm reveals more forgotten answers than A0;
- scope: both isolation checks find nothing;
- judge: the human audit agrees with the judge on at least 27 of 30, otherwise stage 2 is INCONCLUSIVE;
- integrity: CONFIRM was unopened before the freeze commit, models match their digests, every packet is
  logged, and the rollback check passes.

**The strong criterion of the addendum** (useful structure that was not specified, revised by new evidence,
better on unseen tasks than A0 and A1) is met only if H2 static and H2 adaptive both pass. A tie, a gain only in
descriptive counts, or a gain that the shuffled control also shows is not evidence of self-organization.

## What each result changes

- **Nothing passes:** flat retrieval stays. This line is closed for retrieval of this kind, and AtlasKV is not
  opened on its strength. The next memory work is roadmap item 2, the public benchmark against other systems,
  starting from A0's LoCoMo numbers measured here.
- **Only H1 passes:** a time hierarchy helps but discovery does not. A product PR may propose session-first
  retrieval in `recall`, with tests. The self-organization idea is not supported by this data.
- **H2 static passes, H2 adaptive does not:** discovered structure helps, and rebuilding it is enough, which
  agrees with E13-A. A product protocol (issue #165, flags OFF, shadow first) would have the night engine
  rebuild structure as a reversible derived index, under the reflections contract (cascade on forget), for
  `recall` to walk.
- **Both H2 gates pass:** the strong criterion is met. The same product protocol would include versioned
  revisions in shadow, and permanent changes would need human approval.
- **A2 beats A0 but not A2-shuffled:** the gain comes from walking through groups, not from what was
  discovered. It is reported that way and does not count for H2.
- **A hard gate fails:** no integration, whatever the accuracy. Fixing what failed comes first.

Readable labels for groups written by a model, the old question of showing summaries or using them only as
routes, are left out. That question only arises once a structure has earned its place, and the model would
then label it, never build it.

A pass is evidence to propose a product change, not the change itself. The core stays standard-library only:
in a product version, structures would be data written by the engine, not a dependency of `src/infinito3/`.

## Limits known in advance

- Eight CONFIRM conversations are few independent units, and the interval is computed over them.
- LoCoMo's gold answers have known errors. They affect every arm alike, and any the audit meets are reported.
- Conversations are in English, while the core's extractors are Spanish-first. That affects the write filter,
  not ranking, and it affects every arm.
- LoCoMo has no outcomes of actions. The addendum's families of rules that transfer to analogous cases, and of
  patterns contradicted by new conditions, need a task bank with verifiable results. That belongs to the
  agent-learning line (F12, S0) and is not measured here.
- The discovery procedure is fixed by the designer. Only what it finds is not specified in advance.
- One reader model; the judge's family depends on what the maintainer installs.

## Cost and scope

No paid model. The network is used only to download the dataset and the judge model, each approved by the
maintainer before it happens. No personal data, no change to `src/infinito3/`, no deployment. The run happens
on the maintainer's PC, outside the night engine's hours.

Code: `observatory/m1_fractal_memory.py`, with tests in `tests/`. Report:
`docs/INFINITO_M1_FRACTAL_MEMORY_RESULTS_<date>.md`, published whether the result is negative or not.
