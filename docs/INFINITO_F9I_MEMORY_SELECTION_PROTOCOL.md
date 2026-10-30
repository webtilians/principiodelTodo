# F9-I — Does FIFO actually forget the cue in longer navigation?

**Date:** 29 October 2026. **Stage:** forensic replay + frozen-policy diagnostic, no new training.

## Research question

F9-H produced 95% S7 success and 36/40 both-correct paired cue decisions
using contextual F4, but only 38.75% S13 success and **0/40 both-correct pairs**.
An untested hypothesis says longer experience evicts early cue events from
the 64-slot FIFO. F9-G's scripted S13 routes wrote at most 32 events, below
64; some *actual autonomous* F9-H policies looped hundreds of steps.
We MUST distinguish loss of a cue event from a failure to act upon an
available cue, and from navigation timeouts.

### Frozen historic data (NOT retrained)

Use GitHub Actions run #40925227783 and its four original
f9h-development-seed-{0,1,2,3} artifacts. Validate all eight original
checkpoint SHA-256 values (F4 and zero, pre- and post-PPO), the report
experiment ID and original split. Reconstruct each after-RL F4/normal
trajectory on true MiniGrid Memory S7 and S13 using the recorded
seed, cue inversion and all native actions. Require EXACT same
positions, 64-slot FIFO occupancy, cumulative event count and exact
SHA-256 readout prefix at every step. Any discrepancy blocks conclusions.

### Comparison at equal fixed memory capacity

- **FIFO**: exact F9-G ContextF4 with 64 slots, real Event/Store/FIFO.
- **First16**: SAME 64 slots, payload, event creation and 660 input
  features, but preserve the first 16 *chronological occurrence events*;
  evict oldest remaining event as more are seen. This is a deterministic
  nonlearned contextual retention rule, not an oracle or learned memory.
  It is a diagnostic comparison of event availability, **NOT** a
  matched trained-policy study. Early events may be distractors.

For both strategies, quantify the first event originating from the cue
pixel under a same-pose shadow-world cue inversion, whether it persists
before the LAST policy action, and whether ANY such cue event persists.
An observer alone can identify the cue pixel to measure retention;
the memory selector itself is not supplied with this marker. Do not
pretend that the last action always occurs at the fork, especially when
the episode times out.

### Targeted intervention, NOT a training feature

On the frozen F9-H trained FIFO agent, evaluate:
1. FIFO/normal (replicate the existing baseline).
2. FIFO/cue_one: swap the key/ball channels in the **first stored event**
   arising from the initial cue pixel, and no other event, keeping
   current image, event order, occupancy and all other features intact.
3. FIFO/noncue_one: swap the same two channels in the earliest *noncue*
   stored event (negative control).
4. First16/normal (same checkpoint, different selection distribution).
5. First16/cue_one.

The environmental cue observer is **evaluation-only**. Its role in
selecting which individual readout slot to perturb is explicit;
the policy never receives that privileged label. An intervention
with no retained event is recorded as not applied. Noncue and cue
comparisons do not have matched semantic significance or exact causal
histories without further work, and changing stored identity can be
off-distribution. A drop in performance is evidence of readout
sensitivity, not necessarily semantic memory understanding.

All comparisons are paired across inverted-cue worlds; report
both-correct pairs, side flips, errors and timeouts. Frozen First16
tests are a *distribution shift*, not a fair contest between
independently trained selectors.

### LearnedRelevance: adaptation blocked on valid feedback

The original F4 LearnedRelevance controller learns from a clearly
defined prior query identity delivered **after** the storage decision.
MiniGrid's public observation has **no equivalent ground-truth
query identifier**. Guessing it from success_pos, swapped cue,
target label or the observer would insert oracle supervision into
the cognitive mechanism and defeat the external validation.

Do NOT call the original LearnedRelevance directly with future target
or a label inferred from the test. A separate future F9-J protocol
may define an *observable, training-only post-episode* credit signal
based on reward and self-history and compare it against a
nonlearned selector using identical train budgets and heldout rules.
It would not be scientifically sound to claim LearnedRelevance has
already been evaluated in MiniGrid.

### Decision gate

Only if the frozen replay verifies a measurable difference in retention
should we prioritize expensive retraining of First16 or another
selector. If both FIFO and First16 preserve the cue on nearly all
completed S13 trajectories, investigate long-horizon policy learning,
context encoding or repeated-event distraction rather than storage.

No production deploys, no Work/PC/VPS use, no reserved holdout.
