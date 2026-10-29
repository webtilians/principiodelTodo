# F9-I: selective event memory for long trajectories

Exploratory development only. Based on F9-H commit 575c4522cdafa7d0f9742fc6456d1a16c46b6a73; PR #14 is unmerged, so this PR is stacked on it.

## Preassigned design (before running F9-I)
Four training indices 0–3; seed mapping unchanged from checked_seed:
train 1000 + index*10000 + episode (0–599);
development 100000 + index*10000 + episode (0–9).
Both cue versions of each world, S7 and S13. No heldout entry point.
Selection RNG 81000 + index, independent from model/environment RNG, assigned before outcomes.
Every condition sees identical train worlds, imitation actions, minibatch seed and PPO budget.
Random reservoir is the stochastic retention baseline. No candidate chosen by development outcome.

For each capacity 64 (primary F9-H budget) and 8 (preassigned stress control):
zero, FIFO, contextual diversity, random reservoir. Same 660 input width, same
LSTM/actor/critic parameters, 600 paired S7 training worlds, 40 imitation epochs,
20000 requested PPO steps (actual rounded steps recorded).
S13 is zero-shot; no S13 labels for training or selector tuning.
Readout is padded to 512 channels at capacity 8. Across-capacity comparisons also
change age/order occupancy distributions and are sensitivity checks.
Native reward, transition logic and all seven actions unchanged; no teleport,
shaping, action restriction, PC/VPS execution or updates.

## Context selector
Only stored/incoming packed public observations: egocentric x/y, color, age.
Prefer context coverage: on overflow, evict newest event among contexts represented
more than once; if all are distinct, evict oldest. May reject incoming event.
Earliest evidence of a repeated context survives. Neither key/ball identity,
cue visibility, correct side, map, future trajectory nor reward enters priority.
Rule fixed in advance, task-informed and not a learned semantic relevance model.
Same Event/Store and exact event budget as FIFO; no persistent payload side cache.
The encoder's new_events list is transient evaluator telemetry and never feeds
selection or actor; emptied each observation. LSTM capacity is identical.

## Diagnosis and intervention
Re-evaluate original F9-H f4_after_rl checkpoints from run 40925227783, exact
10 development pairs per size/seed. FIFO64 parity against original encoder is
tested. Record first cue event, all cue-event IDs, retention and number of
events since first cue at every step and first arrival at branch corridor.
Cue provenance uses a same-pose swapped-cue shadow environment ONLY in evaluator.
No labels/IDs/provenance are actor or selector inputs.

corrupt_initial swaps ONLY identity channels of the earliest actually observed
cue event when it is still retained AND the cue is no longer visible.
It changes no other stored occurrence, raw image, payload, age, order or LSTM.
If missing/evicted it is a recorded no-op; no reinsertion. Repeated cue
occurrences can remain informative: null effect does not show non-use of ALL
cue memory. Report intervention coverage explicitly. erase and mask_cue are
separate controls on the same frozen checkpoint; no retraining for ablations.

Diagnose first-event eviction separately from loss of ALL cue occurrences.
Report conditional fork denominators, cue exposure, branch reach, timeouts,
native return/length, paired both-correct and valid exit flips. Failure before
branch arrival cannot be attributed to branch memory loss. Retention is not
evidence of policy use; post-intervention trajectory changes are downstream
effects, not direct edits to other memories.

## Real LearnedRelevance feasibility
The actual observatory.selective_memory.LearnedRelevance has eight categorical
key weights and updates discounted demand ONLY with a queried_key after the
answer. F9-G/H event IDs are unbounded occurrence serials; passing serial 8
can raise IndexError once eviction evaluates its priority. Reusing eight keys
would overwrite events and violate equal event capacity/semantics.
Native MiniGrid terminal reward provides no requested historical event ID.
Deriving queried_key from hidden cue position would be an oracle even ex post.
Thus do NOT relabel this contextual heuristic as LearnedRelevance.

A future adapter must separate unique occurrence IDs from a bounded observable
context category, override priority to score that category, and obtain actual
post-decision observable retrieval-query feedback. Train that demand model on
train-only episodes, freeze before development, log timing and feedback source.
Scalar reward alone requires a different credit-assignment learner, not the
current feedback API. No learned arm is presented here because that observable
feedback contract has not been established. Tests demonstrate the real API's
serial mismatch and frozen-feedback behavior; its benchmark main is never run
(it would open a separate heldout).

## Interpretation
F9-H logs verified: S7 F4 95%, zero 50%; S13 F4 per seed 50/25/45/35%,
mean 36.25%, zero 50/50/5/25%, mean 32.5%. Whole-memory swap did not reduce S7
F4 success. Source: https://github.com/webtilians/principiodelTodo/actions/runs/40925227783
Do not extrapolate 95% to S13, assert FIFO causality before auditing eviction,
or equate technical CI with cognitive benefit. Four seeds are exploratory;
report all seeds and matched differences, not pooled episodes as independent
training replicates. No success threshold is used to select a method.
