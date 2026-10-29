# F9-C — Demonstrate actual cue-dependent recall BEFORE evaluating INFINITO (2026-10-29)

## Motivation
The independent MiniGrid Memory S7/S13 audit ([PR #8](https://github.com/webtilians/principiodelTodo/pull/8)) revealed that PPO and PPO+LSTM always picked the lower exit: 54% S7, 46.7% S13. They were **not** using memory. Scaling 100k→300k transitions did not help.

## New objective and diagnostic scope
Build a **transparent, controlled two-observation task** from the *unchanged, third-party* `MiniGrid-MemoryS7-v0` and `MiniGrid-MemoryS13-v0` layouts. This is **not** the original MiniGrid navigation benchmark. It is an isolated *recognition–retention–decision* diagnostic:

1. At time 0, locate the agent at the start-room cue, facing it. Give the usual egocentric 7×7 symbolic image and compass, no privileged cue metadata. The policy emits an action, ignored by the diagnostic's camera transition.
2. At time 1, place the agent at the fork **facing north**, where the **upper candidate object is visibly in front** in the authentic MiniGrid grid and the lower object is behind the camera. Give exactly the same observation type. The start-room cue is behind/away from the view, not visible. The policy chooses **upper or lower**, using a two-action (0 upper, 1 lower) diagnostic interface. **Before training, assert the upper candidate is actually visible for each test seed.**
3. Evaluate the choice according to which object matches the original cue, as specified by the original world's success/failure positions. No labels are given to the policy. Reward +1 for correct choice, 0 otherwise; terminal after two transitions.

Both PPO and recurrent PPO see the same two images and train with the same number of transitions; PPO has no access to past images, LSTM can retain information. This is **a controlled memory assay**, not an honest full-navigation success score. The simulator guides the camera between two observation locations: it does not imply that an agent learned to navigate there. A later stage must test navigation in unmodified S7/S13.

## Pre-registered controls and falsification
- Verify in 100 independently seeded development worlds that the cue is visible **and different** under a paired cue swap in the first image.
- Verify the **decision image is bitwise identical** under a cue swap, while the correct exit flips; this excludes current-frame shortcuts. It must also visibly contain the upper branch object. Without this latter requirement the choice is unknowable even with perfect memory.
- Balance original cue type and correct exit across train and development episodes, report distributions.
- Flip the cue object and matching target **together** while holding geometry, agent position, fork objects and all other observations fixed. The exact same frozen checkpoint runs both conditions; never retrain for counterfactuals.
- Report PPO and LSTM performance separately and **paired decision flips**. A convincing diagnostic requires (on development) LSTM success >= 80% on originals and cue flips >= 80% of paired episodes; random/PPO should be near 50% (do not assume they will be exactly 50% on 1 seed). These are engineering gates, not inferential tests.
- Train only on train RNG seeds (1,000..); evaluate 100 episodes from development seeds (100,000..). All heldout seeds (900,000+) remain unopened. Report all attempts, including failures, and disclose size, CPU, transitions and differences in parameter counts.
- Start with 20k transitions per policy and 1 train seed. If failure, diagnose representation / opportunity to observe / architecture before increasing compute. No tuning to the reserved test.
- Transfer *without retraining* to 30 S13 development worlds with the identical 148-dimensional observation interface and the identical two-action assay; this tests recognition-memory under a new corridor geometry, **not navigation**.

## What this does not demonstrate
No F4/F5/INFINITO module is used. The camera teleport and 2-action fork interface are a designed diagnostic abstraction. Even a successful LSTM in this task **cannot be called competent at full MiniGrid Memory navigation**. Only after establishing true cue-dependent behavior can we return to the 7-action independent environment with a valid curriculum and test F4 Store/FIFO against a matched strong LSTM baseline. A test failure or a collapse to fixed exit is an informative result and must remain in the record.

## Protocol amendment after initial run — information sufficiency (2026-10-29)

The original 20k run (GitHub #40883765649) used an **east-facing fork**; it omitted a test that the agent could see which object stood at the exit. This invalidates any claim about inability to learn memory, because the missing current clue can make the problem unsolvable. A 100k run (#40884217413) already started on that original code and is not scientifically interpretable for the same reason. The amendment above sets the camera **north** and tests that the correct contextual information is available **before** measuring memory. Do not silently merge old and corrected results; use the new branch SHA/checkpoint hashes to distinguish them. This is an exploratory development repair, not heldout tuning.

## Diagnostic positive controls (explicitly supervised)

After verifying start-cue visibility and the upper candidate's visibility
at the north-facing fork, use a rule-based expert that extracts only
**two fixed egocentric pixels from the actual public observation**.
It must solve all original/counterfactual S7/S13 test cases; if it cannot,
the problem still omits necessary information and training should be blocked.
This is an upper-bound *solvability check*, not learned memory or INFINITO.

Separately train a simple **supervised** GRU memory classifier on paired
original/inverted TRAIN split episodes only, using a generic occupancy map
of visible key/ball pixels (two binary channels per cell, 98 input features).
A feedforward supervised policy sees only the final fork frame.
The training label is the exit that matches the initially visible object,
used **only in the cross-entropy loss**. At test time both models see only
visible frames, with no training labels or simulator state.

The supervised control answers a different question than PPO:
**is cue-dependent memory learnable from the observations at all?**
High performance would not mean that sparse-reward PPO or a full-navigation
LSTM has learned the task. Report exact success and cue-swap decision flips
for both models and zero-shot S13 on independent development seeds; never
open the reserved heldout.

This was added after the initial 20k PPO/LSTM negative result, so it is a
*development diagnostic*, not pre-registered confirmation. F4 is not used.
