# F9-H — Comparison of contextual F4 and matched empty memory

**Status:** exploratory development protocol, 29 October 2026. This
comparison tests the **F9-G contextual Event/Store/FIFO** from PR #13,
not the unintegrated F4 LearnedRelevance controller or predictive F5.

## Question

Does a recurrent seven-action MiniGrid Memory S7 agent use contextual
stored observations to choose the correct exit, beyond a matched LSTM
with 512 empty extra features?

The F9-G native teacher-trajectory gate established that a 64-event
store can distinguish both cue variants up to a decision location
in 40/40 development pairs on S7 and 40/40 on S13. That shows
information *availability*, NOT that an autonomous policy learns it.

## Training arms, equal budget and architecture

1. LSTM + ZERO: RecurrentPPO MlpLstmPolicy, 660-dimensional inputs
   comprising 148 original partial-view features and 512 zero channels.
2. LSTM + F4: exactly the same architecture, dimensionality, optimizer
   settings and seed; the 512 channels are the contextual FIFO/Store
   output of F9-G (up to 64 events, 8 channels each).

Both receive identical native seven-action training demonstrations:
600 train worlds per independent seed, two cue variants per world,
40 behavior-cloning epochs, then 20,000 requested RL interactions.
Four independent training seeds (0..3), separate exact training and
development partitions. PPO receives original rewards with **no shaping
or camera teleport**. For honest comparison, report exact parameter
counts and actual RL timesteps for each run.

Teacher can use map/target when producing TRAIN action labels, but
neither actor nor its F4 writer or critic can read privileged labels,
global position, success/failure positions or target identity.
The RecurrentPPO LSTM still has its own hidden state; a measured F4
advantage would be **additional memory** beyond LSTM, not proof of
memory independent of recurrent dynamics.

## Original and counterfactual evaluation

Development-only paired worlds 10 per training seed/size (40 total);
run *both* original and swapped initial cue with the same frozen policy.
For S7 and zero-shot S13, report:
- original success, swapped success and combined success;
- number of pairs with **both** outcomes correct;
- number of correct, valid exit-side changes under swapped cue;
- cue visibility, memory writes/occupancy, steps, timeouts;
- pre-RL and post-RL checkpoints, SHA-256 and source/dependency metadata;
- per-episode compact action, pose and memory-readout hash traces.

Post-RL F4 ablations use the **same trained checkpoint** with:
- ERASE: the F4 storage continues updating, output memory channels
  are all zero, same current observation and same model weights;
- SWAP: swap key/ball identities only in stored readout, not in current
  observation (an out-of-distribution control, not a semantic oracle);
- MASK_CUE: redact pixels changed by a paired cue flip in both raw
  observation and writer input; observer-only shadow environment
  identifies them and never provides positions or targets to policy.

The ZERO model receives a matched MASK_CUE control too. These
ablation comparisons must distinguish causal dependence from
general out-of-distribution degradation. A 50% success rate with
0 valid cue flips is *not* evidence of memory even if CI succeeds.
Any improvement must be replicable across independent training
seeds and accompanied by correct paired choices and a relevant
ablation effect, not just mean reward.

## Stage gates and scope

Start with an integrity run and a tiny smoke budget; its scores are
**not scientific results**. Do not run the four-seed comparison if
integration or causal tests fail. For four-seed development runs,
keep all attempts and unsuccessful seeds, and report the whole sample.

Only development seeds may be used. The preexisting F9-B reserved
heldout partition must remain unopened and **no F9-H result here is
eligible for heldout**. Avoid changing original F9-B protocol or
retroactively using it as confirmation for the adapted F9-G input.

This experiment does NOT test LearnedRelevance, F5, consciousness,
scientific general intelligence or broad INFINITO superiority.
No production portal, VPS or user's Windows PC is modified.
