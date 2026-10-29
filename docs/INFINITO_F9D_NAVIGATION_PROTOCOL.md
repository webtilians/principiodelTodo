# F9-D: autonomous cue acquisition, navigation and memory

Base verified 2026-10-29: PR #9 merged at 15:21:30 UTC into
`experiment/f9b-behavior-audit`, commit
`bb0508bea461f82c5ea067def25fad70fb40efe8`.
Default `master` remains `9a2da8afe9d94f526e316b631344af3d82b45ed4`.
This continuation targets the actual merge branch.

## Audit

- F9-C visible-cue assay teleports the camera, scores two actions. Its supervised
  GRU sees only two frames and predicts a binary label. Those are useful memory
  controls, not native navigation results.
- F9-B supplies seven-action partial-view baselines and a genuine F4 FIFO/Store
  adapter. Its color-corruption intervention is weak for Memory: cue and both
  candidate objects are green. A future memory control must intervene on object
  identity/order or erase the store, and acknowledge distribution shift.
- Native Memory resets can place the agent down the corridor, facing east. Cue
  acquisition is therefore an active navigation problem.
- Existing global ci.yml only targets main/develop. New isolated Actions workflow
  explicitly targets this experiment; it does not deploy or run on self-hosted hardware.
- Historical seed_for admits indices whose train/development ranges overlap.
  This experiment restricts indices to 0..3, train worlds to <=5000 and evaluation
  to <=100; keeps the original assigned ranges, never opens 900000+ holdout.

## Frozen exploratory design

Four training seeds. Each sees 600 native S7 worlds and both cue variants.
A privileged BFS teacher generates complete native action trajectories:
navigate to the cue viewpoint using actual turns/forward actions, observe it,
navigate to fork, choose correct exit. The teacher uses geometry/target to
produce training labels; this is privileged supervision, explicitly not
observation-only teacher discovery. Cue visibility is asserted.

The *student* receives only the normalized 7x7x3 public partial view and compass
(148 numbers). Teacher labels enter cross-entropy loss only. No positions,
cue identity, success_pos, time, info or reward enter actor/critic inputs.
All seven native actions remain available; no action mask, camera relocation,
forced prefix, shortened horizon or shaping reward in student rollouts.

Curriculum: full-trajectory behavior cloning (40 epochs), then 20k native sparse
reward transitions using actual SB3 PPO or RecurrentPPO. No auxiliary BC during
RL. Controls: both corresponding scratch-RL agents; evaluate BC before and after
RL, random, cue-masked and recurrent-state-reset policies.
Sequence BC carries gradients across entire demonstrations; masks padding and
resets recurrent states independently for each episode. Same demonstrations and
RL transitions across reactive/recurrent conditions; parameter counts reported,
not claimed compute matched.

Evaluation uses 10 paired development worlds per size/seed in the initial CI
budget. S13 is zero-shot length transfer, never training or sealed holdout.
The original member retains native reward/dynamics. The swapped member changes
only the cue and consistent reward target; it is a causal diagnostic, not an
untouched benchmark episode. Observer cue visibility compares same-pose frames
under reversible cue intervention. Traces include positions, actions, frame
hashes, cue visibility, native reward, timeout, side and success; labels are
observer-only. Count side flips only when BOTH rollouts choose actual exits.

Cue masking zeroes the same changed image cells in paired worlds. State reset
at every step is an OOD diagnostic, not a separately trained reactive baseline.
Reactive agents may encode information in their physical pose/action history;
a reactive success is not automatically a memory leak.

## Gates and next experiments

Initial gate, on all four seeds after BC+RL: each cue condition >=70%, >=7/10
pairs both correct, >=15 point advantage over random and over both cue masking
and state reset. This small development screen is NOT a significance test.
If passed, repeat 100 paired development episodes with frozen checkpoints before
running F4 comparisons. No holdout eligibility is granted by this workflow.

F4 follow-up must use real `observatory.selective_memory.FIFO/Store` and the
F9-B adapter, with matched extra-channel zero baseline, store erasure, and
object-identity/order corruption on the same checkpoint. Train with identical
BC/RL budgets; report parameter/runtime differences, raw per-seed paired
differences and uncertainty by training seed. Do not label FIFO as LearnedRelevance
or F5: MiniGrid provides no valid future-query feedback for that component.
If baseline fails, F4 remains blocked; diagnose acquisition, navigation,
retention and BC-to-RL forgetting separately before spending larger budgets.

No local PC/VPS changes. GitHub-hosted CPU only. Checkpoints, traces, dependencies,
source and checkpoint hashes are artifacts. No result is asserted until CI logs
are inspected.
