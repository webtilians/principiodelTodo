# F9-G: observational context, before learning

Stack: F9-G -> PR #12 F9-F -> PR #11 F9-E. Development only.
No heldout, model training, checkpoint access, PC deployment or VPS access.

## Explicit extension, not original F4 performance
Uses the unchanged Event, Store and FIFO classes from selective_memory.py.
F9-E had 3 deduplicated identity slots / 12 channels. This variant has **64
occurrence slots / 512 channels**, hence 660 total inputs including the original
148. Capacity is increased deliberately to test contextual information retention
before optimizing memory size. This is not LearnedRelevance, F5, a learned
selector, nor evidence that the old three-slot controller works.

Every visible key/ball pixel generates an Event on every observation. Event.key
is a local occurrence serial. Its integer payload losslessly packs identity,
color, image x/y and observation tick. FIFO evicts the oldest occurrence with no
cue preference. There is no side cache of observations or cue identities.
Readout slots expose key/ball bits, color, normalized egocentric image x/y,
age/(age+1), FIFO rank, occupancy. Absolute observation tick is not exposed.
No global coordinates, object role, target, simulator state, reward, action,
episode seed, or labels enter the writer. The original public compass remains
in raw input, but does not enter memory.

This finite horizon must be reported: repeated observations consume slots and
eventually erase the cue. The eviction test explicitly demonstrates forgetting.
Passing the gate supports only retention on these specified trajectories, not
arbitrary delay robustness or navigation success.

## Controls and gate
f4/zero/erase/swap all have 660 inputs. Zero disables writes and emits zeros.
Erase keeps the exact store and zeros only readout. Swap exchanges every
occupied slot's key/ball bits and preserves all context and raw observation.
It is a global readout intervention, distinct from swapping only the world cue.
No policy is trained in this change.

For each native MiniGrid Memory S7/S13 use 4 historical development indices x
10 episodes = 40 paired worlds per size. A pair has the same seed, layout,
start pose and sequence of native actions. Only the starting cue and associated
native success/failure condition change. The fixture checks the full grids
differ at exactly that cue cell.

An observer-only BFS visits the cue and fork, followed by four native turns at
the fork to expose both candidate objects. It may use geometry to construct the
fixture; it is **not an autonomous policy** and its coordinates never enter the
writer. Require nonidentical readouts at the first cue observation and EVERY
subsequent observation through the fork scan, both key and ball observed by
both members, and identical final raw observations with distinct memories.
Assert that the trajectory does not exceed the declared retention budget.
Negative zero/erase and context-preserving swap are checked at every step.
All 7 original actions (0 through 6) are separately compared against an
unwrapped counterpart for raw observations, rewards, termination and info.
Reset tests check store, serial, clock and exact fresh-instance replay.

CI uploads all paired readout traces plus report.json with source SHA,
dependency versions and SHA-256 of traces. Failures stop the job. There is no
training step to accidentally bypass this gate. A successful representation
gate alone does not authorize claims about learning or causal policy benefit.
