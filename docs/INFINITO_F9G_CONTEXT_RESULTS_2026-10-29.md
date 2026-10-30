# F9-G results — 2026-10-29

Implementation tested: `74a07556ce99af5e21e760a7e778e0159148b368`.
[Successful GitHub-hosted CI](https://github.com/webtilians/principiodelTodo/actions/runs/40923811076)
and [paired traces/report artifact](https://github.com/webtilians/principiodelTodo/actions/runs/40923811076/artifacts/12919027571).

| Native task | Development pairs | Gate passed | Maximum occurrence events |
|---|---:|---:|---:|
| Memory S7 | 40 | 40 | 26 |
| Memory S13 | 40 | 40 | 32 |

Every pair retains distinct readouts from the first cue observation at every
subsequent sampled observation, through arrival and four native turns at the
fork. Both members observe key and ball; final raw observations are identical
while their memories differ. Zero and erase remove all memory channels; swap
changes only key/ball readout bits, preserving context and current observation.

Five F9-G integrity tests and three F9-F regression tests passed.
Integrity covers all seven actions, unchanged native rewards/termination/info,
fresh reset/replay, egocentric position, temporal order/age, real Store/FIFO
classes, bounded eviction and eventual forgetting.

## Scope
This is a 64-occurrence-slot extension (512 memory channels; 660 total inputs).
It is not the original 3-identity-slot F4. FIFO/Event/Store themselves are
unchanged. Geometry is confined to paired trajectory construction; only raw
public observations enter memory. Routes are fixtures, not autonomous agents.
No claim of policy improvement, learned relevance or unlimited retention.
There were zero training steps and no heldout access. All execution was on
GitHub-hosted runners; no dependency installation or deployment to PC/VPS.

Dependencies: numpy 2.2.6, gymnasium 1.4.0, minigrid 3.2.0.
Trace SHA-256:
`c2c7ec2cb9f0cf0a35945030f7166284c6a5548e4c5dd7117c5d907dd1d5d073`.
Artifact ZIP SHA-256:
`c6dd1f6188b404884ba20be17218371509683100e91336046659a2de130ca1bf`.

Reproduce on an isolated Python 3.11 runner using the workflow
`.github/workflows/infinito-f9g-context-gate.yml`.
Artifacts expire after 30 days; summary and hashes remain here.
