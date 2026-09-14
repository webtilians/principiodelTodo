# INFINITO 3.0 — Temporal cognition V2 probe diagnosis

Frozen implementation baseline: `9256992d2ac997bde2ab6a5851f29a9211e9eed7`
Frozen benchmark cases: `34b2b41922370ccbbcb1ba302113d5b431b22e55`
Artifact SHA256: `c29b12c203c231cd28adedd6a8a36221e3be2c965c647d09234497c5bfcc442b`

## Aggregate result

- 4 trajectories / 163 user turns / 26 probes
- W/T/L: **14 / 10 / 2**
- mean baseline answer score: **0.4346**
- mean cognitive answer score: **0.7551**
- mean answer lift: **+0.3205**
- mean context score: **0.8878**
- deterministic regression suite before paid run: **116 passed**

The event-sourced temporal architecture is a large improvement over the pre-temporal V2 result (5/16/5, cognitive ~0.457, context ~0.599), but the remaining failures are concentrated rather than random.

## The two losses

### `rescheduled goal on target day`

Expected dentist Thursday 17:00. The reschedule event was extracted, but goal resolution returned `goal_lifecycle_target_not_found`. The original Wednesday 11:00 goal therefore stayed open. This is a **goal lifecycle target-resolution failure**.

### `cross-language goal cleanup`

`Cancel Thursday's call with my sister; we already spoke.` produced `CANCEL_GOAL`, but target resolution again returned `goal_lifecycle_target_not_found`. The Spanish goal `llamar a mi hermana` remained open. This is a **cross-language goal lifecycle target-resolution failure**.

## The ten ties

1. `initial profile after noise`: context is perfect (Lucía/Sevilla/Trek Slash) but the answer refuses because the probe says “sin usar el historial reciente”. **Answer-generation/prompt interpretation failure**, not memory failure.
2. `profile negative control`: both answers correct. Context contains an unrelated name memory not prohibited by the current oracle. **Benign context leakage not measured by the oracle**.
3. `five concurrent goals`: no goals reach the ContextPacket for “compromisos ... esta semana”. **Goal-query intent/temporal filtering failure**.
4. `remaining goals after completion cancellation reschedule`: several lifecycle operations had failed earlier, and the weekly query did not expose the correct authoritative set. **Goal lifecycle + query selection failure**.
5. `all initial goals closed`: state/context are effectively correct, but the answer mentions that the workshop is already completed and the substring scorer penalizes the word `taller`. **Scorer false negative / over-strict exclusion**.
6. `cross-language new future goal`: both models answer correctly with Spanish `pasaporte`; oracle requires literal English `passport`. Context is correct. **Scorer false negative**.
7. `historical negative preference audit`: historical context surfaces the old positive `ajedrez` memory, but does not encode that it was later retracted, so the model answers “none”. **Historical-state representation failure**.
8. `preference negative control`: answer correct; one unrelated preference is present but not forbidden by oracle. **Benign context leakage not measured by oracle**.
9. `prompt hygiene negative control`: answer is correct, but self-contained math still receives occupation plus instruction-like remembered data. **Context suppression failure for self-contained queries**.
10. `final current location control`: both are correct; intentional control tie.

## Partial-score probes that are actually scorer limitations

`current revised profile` receives 0.833 even though the answer says `naranja quemado`, which is a correct Spanish translation of expected `burnt orange`. This is a **substring scorer false negative**, not a cognition failure.

## Final open goals

The run ends with 4 open goals:

- `renew passport` — **legitimately open**.
- `call electrician` — **stale**, because English completion failed to resolve the Spanish goal.
- `insurance paperwork` — **stale**, because English cancellation failed to resolve the Spanish goal.
- `call sister` — **stale**, because English cancellation failed to resolve the Spanish goal.

So **3/4 final open goals are lifecycle-resolution defects**; only 1 is intended.

## Architectural diagnosis

The remaining high-value defects cluster into four areas:

1. semantic/cross-language matching for goal lifecycle mutations;
2. goal-query intent and calendar-window selection;
3. explicit representation of historical/retracted state in context;
4. aggressive suppression of cognition context for self-contained requests.

No algorithm changes should be made from this diagnosis before the next independent held-out. The tested implementation is frozen as `baseline/temporal-cognition-v1-20260914` at commit `9256992d...`.