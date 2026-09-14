# INFINITO 3.0 — Held-out V3 preregistration

Implementation under test is frozen at `9256992d2ac997bde2ab6a5851f29a9211e9eed7` (`baseline/temporal-cognition-v1-20260914`).

Held-out V3 cases were frozen before first live execution at `cbb05b9a803a6d0ed021f5790d43d6a87e45f426`.

No algorithm changes are permitted before the first V3 result is read.

## Primary decision criteria

Treat the temporal architecture as directionally validated for integration into `infinito-3.0` if all are true:

- mean cognitive answer score >= 0.70;
- mean context score >= 0.85;
- mean answer lift > +0.20;
- baseline wins <= 3 of 28 probes.

Treat it as strongly validated if, in addition:

- mean cognitive answer score >= 0.75;
- mean context score >= 0.88;
- baseline wins <= 2;
- no new catastrophic failure class appears outside known scorer limitations.

Keep it experimental and do not integrate if either is true:

- mean cognitive answer score < 0.65; or
- baseline wins > 5.

## Interpretation rule

The exact substring scorer is known to produce translation/paraphrase false negatives. After the aggregate result is fixed, individual failures may be manually classified as scorer limitations versus genuine architecture defects, but the V3 cases and their expectations must not be edited in response to the run.

## Post-result action

If directional validation passes, integrate the temporal architecture into `infinito-3.0` without merging `master`, preserve the frozen baseline branch, and move the next validation phase to persistent multi-session conversations rather than retuning V3.
