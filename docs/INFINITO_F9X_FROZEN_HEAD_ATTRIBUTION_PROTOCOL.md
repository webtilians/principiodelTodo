# INFINITO F9-X — Frozen-head information attribution (PREREGISTRATION)

**2026-11-02, before any new-world evaluation.** New branch based on F9-V frozen confirmation, retaining unchanged original F9-V actor, F4, causal protocol and all DEV heads. This is an **exploratory mechanistic attribution**, NOT a new confirmation of F9-V.

## Question / competing explanations
Why does the neural `lex_hc` head choose READ in F9-V only for certain cue/history/replica combinations?
- H_cue: recurrent state signals which cue was shown (key/ball); the head exploits cue identity as a shortcut.
- H_history: recurrent state encodes whether F4 was available post-cue.
- H_value: beyond fixed cue/history associations, the head selects positive causal intervention value `D=+1`.
No interpretation of gradient or perturbation alone establishes semantic encoding or generalization.

## Provenance and frozen objects
- Exactly four original F9-P `ppo_sham_after_rl.zip` actors (F9-O/P artifacts runs `40957255987` and `40960267485`) and four original F9-V `lex_hc.pt` heads from DEV run `41345497955`, with SHA checks via `observatory/f9v_frozen_confirm.py`. Also load the frozen `lex_public` comparator.
- Source code/protocol SHA checked before new data. No retraining of actor, head, F4 writer; no change to original `observatory/f9v_lexicographic_causal_f4.py`, `src/`, `f9q_causal_fork.py` or old weights.
- Head input is `public[0:148] + previous H[0:64] + previous C[0:64]` before F4 read, sigmoid threshold exactly 0.5.

## New worlds, never touched before this experiment
- ATTRIBUTION exploration `9_500_000+10_000*index+j`, `index=0..3`, `j=0..31`. Exactly 32 worlds × two cue inversions × two memory history regimes per actor = 128 states/actor, 512 states total.
- Future potential holdout `9_600_000+...` SEALED; do not implement confirm entry in this experiment.
- F9-V used `8_100_000 / 8_200_000 / 8_300_000+`; F9-W used `9_100_000/9_200_000+`, with `9_300_000+` sealed. These are NOT accessed by F9-X; historical F9 `900_000+`, F9-T `6_300_000+`, F9-U `7_300_000+` also remain untouched.
- Each fresh world is generated once with F9-V's unchanged `one_seed_cases`; the two READ/SKIP continuations and labels D are used exclusively by the off-line evaluator. No experimental head receives `cue`, `regime`, `correct_side` or `D` for a decision.

## Prespecified analyses, no threshold/seed picking
1. **Natural predictions**: count `READ`, `D=+1` captured, `D=-1`/neutral selected, first-arm correctness by actor, cue, regime, and four-case world block. Check `public_fork_equal` for each original/reversed cue pair.
2. **Within-world inverted-cue matched comparison**: identical public frame at fork. Report how often `lex_hc` READ decisions/probabilities differ across opposite cues, especially when only one state needs READ. Identical-public pairs where predictions differ mathematically require H/C to contribute, but alone do not imply abstract memory-value understanding.
3. **Frozen-head input interventions, no actor rollout changes**: compare factual logit and sigmoid decision with a replacement of only H, only C, or both H/C by the opposite-cue paired donor. The donor is the same world, regime and public frame, reducing but not eliminating distribution-shift confounding. Report paired decision flips, gain/loss of D=+1 recall and harmful/neutral READ, and magnitude of logit change for each patch. Full H/C patch should exactly reproduce the other member's prediction as a positive unit check; it is not independent evidence.
4. **Normal vs blackout matched history**: within the same world and cue, where public fork is identical, replace H/C with that from the other regime; quantify effects per history direction and whether the true local D differs. Do not call a difference in head output proof of understanding F4.
5. **Fixed nondeployable cue shortcut control**: observer-only rule `READ iff regime=history_f4_normal and cue=key`. This uses privileged labels and can only provide a diagnostic comparison, never a fair deployable baseline. Compare its READ precision, recall and first-arm correctness with frozen `lex_hc` and `lex_public`.
6. **Local gradient sensitivities**: calculate raw absolute gradient sums of head logit per group (public, H and C), plus mean-baseline integrated gradients using exactly 32 interpolation steps from the mean F9-V TRAIN head input (no new fitting). Report separately per actor and group. Gradients quantify sensitivity, not semantic attribution; correlated H/C and input scaling are limitations.

## Decision / reporting criteria
No scientific PASS/FAIL for semantic representation in this exploratory run. Report all four actors and all regimes, even if no READ. No seed/policy/head selection based on outcomes. Interpret evidence for a simple cue shortcut versus richer state use only by **jointly** inspecting cue/régime strata, paired interventions and cue-rule comparisons. No claim that the head autonomously discovers causal memory needs.
All case-level data, before/after logits and summary JSON must be retained as a GitHub Actions artifact. Results do not authorize opening any old confirmation, and a future study must be registered independently before testing on new holdout.
