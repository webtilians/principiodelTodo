# F9-B preregistration: causal test of INFINITO F4 memory in an external world

Date: 2026-10-29. State: DESIGN + implementation work; **no claimed improvement**.
Parent implementation: F9-A MiniGrid, PPO and RecurrentPPO (LSTM). Never use F8 synthetic gains as a justification for adapting the external task.

## Hypothesis and scope

The **first mechanism** tested is the actual F4 \`Store\` and \`FIFO\` controller, adapted to egocentric MiniGrid observations as \`Event(key,payload)\`.
This integration **does not implement F4's \`LearnedRelevance\` feedback loop**: MiniGrid Memory offers no honest past-key query feedback matching F4's API. Pretending reward is a future query would be leakage or a different mechanism. F9-B initially tests the causal contribution of **explicit capacity-limited persistent event storage**; whether *learned selective retention* or *predictive retention* helps is a **separate** hypothesis (F9-C) needing a valid feedback protocol.

All policies receive only the original MiniGrid observation image (7×7×3) and compass direction. The F4 adapter extracts object/type/color/state from **already visible** pixels. It stores at most three distinct event identities. The stored memory readout is concatenated to the current input; there is no complete map, hidden regime label, privileged observation, reward-shaped shortcut, or future query key.

## Six prespecified conditions (same environments and training/evaluation seeds)

| ID | Architecture | Memory input | Training | Scientific role |
|---|---|---|---|---|
| A / ppo | PPO feed-forward | current only | independent | memory-less reference |
| B / lstm | RecurrentPPO LSTM | current only | independent | **strong baseline**, may outperform PPO |
| C / ppo_f4 | PPO feed-forward + F4 Store/FIFO | past visible events, 3 slots | independent | explicit-memory ablation against PPO |
| D / lstm_f4 | RecurrentPPO + F4 Store/FIFO | past visible events, 3 slots | independent | complementary value above LSTM |
| E / lstm_zero | RecurrentPPO + constant-zero memory slots | same input width as D | independent | architecture/input-width control |
| F / lstm_f4_corrupt | **same checkpoint as D** | stored events with color labels systematically altered | **no additional training** | within-model causal intervention (OOD caveat) |

The counterfactual F does not learn separate weights. It uses the **same episode seeds and checkpoint as D**. A change in score indicates sensitivity to memory input, not necessarily useful recollection; color-shift intervention can create unusual combinations. Compare with the trained E zero-memory reference and report exact interventions and limitations.

**Primary comparisons**: D minus B and D minus E (success rate) **and** D minus F (paired within-model). Secondary: C minus A, sample efficiency and costs, and separate parameter-matched baselines in future. Record trainable parameter counts, training time and samples; never claim matched compute just because timesteps match.

## Baseline competence gate

1. Start with development-only baselines at 20k, 100k, or 300k transitions/model/seed (predeclared ladder) and ≥5 training seeds for a confirmatory experiment. Record every attempted budget, including unsuccessful runs. \`ppo\` has no memory; on Memory S7 its competent baseline can be a majority-class/chance-level policy. **Do not demand 70% of a memoryless policy**.
2. Before going on to causal ablations, require LSTM mean success ≥70% on development seeds and ≥15 percentage-point advantage over random, across **at least four** independent train seeds. The gate is a pragmatic preregistered threshold, not proof of generalization.
3. If gate fails, report **INCONCLUSIVE — baseline not competent** and revise development protocol explicitly; do not claim INFINITO outperformed a weak baseline. Do not inspect heldout.
4. CI/SMOKE may bypass the gate only with explicit \`--smoke\`; its outputs must remain clearly labeled unfit for scientific conclusions and ineligible for heldout.

## Evaluation policy

- Training seeds, development and heldout seeds retain the partition already used by F9-A. Same task family does **not** prove cross-task transfer.
- Development selection allowed only in the preregistered budget ladder; no tuning to heldout.
- Holdout opened only once, with checkpoints, code SHA and dependency versions fixed; at least **five** train seeds, competence gate satisfied, and explicit manual action. Protect with one-use marker on the frozen run directory. Do not call it confirmation if a heldout run has already been inspected and code retuned.
- At least 500 episodes per training seed on holdout for all six conditions, with episodes paired. Bootstrap *training seeds*, not episodes, to estimate variability (five seeds are a minimum, not high statistical power).
- No qualitative claim of superiority without paired per-seed differences and uncertainty. Define a practical margin e.g. ≥5 percentage points, and require a credible interval that excludes zero before claiming evidence. Detect increased compute cost or learning time, and report it even when success increases.
- Decision: **retain** only on reproducible gain under fair capacity/cost controls; **redesign** if ablation signals that memory is used but generalization is poor; **discard/no improvement demonstrated** if strong baselines tie or beat it.

## Interfaces and disallowed claims

This study evaluates F4 Store/FIFO, not F4 LearnedRelevance, F5 predictor, F6/F8 world model or subjective experience.
Do NOT quietly relabel the F4 key-value store as a learned cortical memory module or use a hard-coded cue detector: all nonstructural visible objects are treated uniformly.
No fine tuning after examining holdout. Baseline comparisons need compute-matched controls before any broad contribution claim.
