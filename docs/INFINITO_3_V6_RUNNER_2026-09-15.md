# V6 runner freeze and execution boundary

Branch: `feat/context-intent-contract-20260915`.
This adds evaluation infrastructure only. The candidate and frozen V6 cases are
unchanged. The prior V6 freeze receipt remains the historical receipt for the bank.

The runner uses IntentContextBuilder and IntentEventExtractor with the same clock,
semantic SQLite store, temporal state and goal engine. Baseline and cognitive
histories evolve independently. The lexical evaluator is unchanged. A manifest
pins SHA256 hashes of the cognitive sources, scorer, runner, tests and workflow,
plus the model configuration. The runtime records the exact execution commit.

Validation uses eight synthetic runner tests, never the V6 turns. Default invocation
performs a file/configuration preflight only, without importing or replaying V6:

```sh
python scripts/run_infinito3_trajectory_holdout_v6.py
```

The first real execution still requires explicit user authorization. The canonical
handoff and experimental preregistration require manual approval for paid runs.
There have been no live calls and no V6 candidate replay during runner development.

## Frozen execution protocol

- Answer/event/reranker models: gpt-5.6-luna; embeddings: text-embedding-3-small.
- Answer reasoning: none; auxiliary reasoning omitted, matching V5.
- Direct dependencies: OpenAI SDK 3.14.0, pytest 9.1.1; workflow Python 3.12.
  Transitive dependencies are not fully locked; model aliases may change remotely.
- No automatic SDK retries. Stop on provider errors, missing usage, incomplete
  responses or 1,500 provider requests. This ceiling is NOT a monetary budget.
- Append raw requests/responses and per-turn context, intent diagnostics, events
  and goal states to JSONL. Credentials and exception messages are excluded.
- Preserve partial logs on failure; do not automatically restart or resume.
  Any partial exposure must be disclosed before deciding on another execution.
- Require clean checkout, exact expected commit and a fresh output directory.
- Workflow is dispatch-only: pushing this branch cannot run V6. No merges.
  GitHub may require the workflow on the default branch to expose dispatch;
  availability must be checked before execution, without merging to enable it.

After explicit approval, execute on the frozen revision with the existing secret
through the workflow. The equivalent command is:

```sh
python scripts/run_infinito3_trajectory_holdout_v6.py \
  --authorize-live-v6 --expected-revision FULL_APPROVED_COMMIT_SHA \
  --output-dir trajectory-holdout-v6-results
```

## Outputs and review

`report.json` retains all probe responses, original lexical metrics, per-trajectory
means, final events/goals and usage separated into baseline answers, cognitive
answers, extraction, reranking and embeddings. `report.md` is the original lexical
summary; it does not certify integration. The JSON usage is the complete accounting.
Monetary cost is deliberately null: no price table is frozen; do not present tokens
as currency or omit embedding usage when later calculating cost.

`audit-review.json` requires each of the four empty-context controls to have an
actual packet with `items == []`; missing packets fail. Answer >= .85 and context
>= .90 are reported separately. Every probe gets an unfilled semantic audit entry,
including closure, time aliases and literal-data checks. Review the full trajectory
as well as the answer, and add reviewer identity and rationale. Final goals require
checking legitimacy, not simply counting whether the list is empty.

Integration remains NOT_APPROVED until semantic baseline wins and stale goals are
audited. A lexical pass alone cannot authorize integration, and there is no automatic
merge. V5 is development only for this candidate. If V6 informs candidate tuning,
report that exposure and freeze a new independent bank for the next claim.
