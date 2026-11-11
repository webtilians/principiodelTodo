# Cognitive engine (CE-PILOT-7D-v1): protocol

Written 11 November 2026, before the engine's first live run. It implements the dossier CE-PILOT-7D-v1 the owner
approved that day, with two changes the owner decided the same day: the engine **does not stop after seven days**
(the seventh day is the first evaluation, then it keeps running at `budget_eur_after` until the owner pauses it), and
the owner **supervises the supervisor**, so its reflections are published to INFINITO's reflections where the owner
reads and comments on them.

## Question

Does a daily hierarchy of models (observer, investigator, supervisor) produce hypotheses and reflections about the
owner's own INFINITO memory that the owner finds useful and that hold up against evidence, at a cost of at most
1–2 € a day, better than one cheap pass over the same memories?

## Arms

- **Hierarchy**: observer → investigator (with read-only recall checks) → supervisor, which reads the whole day
  itself, not only its children's reports.
- **Baseline**: one pass of a cheap model over the same inventory, the same day, with no checks and no supervisor.

Both arms' hypotheses appear in each daily report under neutral labels, shuffled. The owner rates the ones they want
(`rate LABEL util|no`) without knowing the arm; the arm stays in the diary until the comparison.

## What is measured, every day and at day 7

Coverage (memories read, cited by each role, cited by none, limits); hypotheses new, supported, refuted, untested;
claims the referee set aside (NOT_IN_EVIDENCE); the supervisor's audit findings; proposals and the owner's
decisions; tokens and euros per role and per day; owner ratings per arm; cost per hypothesis rated useful.

## Gates

At day 7 the owner decides whether it keeps running, changes or stops. GO to keep running needs all of:

1. no memory text, key or prompt in the diary, reports aside (checked by tests and by `verify`);
2. no guideline applied without the owner's approval (the diary shows every approval);
3. every day within its budget (`ledger.json`); any overrun of an estimate is recorded;
4. no access to the memory except through the engine's own app session (no server-side key, no night unlock);
5. the owner judges at least one hypothesis useful and checkable.

The hierarchy is judged better than the baseline only if, over the rated hypotheses, the owner rates a larger share
of the hierarchy's useful; with a handful of ratings this is exploratory, never proof. A technical PASS (it runs,
within budget, without leaks) is reported apart from the owner's judgement of value.

## What would change the plan

- The supervisor's reflections are judged not worth reading, or repeat its children: cut the supervisor's input to
  the reports and spend the budget elsewhere, or stop.
- The baseline is rated as useful as the hierarchy: run the baseline alone, at a fraction of the cost.
- The referee sets aside most of a role's claims: change that role's model before its instructions.

## Limits, said plainly

The referee checks that cited memories exist in what a role was shown; it cannot check that the words match them.
The supervisor sees the day's new memories, its recent reflections with the owner's comments, the task queue and
the checks; it does not see the whole memory, nor what was said in conversations that left no memory. INFINITO's
`reflection_context` lists the latest 30 new memories; past that the engine takes the rest by date from
`list_memories` (the latest 200) and says so in the report when coverage is partial.
