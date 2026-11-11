# INFINITO's cognitive engine

A hierarchy of models that looks at the person's INFINITO memory every day, for as long as they let it run:
an **observer** (cheap) marks what is new, contradictory, lost or wrong; an **investigator** turns it into
falsifiable hypotheses, each with a read-only check the engine runs against the memory; once a day the
**supervisor** (the strongest model) reads the whole day itself, audits its children, judges the hypotheses
against the checks, writes its own **reflection** and proposes **guidelines** for each role. A **baseline**
(one cheap pass alone) is kept blind in the report, to compare the hierarchy against.

The person supervises the supervisor:

- its reflection goes to INFINITO's own reflections (memory page → Reflexiones), where they read, comment on,
  hide or delete it; the supervisor reads those comments the next day;
- a guideline it proposes changes nothing until they approve it, and an approved one can be revoked;
- everything is in an append-only diary with a hash chain, and a daily report.

Design and gates: [docs/COGNITION_ENGINE_PROTOCOL_2026-11-11.md](../docs/COGNITION_ENGINE_PROTOCOL_2026-11-11.md).

## Privacy

INFINITO's server cannot open a sealed memory on its own: only a request that brings a password or a live app
token does. The engine does not change that. It is one more app connected to the person's memory, with its own
OAuth session (shown under Apps on the memory page, where it should get the level "read and add", and where it can
be disconnected). The session is kept in the engine's folder, on the computer that runs it, never on the server.
What the models read is sent to the provider the person chose; nothing goes to any other.

## Use

```bash
python -m cognition login                       # sign it in (your password goes on INFINITO's own page)
python -m cognition models supervisor anthropic <model> --price-in <usd> --price-out <usd>
python -m cognition models observer openai <model> --price-in <usd> --price-out <usd> [--base-url ... --key-name ...]
# ...the same for investigator and baseline; the key in ANTHROPIC_API_KEY / OPENAI_API_KEY / --key-name
python -m cognition review                      # a dry run until you authorize: no model, no writes, no cost
python -m cognition authorize                   # let it spend, within the daily budget
powershell -File scripts/install_cognition_task.ps1   # every hour, with no end date
python -m cognition status | proposals | approve ID | reject ID | revoke ID | rate LABEL util|no | pause | resume
```

Its folder is `~/.infinito/cognition` (or `INFINITO_COGNITION_HOME`): `config.json`, `session.json`,
`diary.jsonl` and `bodies/`, `ledger.json`, `reports/` and `tick.log`.

Budget: 0.50, 0.50, 1, 1, 2, 2, 1 € the first seven days, then 1 € a day (`budget_eur_after`), every day, until the
person pauses it. Each call reserves a high estimate first and is not made if the day cannot hold it; the
supervisor's share is held back from its children. A model without a price in `config.json` is never called.

Standard library only. Tests: `python -m pytest tests/test_cognition_v3.py`.
