# Guidance for coding agents

INFINITO is a memory layer for LLM assistants (`src/infinito3/`) plus a research
lab (`observatory/`, `lab_portal/`, `docs/`). Read [README.md](README.md) first.

## Priorities

1. The product core comes first. The current milestone is closing the gap with
   the store-everything semantic RAG baseline (see the Evaluation section of the
   README and [docs/FINDINGS.md](docs/FINDINGS.md)).
2. Every change to `src/infinito3/` keeps `python -m pytest tests/test_*v3.py`
   green and adds tests for new behavior. The core stays standard-library only.
3. Claims of improvement need a comparison against the RAG baseline at equal
   token budget, on data not used to tune the change.

## Rules

- Never store or send secrets; never weaken `SensitiveInformationFilter`.
- Do not start a new research experiment without a written protocol in `docs/`
  that states the question, the baseline, the pass/fail gate and what result
  would change the plan. Prefer one larger experiment over many small follow-ups.
- Never reuse sealed or consumed seed partitions as new confirmations.
- `legacy/` is archived: do not build on it or revive its "consciousness" metrics.
- Branch from `master`, open a pull request against `master`; the maintainer merges.
