# E13-A reproducibility bundle

Research only. Product code and personal memories are not modified.

- `dev/`: initial development run, before fairness corrections.
- `dev-final/`: corrected shared text and budget adapter, before final disk/audit instrumentation.
- `dev-frozen/`: final DEV measurements and exported index evidence.
- `holdout/`: the ONE confirmatory run, 32 previously unopened worlds.
- `replay/`: deterministic repeat of that SAME consumed partition. Not a second confirmation.
- `audit-holdout.json`, `audit-replay.json`: independent artifact audits.

Every world records the synthetic sources, deletion IDs, complete contexts,
predictions and gold answers, per-system measurements, and provenance indexes.
`summary.json` binds world files by SHA256 and records environment/model hashes.
The original preregistration is `docs/INFINITO_E13A_PREREG_2026-11-05.md`;
development corrections are in `docs/INFINITO_E13A_DEV_NOTES.md`.

## Reproduce the scored comparison

From this repository root, in an isolated Python 3.11 environment (original:
Windows, Python 3.11.9), install `observatory/requirements_e13a.txt`.
The following use `python` from that environment. They never use a real memory
database. Only model preparation needs network access; scoring is local.

```sh
python -m pip install -r observatory/requirements_e13a.txt
python -m src.infinito3.semantic save work/e13-model
python -m pytest tests/test_e13a.py -q
python observatory/e13a.py --partition DEV --model work/e13-model --output work/e13-dev
python observatory/e13a.py --partition HOLDOUT --freeze docs/INFINITO_E13A_FREEZE.json --model work/e13-model --output work/e13-replay
python scripts/audit_e13a.py work/e13-replay --compare observatory_results/e13a/holdout --disjoint observatory_results/e13a/dev-frozen
```

Do not overwrite an output directory. The runner refuses changed code or model
files and requires a committed freeze manifest. Model revision and exported
file hashes are recorded in the DEV notes/freeze. If upstream model files change,
restore the pinned revision/export instead of silently substituting new weights.
Exported auxiliary text bytes can differ across operating systems; confirmation
requires the exact recorded export. Discrete outputs, not wall-clock timings,
are the reproducibility criterion. Model weights are not committed to Git.

To audit committed evidence without downloading a model or installing packages:

```sh
python scripts/audit_e13a.py observatory_results/e13a/holdout --disjoint observatory_results/e13a/dev-frozen
python scripts/audit_e13a.py observatory_results/e13a/replay --compare observatory_results/e13a/holdout
```

DEV and HOLDOUT use SHA256 namespaces documented in the protocol; HOLDOUT is
now consumed in its entirety. New confirmation must use a new preregistered
namespace and frozen policy. There are no LLM API calls or model training here.
