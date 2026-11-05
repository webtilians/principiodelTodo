# INFINITO — project scope, 2026-11-05

## Origin and reproduction
E13-B draft PR #84 showed that `AgentMemory.recall(query, project="ProyectoNube")` could include a record tagged `metadata.project="TENANT_B_ProyectoNube"` because the name occurred in the record's text. Reproduction: Actions run 42019082859. This was a **same-store project-filter defect**, not proof of cross-account access.

## Minimal correction
- A record tagged with a project belongs to that project alone: it is returned only when **metadata.project** exactly matches the requested project after whitespace trimming and Unicode casefolding, whatever its text names.
- An untagged record is returned when its content names the project as a whole word, so "Ana" does not match "mañana". A blank/whitespace-only project selects nothing.
- Unscoped `recall(query)` still performs broad thematic recall; the caller must not treat this as tenant authorization.
- Both `recall(project=...)` and `list_memories(project=...)` use the same filter.
- The authenticated remote server isolates **user account** data in distinct encrypted per-owner stores in `src/infinito3/remote.py`. This patch does not add a second user authorization layer and should not be described as doing so.

## Compatibility / migration risk
A first version of this fix filtered by metadata only. It was changed before merging: most of a real person's memory has no tag, so "where were we with DH Trails?" answered "No memories about that" for facts that name DH Trails. Untagged records still match by name, as they did before, and no old memory needs retagging. Records tagged with a different project no longer match by text; to find them, call unscoped recall.

## Tests and gates
- Added regressions for overlapping names, another project's record that names this one, untagged records, whole-word names, case and empty scope, including list_memories.
- CI requirements: `python -m pytest tests/test_*v3.py -q` on the branch/PR; do not claim PASS until workflow completes.
- Independent cross-account verification must also be reviewed before anyone claims a strong security boundary. Existing remote account isolation is a separate layer.

## Research separation
Do not merge automatically; this is a product behavior change requiring maintainer review. E13-B DEV remains blocked and its pre-registered confirmation seeds sealed until a *new* study design explicitly accounts for the corrected interface. No model training or scientific E13-B confirmation is performed here.
