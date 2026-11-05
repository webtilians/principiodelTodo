# INFINITO — strict project scope, 2026-11-05

## Origin and reproduction
E13-B draft PR #84 showed that `AgentMemory.recall(query, project="ProyectoNube")` could include a record tagged `metadata.project="TENANT_B_ProyectoNube"` because the name occurred in the record's text. Reproduction: Actions run 42019082859. This was a **same-store project-filter defect**, not proof of cross-account access.

## Minimal correction
- An explicit `project` filter now accepts only records whose **metadata.project** exactly matches the requested project after whitespace trimming and Unicode casefolding.
- An untagged record is excluded from a scoped query even if its content mentions the project. A blank/whitespace-only project cannot select untagged records.
- Unscoped `recall(query)` still performs broad thematic recall; the caller must not treat this as tenant authorization.
- Both `recall(project=...)` and `list_memories(project=...)` use the same filter.
- The authenticated remote server isolates **user account** data in distinct encrypted per-owner stores in `src/infinito3/remote.py`. This patch does not add a second user authorization layer and should not be described as doing so.

## Compatibility / migration risk
Historically, `project` also found untagged personal facts that merely mentioned a client. Such results will no longer appear with an explicit project filter. To find those facts, call unscoped recall and review their source; assign explicit project metadata via a separate controlled migration if needed. Do not auto-retag or copy old memories.

## Tests and gates
- Existing work-memory tests updated to distinguish thematic from strict project filtering.
- Added overlapping-name, projectless, case-insensitive and empty scope regressions, including list_memories.
- CI requirements: `python -m pytest tests/test_*v3.py -q` on the branch/PR; do not claim PASS until workflow completes.
- Independent cross-account verification must also be reviewed before anyone claims a strong security boundary. Existing remote account isolation is a separate layer.

## Research separation
Do not merge automatically; this is a product behavior change requiring maintainer review. E13-B DEV remains blocked and its pre-registered confirmation seeds sealed until a *new* study design explicitly accounts for the corrected interface. No model training or scientific E13-B confirmation is performed here.
