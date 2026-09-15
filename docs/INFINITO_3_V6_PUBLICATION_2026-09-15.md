# Publication receipt — INFINITO 3.0 V6 runner

Published to `feat/context-intent-contract-20260915` through the authenticated GitHub connector.
The connector creates new commit metadata, so remote commit IDs differ from local IDs.
Every corresponding Git tree SHA was checked for exact equality. Commit order was
preserved: candidate, V6 bank freeze, structure test/receipt, runner freeze.
The local IDs in earlier receipts remain valid historical identifiers; use this map
for GitHub links and the remote runner revision for the approved execution.

| Stage | Local commit | Published commit |
| --- | --- | --- |
| Candidate | f18876c6ff659c533fa48bd10ad93c9af3eefd0b | d7c85a4c63fcaa4f71b0d5ac29df46d7e1be5589 |
| V6 bank freeze | efa63dcf80ce85104072a01c5fcdc8fc3feb2dbd | 29cc5806f617ddcbc0872bdc80305cf81576b06d |
| Structure verification | 264a8840c642050134a95b0af8cb5fed92af1313 | 390876a201b30e3cffdc670818b6dc7d9ba3599c |
| Runner freeze | 5819c347641843aefc8f8b3ddfc651c4480e0d41 | a2388ce0ddfd4659356b436ec937c9935f4e6e2c |

Approved execution target (authorization still pending):
`a2388ce0ddfd4659356b436ec937c9935f4e6e2c`.

Validation: 166 deterministic tests passed; V6 hash/config preflight passed.
No V6 turns were replayed and no live provider calls were made.
No merge to master or infinito-3.0. This receipt does not authorize a live run.
