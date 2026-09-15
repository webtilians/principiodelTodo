# V6 freeze receipt

- Experimental branch: `feat/context-intent-contract-20260915`
- Candidate implementation: `f18876c` (context intent contract, opt-in).
- Frozen bank commit: `efa63dcf80ce85104072a01c5fcdc8fc3feb2dbd`.
- Frozen source: `src/infinito3/trajectory_holdout_v6_cases.py`.
- SHA256: `19c645820733bcf0b1a9f6341f656fe5b6a02fa653770982a87b372170265c1a`.
- Shape: 4 trajectories, 122 user turns, 28 probes, history limit 5.
- Per trajectory: 31/37/26/28 turns; 7 probes each.
- Four strict empty-context controls plus closure and literal-data audit tags.

Order: implementation and development tests committed; bank committed; only then
structural import/hash test added. No runner/workflow or trigger was added and no
V6 candidate replay or live execution has occurred. Do not infer V6 results from
passing structural tests. Do not edit the bank after this receipt.

See INFINITO_3_CONTEXT_INTENT_2026-09-15.md for scope, known limitations and
preregistered evaluation requirements. V5 is development for this new candidate.
The original V5 result remains valid only for its original implementation.

Next: separately implement and freeze a manual V6 runner using IntentContextBuilder
and IntentEventExtractor, including the strict empty-context audit and all usage
accounting. Obtain explicit approval before the first paid run. No merge to master
or infinito-3.0 is authorized by this experiment.
