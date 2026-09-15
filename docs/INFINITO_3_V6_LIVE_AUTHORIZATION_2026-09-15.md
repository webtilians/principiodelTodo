# First V6 live execution authorization

User explicitly authorized one live V6 execution and probe-by-probe analysis on 2026-09-15 ("Si lo autorizo").

Target: a2388ce0ddfd4659356b436ec937c9935f4e6e2c.
Frozen suite SHA256: 19c645820733bcf0b1a9f6341f656fe5b6a02fa653770982a87b372170265c1a.

The connector has no workflow-dispatch operation. A separate launcher executes only on creation of eval/v6-first-live-20260915, attempt 1. Subsequent pushes and reruns cannot execute this job. It checks out the exact approved revision and its original freeze manifest. No candidate, bank, scorer, runner or frozen configuration changes. No merge to master/infinito-3.0. The wrapper preserves the same Python/direct dependencies, 30-minute timeout, usage logging and artifact retention as the frozen workflow.

Do not automatically retry partial or failed exposure. Preserve the first run's evidence and report it. Approval does not authorize tuning against V6 or integration.
