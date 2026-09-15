# V6 R2: output protocol correction

The user requested fixing execution errors and executing the tests on 2026-09-15.
This authorizes the corrected evaluation; no merge to master/infinito-3.0.

V5's adapter consumed response text without treating length truncation as a fatal
provider error. The new V6 audit wrapper added a blanket fatal check while keeping
the bank's 96-token budget. The first attempt stopped on a filler explanation.
That was an evaluation-protocol regression, not a demonstrated cognitive failure.

R2 is a separate runner and manifest. Original runner, bank, scorer and first-run
evidence remain unchanged. No context-intent algorithms are changed.

Before every provider call, R2 applies category-wide output floors: 1024 answer
tokens (identical for both arms), 2048 extraction tokens, 512 reranker tokens.
These are maxima, not targets; actual usage is recorded. No query-specific rules,
expected-answer inspection or bank edits. Models/reasoning settings unchanged.

A nonempty answer truncated specifically by max_output_tokens is logged and
retained as received, without retry. The trajectory continues, lexical scores
remain unmodified, and answer_completion_gate fails. Genuine provider failures,
missing usage, empty incomplete responses and incomplete structured auxiliary
responses still stop the run with partial evidence. This does not suppress errors
to make a test green or permit incomplete structured state to be ingested.

The revision is frozen before a new live run. Deterministic tests use synthetic
inputs only and cover both-arm budgets, explicit truncation warnings, fatal
auxiliary truncation, provider failure, usage, audit gates and report writing.

V6 has six user inputs exposed by run 34984319827 and no probes previously scored.
R2 records this exposure and is NOT the first clean V6 execution. All quality
and semantic integration gates still apply. No automatic merge or whole-run retry.
