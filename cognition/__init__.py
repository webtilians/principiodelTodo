"""INFINITO's cognitive engine: a hierarchy of models that looks at the person's memory every day, for as long as they
let it run.

    observer      a cheap model: what is new, contradictory, lost or wrong among the memories since the last review
    investigator  a mid model: falsifiable hypotheses about them, each with what would refute it and a check
    supervisor    the strongest model, once a day: reads the whole day itself (not only its children's reports),
                  audits them, judges the hypotheses against the evidence fetched to test them, writes its own
                  reflection and proposes guidelines for each role
    baseline      one cheap single pass over the same day, kept blind in the report, to compare the hierarchy against

The person supervises the supervisor. Its reflection goes to INFINITO's own reflections, where the person reads,
comments on, hides or deletes it, and the supervisor reads those comments the next day. A guideline it proposes is
only a proposal: it changes a role's instructions after the person approves it (python -m cognition approve), never
before, and can be revoked.

The engine is an app connected to the person's INFINITO like any other (its own OAuth session, revocable on the memory
page, best given the level "read and add"). The server cannot open a sealed memory on its own, so nothing here asks
it to: the engine brings its token, as Claude or ChatGPT do. It runs wherever that session is kept (a scheduled task
on the person's computer by default), with a daily budget in euros and no end date; pausing it is the person's call.

Everything it does is written to an append-only diary with a hash chain (diary.py). Memory text is never written
there: only memory ids and what each model said, and those bodies can be purged when the person forgets something.

Standard library only.
"""
