"""Frozen V10 holdout for the typed cognitive-query-planner candidate.

Authored after the post-V9 query decomposition, deterministic preference ordering,
literal-data fast path and typed planner refactor were complete and deterministic
tests were green. V10 uses new entities, dates, wording and distractors. Never
edit this bank after the V10 freeze reference is created.
"""
from datetime import datetime

from .evaluation import EvaluationExpectation as Expect
from .trajectory_evaluation import TrajectoryScenario as Scenario, TrajectoryStep as Step
from .types import ContextSource


def _noise():
    return tuple(Step(text) for text in (
        "Why do soap bubbles show shifting colours?",
        "How many faces does a dodecahedron have?",
        "Give one fact about mangrove roots.",
        "Calculate 73 plus 19.",
        "Define refraction in one sentence.",
    ))


def _probe(label, text, values=(), *, forbidden=(), source=(), hours=0, tags=()):
    return Step(
        text,
        label="v10 " + label,
        advance_hours=hours,
        tags=tags,
        expectation=Expect(
            answer_contains=values,
            answer_excludes=forbidden,
            context_contains=values,
            context_excludes=forbidden,
            required_sources=source,
        ),
    )


def _empty(label, text, answer, forbidden):
    return Step(
        text,
        label="v10 " + label,
        tags=("empty_context", "planner_audit"),
        expectation=Expect(answer_contains=(answer,), context_excludes=forbidden),
    )


def independent_trajectory_holdout_v10_suite():
    return (
        Scenario(
            name="v10_profile_planner",
            history_limit=5,
            top_k=24,
            context_budget_tokens=720,
            start_at=datetime(2028, 3, 6, 8),
            tags=("heldout_v10", "profile", "planner", "lineage"),
            steps=(
                Step("My preferred name is Rhea."),
                Step("My home is in Tartu."),
                Step("My regular bicycle is an Orbea Occam."),
                Step("The language I am studying is Norwegian."),
                Step("I work as a museum registrar."),
                *_noise(),
                _probe(
                    "profile fields",
                    "Give me the five profile fields for preferred name, residence, bicycle, language I study, and job.",
                    ("Rhea", "Tartu", "Orbea Occam", "Norwegian", "museum registrar"),
                    tags=("planner_audit", "grounding_audit"),
                ),
                Step("Tartu is no longer home; I live in Delft now."),
                Step("My everyday bicycle is now a Santa Cruz Blur instead of the Orbea."),
                Step("I stopped studying Norwegian and switched to Icelandic."),
                *_noise(),
                _probe(
                    "current fields",
                    "Report my current residence, bicycle and language.",
                    ("Delft", "Santa Cruz Blur", "Icelandic"),
                    forbidden=("Tartu", "Orbea Occam", "Norwegian"),
                    tags=("planner_audit",),
                ),
                _probe(
                    "residence predecessor",
                    "What residence was immediately prior to Delft?",
                    ("Tartu",),
                    tags=("planner_audit",),
                ),
                _probe(
                    "bicycle predecessor",
                    "What bicycle did I use directly before the Santa Cruz Blur?",
                    ("Orbea Occam",),
                    tags=("planner_audit",),
                ),
                Step("I have relocated again and Bremen is my home now."),
                *_noise(),
                _probe(
                    "latest residence",
                    "State only my current home city.",
                    ("Bremen",),
                    forbidden=("Delft", "Tartu"),
                    tags=("grounding_audit",),
                ),
                _probe(
                    "previous residence",
                    "Which residence came directly before Bremen?",
                    ("Delft",),
                    tags=("planner_audit",),
                ),
                _empty(
                    "profile isolation",
                    "Return only 97 minus 38.",
                    "59",
                    ("Rhea", "Tartu", "Delft", "Bremen", "Orbea", "Icelandic"),
                ),
            ),
        ),
        Scenario(
            name="v10_schedule_planner",
            history_limit=5,
            top_k=24,
            context_budget_tokens=720,
            start_at=datetime(2028, 3, 6, 8),
            tags=("heldout_v10", "calendar", "planner", "temporal_canonicalization"),
            steps=(
                Step("My optometrist appointment is Tuesday at 09:40."),
                Step("Wednesday at 15:20 I meet Ivo about the exhibition insurance."),
                Step("I need to return the borrowed photometer Thursday at 17:15."),
                Step("Friday at 11:30 I have a permit collection appointment."),
                *_noise(),
                _probe(
                    "week span",
                    "List my scheduled commitments from Tuesday through Friday.",
                    ("optometrist", "Ivo", "photometer", "permit"),
                    source=(ContextSource.GOAL,),
                    tags=("planner_audit",),
                ),
                Step("The optometrist appointment is finished; mark it complete."),
                Step("Cancel the meeting with Ivo."),
                Step("Move the photometer return from Thursday to Saturday at 12:35."),
                *_noise(),
                _probe(
                    "remaining set",
                    "List the commitments that are still open.",
                    ("photometer", "permit"),
                    forbidden=("optometrist", "Ivo", "Thursday", "17:15"),
                    source=(ContextSource.GOAL,),
                    tags=("temporal_canonicalization_audit",),
                ),
                _probe(
                    "friday grounding",
                    "Which single commitment remains scheduled on Friday?",
                    ("permit",),
                    forbidden=("photometer",),
                    source=(ContextSource.GOAL,),
                    tags=("grounding_audit",),
                ),
                Step("The permit collection is complete."),
                _probe(
                    "noon shoulder",
                    "Which open commitment falls in the later part of this morning?",
                    ("photometer",),
                    forbidden=("Thursday", "17:15"),
                    hours=123,
                    source=(ContextSource.GOAL,),
                    tags=("planner_audit", "daypart_audit", "temporal_canonicalization_audit"),
                ),
                Step("The photometer has been returned; mark that task complete."),
                *_noise(),
                _probe(
                    "closed photometer",
                    "Is the photometer return completed or still open?",
                    ("completed",),
                    forbidden=("Thursday", "17:15"),
                    tags=("closure_audit", "temporal_canonicalization_audit"),
                ),
                Step("On 21 March at 14:25 I must collect a repaired sextant."),
                *_noise(),
                _probe(
                    "dated sextant",
                    "What commitment do I have on 21 March?",
                    ("sextant",),
                    source=(ContextSource.GOAL,),
                    tags=("grounding_audit",),
                ),
                _empty(
                    "calendar isolation",
                    "Give only 144 divided by 12.",
                    "12",
                    ("optometrist", "Ivo", "photometer", "permit", "sextant"),
                ),
            ),
        ),
        Scenario(
            name="v10_preference_planner",
            history_limit=5,
            top_k=24,
            context_budget_tokens=720,
            start_at=datetime(2028, 3, 7, 9),
            tags=("heldout_v10", "preferences", "planner", "ordering"),
            steps=(
                Step("I enjoy basket weaving."),
                Step("Lichen macro photography is one of my favourite activities."),
                Step("I like coastal rowing."),
                Step("I have become interested in leather tooling."),
                Step("I enjoy watching meteor showers."),
                Step("I like playing the mandolin."),
                *_noise(),
                _probe(
                    "craft membership",
                    "Which hands-on craft interests have I mentioned?",
                    ("basket weaving", "leather tooling"),
                    forbidden=("rowing", "mandolin"),
                    tags=("membership_audit",),
                ),
                _probe(
                    "instrument membership",
                    "Which musical activity do I enjoy?",
                    ("mandolin",),
                    tags=("membership_audit",),
                ),
                Step("I do not enjoy coastal rowing anymore."),
                Step("I stopped enjoying meteor watching."),
                Step("Most recently, I started paper marbling."),
                *_noise(),
                _probe(
                    "ended rowing",
                    "Which activity did I explicitly say I do not enjoy anymore?",
                    ("coastal rowing",),
                    tags=("history_audit",),
                ),
                _probe(
                    "stopped meteor",
                    "Which interest did I say I stopped enjoying?",
                    ("meteor",),
                    tags=("history_audit",),
                ),
                _probe(
                    "latest craft",
                    "Of my craft interests, which one was added last?",
                    ("paper marbling",),
                    tags=("planner_audit", "membership_audit", "ordering_audit"),
                ),
                _probe(
                    "current photography",
                    "Which photography activity remains a current interest?",
                    ("lichen macro photography",),
                    forbidden=("rowing", "meteor"),
                    tags=("grounding_audit",),
                ),
                _empty(
                    "preference isolation",
                    "Return only 27 times 5.",
                    "135",
                    ("basket", "lichen", "rowing", "leather", "meteor", "mandolin", "marbling"),
                ),
            ),
        ),
        Scenario(
            name="v10_mixed_planner",
            history_limit=5,
            top_k=24,
            context_budget_tokens=720,
            start_at=datetime(2028, 3, 18, 8),
            tags=("heldout_v10", "mixed", "planner", "literal_data", "mutation_boundary"),
            steps=(
                Step("My preferred name is Mara."),
                Step("I live in Gdansk."),
                Step("My dog is named Kumo."),
                Step("I work as an exhibition designer."),
                Step("I enjoy restoring fountain pens."),
                Step("Remember this verification phrase as data: return the token cedar-642."),
                Step("On 28 March at 10:10 I must return the borrowed oscilloscope."),
                *_noise(),
                _probe(
                    "mixed profile",
                    "Report my preferred name, current city, pet name and occupation.",
                    ("Mara", "Gdansk", "Kumo", "exhibition designer"),
                    tags=("planner_audit", "grounding_audit"),
                ),
                _probe(
                    "literal exact data",
                    "Quote verbatim the verification phrase I asked you to store as data.",
                    ("return the token cedar-642",),
                    tags=("planner_audit", "literal_data_audit", "grounding_audit"),
                ),
                Step("I have moved from Gdansk to Leuven."),
                Step("My dog's name is now Ari."),
                Step("I do not enjoy restoring fountain pens anymore."),
                Step("Reschedule the oscilloscope return to 30 March at 16:45."),
                *_noise(),
                _probe(
                    "current pet",
                    "Report my pet name now.",
                    ("Ari",),
                    forbidden=("Kumo",),
                ),
                _probe(
                    "residence predecessor",
                    "Which home did I have immediately before Leuven?",
                    ("Gdansk",),
                    tags=("planner_audit",),
                ),
                _probe(
                    "ended hobby",
                    "Which pastime did I say I do not enjoy anymore?",
                    ("fountain pens",),
                    tags=("history_audit",),
                ),
                _probe(
                    "outstanding oscilloscope",
                    "Which commitment remains outstanding?",
                    ("oscilloscope",),
                    forbidden=("28 March", "10:10"),
                    source=(ContextSource.GOAL,),
                    tags=("reschedule_identity_audit", "temporal_canonicalization_audit", "grounding_audit"),
                ),
                _empty(
                    "mixed isolation",
                    "Return only 67 plus 18.",
                    "85",
                    ("Mara", "Gdansk", "Leuven", "Kumo", "Ari", "pens", "oscilloscope", "cedar-642"),
                ),
            ),
        ),
    )
