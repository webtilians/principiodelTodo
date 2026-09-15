"""Typed cognitive query planning over the existing ContextIntent contract."""
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

from .context_intent import ContextIntent, resolve_context_intent

_LITERAL_PREDICATES = frozenset((
    "test_phrase", "verification_phrase", "literal_test_phrase",
    "literal_verification_phrase",
))


class CognitiveQueryOperator(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    FALLBACK_RETRIEVAL = "fallback_retrieval"
    READ_FACTS = "read_facts"
    READ_GOALS = "read_goals"
    READ_PREFERENCES = "read_preferences"
    READ_HISTORY = "read_history"
    PREDECESSOR = "predecessor"
    FILTER_WINDOW = "filter_window"
    FILTER_GOAL_STATUS = "filter_goal_status"
    FILTER_FUTURE = "filter_future"
    SEMANTIC_MEMBERSHIP = "semantic_membership"
    ORDER_LATEST = "order_latest"
    LITERAL_READ = "literal_read"


@dataclass(frozen=True)
class CognitiveQueryStep:
    operator: CognitiveQueryOperator
    predicates: Tuple[str, ...] = ()
    value: Optional[str] = None
    window: Optional[Tuple[datetime, datetime]] = None

    def to_dict(self):
        return {
            "operator": self.operator.value,
            "predicates": list(self.predicates),
            "value": self.value,
            "window": [item.isoformat() for item in self.window] if self.window else None,
        }


@dataclass(frozen=True)
class CognitiveQueryPlan:
    query: str
    intent: ContextIntent
    steps: Tuple[CognitiveQueryStep, ...]
    version: str = "cognitive_query_plan_v1"

    @property
    def retrieve(self):
        return self.intent.retrieve

    def has(self, operator):
        return any(step.operator == operator for step in self.steps)

    def first(self, operator):
        return next((step for step in self.steps if step.operator == operator), None)

    def to_dict(self):
        intent = asdict(self.intent)
        intent["predicates"] = sorted(self.intent.predicates)
        intent["window"] = [item.isoformat() for item in self.intent.window] if self.intent.window else None
        return {
            "version": self.version,
            "query": self.query,
            "retrieve": self.retrieve,
            "intent": intent,
            "steps": [step.to_dict() for step in self.steps],
        }


def build_cognitive_query_plan(text: str, now: Optional[datetime] = None):
    intent = resolve_context_intent(text, now)
    steps = []
    if not intent.retrieve:
        return CognitiveQueryPlan(
            text, intent,
            (CognitiveQueryStep(CognitiveQueryOperator.NO_RETRIEVAL),),
        )

    literal = tuple(sorted(set(intent.predicates) & _LITERAL_PREDICATES))
    if literal:
        steps.append(CognitiveQueryStep(CognitiveQueryOperator.LITERAL_READ, predicates=literal))

    if intent.mode == "facts":
        steps.append(CognitiveQueryStep(
            CognitiveQueryOperator.READ_FACTS,
            predicates=tuple(sorted(intent.predicates)),
        ))
    elif intent.mode == "goals":
        steps.append(CognitiveQueryStep(CognitiveQueryOperator.READ_GOALS))
    elif intent.mode == "preferences":
        steps.extend((
            CognitiveQueryStep(CognitiveQueryOperator.READ_PREFERENCES),
            CognitiveQueryStep(CognitiveQueryOperator.SEMANTIC_MEMBERSHIP),
        ))
    else:
        steps.append(CognitiveQueryStep(CognitiveQueryOperator.FALLBACK_RETRIEVAL))

    if intent.historical:
        steps.append(CognitiveQueryStep(
            CognitiveQueryOperator.READ_HISTORY, value=intent.history_cue,
        ))
    if intent.history_cue == "predecessor":
        steps.append(CognitiveQueryStep(
            CognitiveQueryOperator.PREDECESSOR,
            predicates=tuple(sorted(intent.predicates)),
        ))
    if intent.window:
        steps.append(CognitiveQueryStep(
            CognitiveQueryOperator.FILTER_WINDOW,
            window=intent.window,
            value=intent.daypart,
        ))
    if intent.goal_status:
        steps.append(CognitiveQueryStep(
            CognitiveQueryOperator.FILTER_GOAL_STATUS,
            value=intent.goal_status,
        ))
    if intent.future_only:
        steps.append(CognitiveQueryStep(CognitiveQueryOperator.FILTER_FUTURE))
    if intent.ordering == "latest":
        steps.append(CognitiveQueryStep(CognitiveQueryOperator.ORDER_LATEST))

    return CognitiveQueryPlan(text, intent, tuple(steps))


__all__ = [
    "CognitiveQueryOperator", "CognitiveQueryPlan", "CognitiveQueryStep",
    "build_cognitive_query_plan",
]
