from .cognitive_loop import CognitiveLoop
from .cognitive_query import CognitiveQueryOperator, build_cognitive_query_plan
from .literal_grounding import LiteralDataFastPathAdapter

_LITERAL_PREDICATES = {
    "test_phrase", "verification_phrase", "literal_test_phrase",
    "literal_verification_phrase",
}


def planned_literal_value(text, decision):
    plan = build_cognitive_query_plan(text)
    if not plan.has(CognitiveQueryOperator.LITERAL_READ):
        return None
    packet = decision.context_packet if decision is not None else None
    if packet is None:
        return None
    values = []
    for item in packet.items:
        if item.metadata.get("fact_predicate") not in _LITERAL_PREDICATES:
            continue
        value = str(item.metadata.get("fact_value") or item.content).strip()
        if value and value not in values:
            values.append(value)
    return values[0] if len(values) == 1 else None


class PlannedLiteralGroundedCognitiveLoop(CognitiveLoop):
    def __init__(self, engine, llm, **kwargs):
        super().__init__(engine, LiteralDataFastPathAdapter(llm), **kwargs)

    def _make_request(self, text, *, history, decision, max_output_tokens, use_cognition):
        request = super()._make_request(
            text,
            history=history,
            decision=decision,
            max_output_tokens=max_output_tokens,
            use_cognition=use_cognition,
        )
        value = planned_literal_value(text, decision)
        if value is not None:
            request.metadata["infinito_literal_data_value"] = value
            request.metadata["literal_data_fastpath"] = True
            request.metadata["cognitive_query_plan"] = build_cognitive_query_plan(text).to_dict()
        return request


__all__ = ["PlannedLiteralGroundedCognitiveLoop", "planned_literal_value"]
