"""Strong grounding for an explicitly requested, uniquely retrieved literal datum."""
from .cognitive_loop import CognitiveLoop
from .context_intent import resolve_context_intent
from .types import LLMResponse


_LITERAL_PREDICATES = {
    "test_phrase", "verification_phrase", "literal_test_phrase",
    "literal_verification_phrase",
}


def literal_value_from_decision(text, decision):
    intent = resolve_context_intent(text)
    if not (set(intent.predicates) & _LITERAL_PREDICATES):
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


class LiteralDataFastPathAdapter:
    def __init__(self, delegate):
        self.delegate = delegate

    def generate(self, request):
        value = request.metadata.get("infinito_literal_data_value")
        if isinstance(value, str) and value:
            return LLMResponse(
                text=value,
                provider="infinito",
                model="literal-data-fastpath",
                usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                metadata={"literal_data_fastpath": True},
            )
        return self.delegate.generate(request)


class LiteralGroundedCognitiveLoop(CognitiveLoop):
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
        value = literal_value_from_decision(text, decision)
        if value is not None:
            request.metadata["infinito_literal_data_value"] = value
            request.metadata["literal_data_fastpath"] = True
        return request
