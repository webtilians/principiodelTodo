from src.infinito3.literal_grounding import LiteralGroundedCognitiveLoop, literal_value_from_decision
from src.infinito3.types import CognitiveDecision, ContextItem, ContextPacket, ContextSource, LLMResponse, SafetyDecision, SafetyLevel


def _decision(values):
    items = [ContextItem(source=ContextSource.MEMORY, content=value, score=1.0, estimated_tokens=5, memory_id=str(index), metadata={"fact_predicate": "literal_verification_phrase", "fact_value": value}) for index, value in enumerate(values)]
    packet = ContextPacket(query="q", rendered="data", estimated_tokens=10, budget_tokens=200, items=items)
    return CognitiveDecision(input_text="q", safety=SafetyDecision(SafetyLevel.SAFE), gate=None, context_packet=packet)


def test_unique_literal_is_returned_without_delegate_call():
    class Delegate:
        def generate(self, request):
            raise AssertionError("delegate should not run")
    text = "Repeat exactly the verification phrase I asked you to remember as data."
    decision = _decision(["violet token 407"])
    assert literal_value_from_decision(text, decision) == "violet token 407"
    loop = LiteralGroundedCognitiveLoop(object(), Delegate(), history_limit=0)
    request = loop._make_request(text, history=[], decision=decision, max_output_tokens=64, use_cognition=True)
    response = loop.llm.generate(request)
    assert response.text == "violet token 407"
    assert response.model == "literal-data-fastpath"


def test_ambiguous_literal_uses_delegate():
    class Delegate:
        def __init__(self):
            self.calls = 0
        def generate(self, request):
            self.calls += 1
            return LLMResponse("ambiguous")
    text = "What verification phrase did I ask you to remember?"
    decision = _decision(["alpha 12", "beta 34"])
    assert literal_value_from_decision(text, decision) is None
    delegate = Delegate()
    loop = LiteralGroundedCognitiveLoop(object(), delegate, history_limit=0)
    request = loop._make_request(text, history=[], decision=decision, max_output_tokens=64, use_cognition=True)
    assert loop.llm.generate(request).text == "ambiguous"
    assert delegate.calls == 1
