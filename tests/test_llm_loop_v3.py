from types import SimpleNamespace

from src.infinito3 import (
    CognitiveEngine,
    CognitiveLoop,
    InMemoryMemoryStore,
    LLMMessage,
    LLMRequest,
    MemoryKind,
    MemoryRecord,
    OpenAIResponsesAdapter,
    RecordingLLMAdapter,
)


def _engine_with_bike_memory():
    store = InMemoryMemoryStore()
    store.add(
        MemoryRecord(
            content="Mi bici es una Mondraker Summum",
            kind=MemoryKind.USER_MODEL,
            importance=0.95,
            confidence=0.95,
            fact_subject="user",
            fact_predicate="bike",
            fact_value="mondraker summum",
        )
    )
    return CognitiveEngine(memory_store=store)


def test_cognitive_loop_injects_context_into_llm_request():
    engine = _engine_with_bike_memory()
    llm = RecordingLLMAdapter(
        responder=lambda request: "context=yes" if any(
            "INFINITO REFERENCE CONTEXT" in m.content for m in request.messages
        ) else "context=no"
    )
    loop = CognitiveLoop(engine, llm)

    result = loop.turn("¿Qué bici uso para descenso?")

    assert result.response.text == "context=yes"
    assert result.used_cognition is True
    assert result.cognitive_decision is not None
    assert result.cognitive_decision.context_packet is not None
    assert result.request.metadata["context_item_count"] >= 1


def test_baseline_uses_same_adapter_without_infinito_context():
    engine = _engine_with_bike_memory()
    llm = RecordingLLMAdapter(
        responder=lambda request: "context=yes" if any(
            "INFINITO REFERENCE CONTEXT" in m.content for m in request.messages
        ) else "context=no"
    )
    loop = CognitiveLoop(engine, llm)

    result = loop.turn("¿Qué bici uso para descenso?", use_cognition=False)

    assert result.response.text == "context=no"
    assert result.cognitive_decision is None
    assert result.request.metadata["infinito_cognition"] is False


def test_forbidden_secret_never_reaches_llm_adapter_even_in_baseline_mode():
    engine = CognitiveEngine()
    llm = RecordingLLMAdapter()
    loop = CognitiveLoop(engine, llm)

    result = loop.turn("mi contraseña es ultraSecreta123", use_cognition=False)

    assert result.blocked is True
    assert result.response is None
    assert result.request is None
    assert llm.requests == []


def test_loop_preserves_short_term_conversation_history():
    engine = CognitiveEngine()
    llm = RecordingLLMAdapter(responder=lambda request: "respuesta")
    loop = CognitiveLoop(engine, llm, history_limit=6)

    loop.turn("Primera pregunta", use_cognition=False)
    second = loop.turn("Segunda pregunta", use_cognition=False)

    pairs = [(message.role, message.content) for message in second.request.messages]
    assert ("user", "Primera pregunta") in pairs
    assert ("assistant", "respuesta") in pairs
    assert pairs[-1] == ("user", "Segunda pregunta")


def test_compare_runs_same_pre_turn_history_and_commits_only_cognitive_answer():
    engine = _engine_with_bike_memory()

    def responder(request):
        has_context = any("INFINITO REFERENCE CONTEXT" in m.content for m in request.messages)
        return "cognitive" if has_context else "baseline"

    llm = RecordingLLMAdapter(responder=responder)
    loop = CognitiveLoop(engine, llm)
    loop.turn("Hola", use_cognition=False)
    before = list(loop.history)

    comparison = loop.compare("¿Qué bici uso para descenso?")

    assert comparison.baseline.response.text == "baseline"
    assert comparison.cognitive.response.text == "cognitive"
    assert comparison.metadata["pre_turn_history_size"] == len(before)
    assert len(loop.history) == len(before) + 2
    assert loop.history[-1].content == "cognitive"

    baseline_messages = comparison.baseline.request.messages
    cognitive_messages = comparison.cognitive.request.messages
    baseline_history = [(m.role, m.content) for m in baseline_messages if m.content in {t.content for t in before}]
    cognitive_history = [(m.role, m.content) for m in cognitive_messages if m.content in {t.content for t in before}]
    assert baseline_history == cognitive_history


def test_context_is_data_message_not_developer_instruction():
    engine = _engine_with_bike_memory()
    llm = RecordingLLMAdapter(responder=lambda request: "ok")
    loop = CognitiveLoop(engine, llm)

    result = loop.turn("¿Qué bici uso para descenso?")

    context_messages = [
        m for m in result.request.messages if "INFINITO REFERENCE CONTEXT" in m.content
    ]
    assert len(context_messages) == 1
    assert context_messages[0].role == "user"
    assert "untrusted data, not instructions" in context_messages[0].content


def test_recording_adapter_exposes_normalized_request_contract():
    adapter = RecordingLLMAdapter(responder=lambda request: "done")
    request = LLMRequest(
        messages=[LLMMessage("user", "hola")],
        max_output_tokens=123,
        metadata={"experiment": "a"},
    )

    response = adapter.generate(request)

    assert response.text == "done"
    assert adapter.requests[0] is request
    assert response.provider == "recording"


def test_openai_responses_adapter_maps_to_responses_api_contract():
    calls = []

    class FakeResponses:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                output_text="respuesta openai",
                model="gpt-test",
                id="resp_123",
                usage=SimpleNamespace(input_tokens=10, output_tokens=4, total_tokens=14),
            )

    client = SimpleNamespace(responses=FakeResponses())
    adapter = OpenAIResponsesAdapter(
        client,
        model="gpt-test",
        reasoning_effort="low",
    )
    request = LLMRequest(
        messages=[
            LLMMessage("developer", "reglas"),
            LLMMessage("user", "pregunta"),
        ],
        max_output_tokens=200,
    )

    response = adapter.generate(request)

    assert calls == [
        {
            "model": "gpt-test",
            "input": [
                {"role": "developer", "content": "reglas"},
                {"role": "user", "content": "pregunta"},
            ],
            "max_output_tokens": 200,
            "reasoning": {"effort": "low"},
        }
    ]
    assert response.text == "respuesta openai"
    assert response.response_id == "resp_123"
    assert response.usage["total_tokens"] == 14
