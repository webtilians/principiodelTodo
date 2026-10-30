from typing import Any, Callable, Dict, List, Optional

from .types import LLMMessage, LLMRequest, LLMResponse


class RecordingLLMAdapter:
    """Deterministic adapter for tests, experiments and offline development.

    Every request is recorded. A responder callback can inspect the full request
    and synthesize a response without calling any external model.
    """

    def __init__(
        self,
        responder: Optional[Callable[[LLMRequest], str]] = None,
        *,
        provider: str = "recording",
        model: str = "deterministic-test-model",
    ):
        self.responder = responder or self._default_responder
        self.provider = provider
        self.model = model
        self.requests: List[LLMRequest] = []

    def generate(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        text = self.responder(request)
        return LLMResponse(
            text=text,
            provider=self.provider,
            model=self.model,
            usage={},
            metadata={"recorded_request_index": len(self.requests) - 1},
        )

    @staticmethod
    def _default_responder(request: LLMRequest) -> str:
        for message in reversed(request.messages):
            if message.role == "user":
                return f"echo: {message.content}"
        return "echo:"


class OpenAIResponsesAdapter:
    """OpenAI Responses API adapter with no dependency on the cognitive core.

    The caller supplies an already-created OpenAI client. Keeping client
    construction outside this class avoids secret handling and makes the
    adapter straightforward to replace in tests.
    """

    def __init__(
        self,
        client: Any,
        *,
        model: str = "gpt-5.6-luna",
        reasoning_effort: Optional[str] = None,
    ):
        self.client = client
        self.model = model
        self.reasoning_effort = reasoning_effort

    def generate(self, request: LLMRequest) -> LLMResponse:
        payload: Dict[str, Any] = {
            "model": self.model,
            "input": [
                {"role": message.role, "content": message.content}
                for message in request.messages
            ],
        }
        if request.max_output_tokens is not None:
            payload["max_output_tokens"] = int(request.max_output_tokens)
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}

        response = self.client.responses.create(**payload)
        usage = self._usage_to_dict(getattr(response, "usage", None))
        return LLMResponse(
            text=str(getattr(response, "output_text", "") or ""),
            provider="openai",
            model=str(getattr(response, "model", self.model) or self.model),
            response_id=getattr(response, "id", None),
            usage=usage,
            metadata={"request_metadata": dict(request.metadata)},
        )

    @staticmethod
    def _usage_to_dict(usage: Any) -> Dict[str, Any]:
        if usage is None:
            return {}
        if isinstance(usage, dict):
            return dict(usage)

        result: Dict[str, Any] = {}
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = getattr(usage, key, None)
            if value is not None:
                result[key] = value
        return result
