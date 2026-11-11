"""The models, behind one call: complete(system, prompt, max_output) -> Reply(text, usage, finished).

Two kinds of API: Anthropic's (Claude) and OpenAI's chat completions, which OpenAI and many others speak (DeepSeek,
Qwen on DashScope, OpenRouter...: base_url and key_name in the role's settings). Keys come from the environment
(ANTHROPIC_API_KEY, OPENAI_API_KEY or the key_name given) and go only into the request header: never into the diary,
a report or an error. An error carries one of a few fixed codes, never the provider's own message, which can echo
what was sent.
"""
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}
TIMEOUT = 300
MAX_BYTES = 4_000_000


class ProviderError(Exception):
    """A fixed code: missing_api_key, http_<status>, transport_error, invalid_response."""


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    thinking: int = 0

    def as_dict(self) -> dict:
        return {"input": self.input, "output": self.output, "thinking": self.thinking}


@dataclass
class Reply:
    text: str
    usage: Usage = field(default_factory=Usage)
    finished: bool = True  # False when the model stopped at the output limit: paid for, but not usable


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ProviderError("transport_error")


def post_json(url: str, headers: dict, body: dict, timeout: float = TIMEOUT) -> dict:
    request = urllib.request.Request(url, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.build_opener(_NoRedirect).open(request, timeout=timeout) as response:
            raw = response.read(MAX_BYTES + 1)
    except urllib.error.HTTPError as error:
        raise ProviderError(f"http_{error.code}") from None
    except ProviderError:
        raise
    except Exception:  # noqa: BLE001  # its message may hold the URL or what was sent
        raise ProviderError("transport_error") from None
    if len(raw) > MAX_BYTES:
        raise ProviderError("invalid_response")
    try:
        return json.loads(raw)
    except ValueError:
        raise ProviderError("invalid_response") from None


def _key(provider: str, env) -> str:
    key = (env or os.environ).get(KEYS[provider], "")
    if not key:
        raise ProviderError("missing_api_key")
    return key


class Anthropic:
    def __init__(self, model: str, env=None, post: Callable = post_json):
        self.model, self.key, self.post = model, _key("anthropic", env), post

    def complete(self, system: str, prompt: str, max_output: int) -> Reply:
        data = self.post("https://api.anthropic.com/v1/messages",
                         {"x-api-key": self.key, "anthropic-version": "2023-06-01"},
                         {"model": self.model, "system": system, "max_tokens": max_output,
                          "messages": [{"role": "user", "content": prompt}]})
        try:
            usage = Usage(int(data["usage"]["input_tokens"]), int(data["usage"]["output_tokens"]))
            text = "".join(part.get("text", "") for part in data["content"] if part.get("type") == "text")
            return Reply(text, usage, data.get("stop_reason") == "end_turn")
        except (KeyError, TypeError, ValueError):
            raise ProviderError("invalid_response") from None


class OpenAICompatible:
    """OpenAI, or any server that speaks its chat completions (base_url in the role's settings)."""

    def __init__(self, model: str, env=None, post: Callable = post_json, base_url: str = "https://api.openai.com/v1",
                 key_name: str = "OPENAI_API_KEY", effort: Optional[str] = None):
        self.model, self.post, self.base_url, self.effort = model, post, base_url.rstrip("/"), effort
        self.key = (env or os.environ).get(key_name, "")
        if not self.key:
            raise ProviderError("missing_api_key")

    def complete(self, system: str, prompt: str, max_output: int) -> Reply:
        # No response_format: not every reasoning model takes it, and roles.parse() finds the JSON in the text.
        body = {"model": self.model, "max_completion_tokens": max_output,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]}
        if self.effort:
            body["reasoning_effort"] = self.effort
        data = self.post(f"{self.base_url}/chat/completions", {"Authorization": f"Bearer {self.key}"}, body)
        try:
            usage = data["usage"]
            reasoning = int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))
            choice = data["choices"][0]
            return Reply(choice["message"].get("content") or "",
                         Usage(int(usage["prompt_tokens"]), int(usage["completion_tokens"]) - reasoning, reasoning),
                         choice.get("finish_reason") == "stop")
        except (KeyError, TypeError, ValueError, IndexError):
            raise ProviderError("invalid_response") from None


def make(settings: dict, env=None, post: Callable = post_json):
    provider = settings.get("provider")
    if not provider or not settings.get("model"):
        raise ProviderError("no_model_chosen")
    if provider == "anthropic":
        return Anthropic(settings["model"], env, post)
    if provider == "openai":
        return OpenAICompatible(settings["model"], env, post, settings.get("base_url", "https://api.openai.com/v1"),
                                settings.get("key_name", "OPENAI_API_KEY"), settings.get("reasoning_effort"))
    raise ProviderError("unknown_provider")


class Fake:
    """No model: answers in each role's shape, citing the first memory ids it is shown. For dry runs and tests; it
    proves the plumbing, never anything about thinking."""

    def __init__(self, role: str, answer: Optional[Callable[[str, str], str]] = None):
        self.role, self.answer, self.calls = role, answer, []

    def complete(self, system: str, prompt: str, max_output: int) -> Reply:
        import re

        self.calls.append((system, prompt))
        if self.answer:
            return Reply(self.answer(system, prompt), Usage(len(prompt) // 4, 50))
        ids = list(dict.fromkeys(re.findall(r"\[([0-9a-f]{8}-[0-9a-f-]{27})\]", prompt)))
        first = ids[:2]
        shapes = {
            "observer": {"observations": [{"id": "O1", "kind": "novelty", "memory_ids": first,
                                           "note": "Dry run: the observer saw these memories."}], "skipped_ids": []},
            "investigator": {"hypotheses": [{"id": "H1", "statement": "Dry run hypothesis.", "evidence_ids": first,
                                             "falsifier": "A memory that says otherwise.",
                                             "check_query": "dry run check"}]},
            "baseline": {"hypotheses": [{"id": "H1", "statement": "Dry run baseline hypothesis.",
                                         "evidence_ids": first[:1], "falsifier": "-", "check_query": ""}]},
            "supervisor": {"audit": [], "verdicts": [{"hypothesis": "H1", "verdict": "untested", "ids": first,
                                                      "why": "Dry run."}],
                           "reflection": {"text": "Dry run: no model was called.", "sources": first,
                                          "scope": "project", "project": "INFINITO", "advice": "", "questions": []},
                           "proposals": [], "summary_for_owner": "Dry run."},
        }
        return Reply(json.dumps(shapes[self.role]), Usage(len(prompt) // 4, 50))
