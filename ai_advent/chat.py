from dataclasses import dataclass
from typing import Any, Protocol


class ChatCompletionsAPI(Protocol):
    def create(self, **kwargs: object) -> object: ...


@dataclass(frozen=True)
class ChatResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatSession:
    """A multi-turn chat session backed by the Chat Completions API."""

    def __init__(self, chat_completions_api: ChatCompletionsAPI, model: str) -> None:
        self._chat_completions_api = chat_completions_api
        self._model = model
        self._messages: list[dict[str, str]] = []

    def ask(
        self,
        message: str,
        *,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
    ) -> str:
        return self.ask_with_metadata(
            message,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            stop=stop,
            temperature=temperature,
        ).text

    def ask_with_metadata(
        self,
        message: str,
        *,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        temperature: float | None = None,
    ) -> ChatResponse:
        self._messages.append({"role": "user", "content": message})

        request: dict[str, object] = {
            "model": self._model,
            "messages": list(self._messages),
        }
        if max_completion_tokens is not None:
            request["max_completion_tokens"] = max_completion_tokens
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if stop is not None:
            request["stop"] = stop
        if temperature is not None:
            request["temperature"] = temperature

        response = self._chat_completions_api.create(**request)
        answer = _response_text(response)
        self._messages.append({"role": "assistant", "content": answer})
        return ChatResponse(answer, *_response_usage(response))


def _response_text(response: object) -> str:
    choices = getattr(response, "choices")
    if not choices:
        return ""

    message = getattr(choices[0], "message")
    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(_content_part_text(part) for part in content)

    return "" if content is None else str(content)


def _response_usage(response: object) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None, None, None

    return (
        _usage_value(usage, "prompt_tokens"),
        _usage_value(usage, "completion_tokens"),
        _usage_value(usage, "total_tokens"),
    )


def _usage_value(usage: object, key: str) -> int | None:
    if isinstance(usage, dict):
        value = usage.get(key)
    else:
        value = getattr(usage, key, None)

    if isinstance(value, int):
        return value
    return None


def _content_part_text(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("text", ""))

    return str(getattr(part, "text", ""))
