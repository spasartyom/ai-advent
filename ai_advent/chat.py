from typing import Any, Protocol


class ChatCompletionsAPI(Protocol):
    def create(self, **kwargs: object) -> object: ...


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
        stop: list[str] | None = None,
        temperature: float | None = None,
    ) -> str:
        self._messages.append({"role": "user", "content": message})

        request: dict[str, object] = {
            "model": self._model,
            "messages": list(self._messages),
        }
        if max_completion_tokens is not None:
            request["max_completion_tokens"] = max_completion_tokens
        if stop is not None:
            request["stop"] = stop
        if temperature is not None:
            request["temperature"] = temperature

        response = self._chat_completions_api.create(**request)
        answer = _response_text(response)
        self._messages.append({"role": "assistant", "content": answer})
        return answer


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


def _content_part_text(part: Any) -> str:
    if isinstance(part, dict):
        return str(part.get("text", ""))

    return str(getattr(part, "text", ""))
