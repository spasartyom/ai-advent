from dataclasses import dataclass

from ai_advent.chat import (
    ChatCompletionsAPI,
    Message,
    create_chat_completion,
)

DEFAULT_SYSTEM_PROMPT = (
    "Ты полезный AI-агент. Отвечай кратко и по делу. "
    "Не задавай уточняющих вопросов; если данных не хватает, сделай разумное "
    "предположение и явно обозначь его."
)


@dataclass(frozen=True)
class AgentResponse:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class Agent:
    """A first AI Advent agent with encapsulated request/response logic."""

    def __init__(
        self,
        chat_completions_api: ChatCompletionsAPI,
        model: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        messages: list[Message] | None = None,
    ) -> None:
        self._chat_completions_api = chat_completions_api
        self._model = model
        self._system_prompt = system_prompt
        self._messages = _copy_messages(messages or [])

    @property
    def messages(self) -> list[Message]:
        return _copy_messages(self._messages)

    def run_turn(
        self,
        user_message: str,
        *,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        self._messages.append({"role": "user", "content": user_message})
        request_messages = self._build_request_messages()
        response = create_chat_completion(
            self._chat_completions_api,
            self._model,
            request_messages,
            max_completion_tokens=max_completion_tokens,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        self._messages.append({"role": "assistant", "content": response.text})
        return AgentResponse(
            text=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

    def _build_request_messages(self) -> list[Message]:
        if not self._system_prompt:
            return _copy_messages(self._messages)

        return [
            {"role": "system", "content": self._system_prompt},
            *_copy_messages(self._messages),
        ]


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.copy() for message in messages]
