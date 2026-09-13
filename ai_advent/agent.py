from dataclasses import dataclass

from ai_advent.chat import (
    ChatCompletionsAPI,
    Message,
    create_chat_completion,
)
from ai_advent.memory import AgentMemory
from ai_advent.tokens import TokenReport

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
    token_report: TokenReport | None = None


class Agent:
    """A first AI Advent agent with encapsulated request/response logic."""

    def __init__(
        self,
        chat_completions_api: ChatCompletionsAPI,
        model: str,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        memory: AgentMemory | None = None,
        messages: list[Message] | None = None,
    ) -> None:
        self._chat_completions_api = chat_completions_api
        self._model = model
        self._system_prompt = system_prompt
        self._memory = memory
        initial_messages = messages if messages is not None else _load_memory(memory)
        self._messages = _copy_messages(initial_messages)

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

        try:
            response = create_chat_completion(
                self._chat_completions_api,
                self._model,
                request_messages,
                max_completion_tokens=max_completion_tokens,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            self._messages.pop()
            raise
        self._messages.append({"role": "assistant", "content": response.text})
        self._save_messages()
        token_report = TokenReport(
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )
        return AgentResponse(
            text=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
            token_report=token_report,
        )

    def _build_request_messages(self) -> list[Message]:
        if not self._system_prompt:
            return _copy_messages(self._messages)

        return [
            {"role": "system", "content": self._system_prompt},
            *_copy_messages(self._messages),
        ]

    def _save_messages(self) -> None:
        if self._memory is not None:
            self._memory.save_messages(self._messages)


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.copy() for message in messages]


def _load_memory(memory: AgentMemory | None) -> list[Message]:
    if memory is None:
        return []
    return memory.load_messages()
