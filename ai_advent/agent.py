from dataclasses import dataclass

from ai_advent.chat import (
    ChatCompletionsAPI,
    Message,
    create_chat_completion,
)
from ai_advent.context import ContextState, ContextStrategy, FullContextStrategy
from ai_advent.memory import AgentMemory, AgentMemoryState
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
        context_strategy: ContextStrategy | None = None,
        messages: list[Message] | None = None,
        summary: str = "",
    ) -> None:
        self._chat_completions_api = chat_completions_api
        self._model = model
        self._system_prompt = system_prompt
        self._memory = memory
        self._context_strategy = context_strategy or FullContextStrategy()
        if messages is None:
            memory_state = _load_memory(memory)
        else:
            memory_state = AgentMemoryState(
                messages=messages,
                summary=summary,
            )
        self._messages = _copy_messages(memory_state.messages)
        self._summary = memory_state.summary

    @property
    def messages(self) -> list[Message]:
        return _copy_messages(self._messages)

    @property
    def summary(self) -> str:
        return self._summary

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
        self._compress_context()
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
        context_messages = self._context_strategy.build_messages(
            ContextState(
                messages=self._messages,
                summary=self._summary,
            )
        )
        if not self._system_prompt:
            return context_messages

        return [
            {"role": "system", "content": self._system_prompt},
            *context_messages,
        ]

    def _compress_context(self) -> None:
        result = self._context_strategy.compress(
            ContextState(
                messages=self._messages,
                summary=self._summary,
            ),
            self._chat_completions_api,
            self._model,
        )
        if result.changed:
            self._messages = _copy_messages(result.state.messages)
            self._summary = result.state.summary

    def _save_messages(self) -> None:
        if self._memory is not None:
            self._memory.save_state(
                AgentMemoryState(
                    messages=self._messages,
                    summary=self._summary,
                )
            )


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.copy() for message in messages]


def _load_memory(memory: AgentMemory | None) -> AgentMemoryState:
    if memory is None:
        return AgentMemoryState(messages=[])
    return memory.load_state()
