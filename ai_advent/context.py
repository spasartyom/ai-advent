from dataclasses import dataclass
from typing import Protocol

from ai_advent.chat import ChatCompletionsAPI, Message, create_chat_completion

SUMMARY_SYSTEM_PROMPT = (
    "Ты сжимаешь историю диалога для AI-агента. "
    "Сохраняй цели, ограничения, предпочтения, решения, факты, имена, команды "
    "и договоренности. Пиши кратко, структурно и без Markdown-таблиц."
)


@dataclass(frozen=True)
class ContextState:
    messages: list[Message]
    summary: str = ""


@dataclass(frozen=True)
class CompressionResult:
    state: ContextState
    changed: bool


class ContextStrategy(Protocol):
    def build_messages(self, state: ContextState) -> list[Message]: ...

    def compress(
        self,
        state: ContextState,
        chat_completions_api: ChatCompletionsAPI,
        model: str,
    ) -> CompressionResult: ...


class FullContextStrategy:
    def build_messages(self, state: ContextState) -> list[Message]:
        return _copy_messages(state.messages)

    def compress(
        self,
        state: ContextState,
        chat_completions_api: ChatCompletionsAPI,
        model: str,
    ) -> CompressionResult:
        return CompressionResult(state=state, changed=False)


class SummaryContextStrategy:
    def __init__(self, keep_last: int) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1.")
        self._keep_last = keep_last

    def build_messages(self, state: ContextState) -> list[Message]:
        messages = []
        if state.summary:
            messages.append(
                {
                    "role": "system",
                    "content": (
                        "Краткое summary предыдущей части диалога:\n"
                        f"{state.summary}"
                    ),
                }
            )
        messages.extend(_copy_messages(state.messages))
        return messages

    def compress(
        self,
        state: ContextState,
        chat_completions_api: ChatCompletionsAPI,
        model: str,
    ) -> CompressionResult:
        if len(state.messages) <= self._keep_last:
            return CompressionResult(state=state, changed=False)

        messages_to_summarize = state.messages[:-self._keep_last]
        recent_messages = state.messages[-self._keep_last :]
        summary = summarize_messages(
            chat_completions_api,
            model,
            state.summary,
            messages_to_summarize,
        )
        return CompressionResult(
            state=ContextState(
                messages=_copy_messages(recent_messages),
                summary=summary,
            ),
            changed=True,
        )


def summarize_messages(
    chat_completions_api: ChatCompletionsAPI,
    model: str,
    existing_summary: str,
    messages: list[Message],
) -> str:
    transcript = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    prompt = (
        "Обнови summary диалога.\n\n"
        f"Текущее summary:\n{existing_summary or '(пока пусто)'}\n\n"
        f"Новая часть диалога для сжатия:\n{transcript}\n\n"
        "Верни только обновленное summary."
    )
    response = create_chat_completion(
        chat_completions_api,
        model,
        [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=700,
    )
    return response.text.strip()


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.copy() for message in messages]
