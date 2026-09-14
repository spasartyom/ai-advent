import json
from dataclasses import dataclass
from typing import Protocol

from ai_advent.chat import ChatCompletionsAPI, Message, create_chat_completion

SUMMARY_SYSTEM_PROMPT = (
    "Ты сжимаешь историю диалога для AI-агента. "
    "Сохраняй цели, ограничения, предпочтения, решения, факты, имена, команды "
    "и договоренности. Пиши кратко, структурно и без Markdown-таблиц."
)
FACTS_SYSTEM_PROMPT = (
    "Ты обновляешь key-value memory AI-агента. "
    "Сохраняй только важные устойчивые факты: цель, ограничения, предпочтения, "
    "решения, договоренности, имена, команды и критичные параметры. "
    "Верни только JSON object со строковыми ключами и строковыми значениями."
)


@dataclass(frozen=True)
class ContextState:
    messages: list[Message]
    summary: str = ""
    facts: dict[str, str] | None = None


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


class SlidingWindowContextStrategy:
    def __init__(self, keep_last: int) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1.")
        self._keep_last = keep_last

    def build_messages(self, state: ContextState) -> list[Message]:
        return _copy_messages(state.messages[-self._keep_last :])

    def compress(
        self,
        state: ContextState,
        chat_completions_api: ChatCompletionsAPI,
        model: str,
    ) -> CompressionResult:
        if len(state.messages) <= self._keep_last:
            return CompressionResult(state=state, changed=False)

        return CompressionResult(
            state=ContextState(
                messages=_copy_messages(state.messages[-self._keep_last :]),
                summary=state.summary,
                facts=dict(state.facts or {}),
            ),
            changed=True,
        )


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
                facts=dict(state.facts or {}),
            ),
            changed=True,
        )


class StickyFactsContextStrategy:
    def __init__(self, keep_last: int) -> None:
        if keep_last < 1:
            raise ValueError("keep_last must be at least 1.")
        self._keep_last = keep_last

    def build_messages(self, state: ContextState) -> list[Message]:
        messages = []
        facts = state.facts or {}
        if facts:
            messages.append(
                {
                    "role": "system",
                    "content": "Sticky facts:\n" + _format_facts(facts),
                }
            )
        messages.extend(_copy_messages(state.messages[-self._keep_last :]))
        return messages

    def compress(
        self,
        state: ContextState,
        chat_completions_api: ChatCompletionsAPI,
        model: str,
    ) -> CompressionResult:
        facts = update_facts(
            chat_completions_api,
            model,
            state.facts or {},
            state.messages,
        )
        recent_messages = state.messages[-self._keep_last :]
        changed = facts != (state.facts or {}) or len(state.messages) > self._keep_last
        return CompressionResult(
            state=ContextState(
                messages=_copy_messages(recent_messages),
                summary=state.summary,
                facts=facts,
            ),
            changed=changed,
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


def update_facts(
    chat_completions_api: ChatCompletionsAPI,
    model: str,
    existing_facts: dict[str, str],
    messages: list[Message],
) -> dict[str, str]:
    transcript = "\n".join(
        f"{message['role']}: {message['content']}" for message in messages
    )
    prompt = (
        "Обнови facts по диалогу.\n\n"
        f"Текущие facts JSON:\n{json.dumps(existing_facts, ensure_ascii=False)}\n\n"
        f"Диалог:\n{transcript}\n\n"
        "Верни только JSON object. Удали устаревшие facts, если они явно "
        "заменены в диалоге."
    )
    response = create_chat_completion(
        chat_completions_api,
        model,
        [
            {"role": "system", "content": FACTS_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=700,
    )
    return _parse_facts(response.text, existing_facts)


def _parse_facts(text: str, fallback: dict[str, str]) -> dict[str, str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return dict(fallback)

    if not isinstance(data, dict):
        return dict(fallback)

    facts: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            facts[key] = value
    return facts


def _format_facts(facts: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in sorted(facts.items()))


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.copy() for message in messages]
