from dataclasses import dataclass

from ai_advent.chat import (
    ChatCompletionsAPI,
    Message,
    create_chat_completion,
)
from ai_advent.context import ContextState, ContextStrategy, FullContextStrategy
from ai_advent.memory import AgentMemory, AgentMemoryState
from ai_advent.task import TaskState, create_task_state, validate_task_transition
from ai_advent.tokens import TokenReport

DEFAULT_SYSTEM_PROMPT = (
    "Ты Study Coach Agent: персональный учебный ассистент. "
    "Объясняй кратко и по делу, помогай учиться через понятные шаги и практику. "
    "Учитывай профиль пользователя и явные слои памяти. "
    "Если данных не хватает, сделай разумное предположение и явно обозначь его."
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
        facts: dict[str, str] | None = None,
        working_memory: dict[str, str] | None = None,
        long_term_memory: dict[str, str] | None = None,
        user_profile: dict[str, str] | None = None,
        task_state: TaskState | None = None,
        invariants: dict[str, str] | None = None,
        branch: str | None = None,
    ) -> None:
        self._chat_completions_api = chat_completions_api
        self._model = model
        self._system_prompt = system_prompt
        self._memory = memory
        self._context_strategy = context_strategy or FullContextStrategy()
        if messages is None:
            memory_state = _load_memory(memory)
            memory_state = _with_memory_layer_overrides(
                memory_state,
                working_memory=working_memory,
                long_term_memory=long_term_memory,
                user_profile=user_profile,
                task_state=task_state,
                invariants=invariants,
            )
        else:
            memory_state = AgentMemoryState(
                messages=messages,
                summary=summary,
                facts=facts or {},
                working_memory=working_memory or {},
                long_term_memory=long_term_memory or {},
                user_profile=user_profile or {},
                task_state=task_state or create_task_state(),
                invariants=invariants or {},
                current_branch=branch or "main",
            )
        self._branches = _copy_message_map(memory_state.branches or {})
        self._checkpoints = _copy_message_map(memory_state.checkpoints or {})
        self._current_branch = branch or memory_state.current_branch
        if self._current_branch not in self._branches:
            self._branches[self._current_branch] = _copy_messages(
                memory_state.messages
            )
        self._messages = _copy_messages(self._branches[self._current_branch])
        self._summary = memory_state.summary
        self._facts = dict(memory_state.facts or {})
        self._working_memory = dict(memory_state.working_memory or {})
        self._long_term_memory = dict(memory_state.long_term_memory or {})
        self._user_profile = dict(memory_state.user_profile or {})
        self._task_state = memory_state.task_state or create_task_state()
        self._invariants = dict(memory_state.invariants or {})

    @property
    def messages(self) -> list[Message]:
        return _copy_messages(self._messages)

    @property
    def summary(self) -> str:
        return self._summary

    @property
    def facts(self) -> dict[str, str]:
        return dict(self._facts)

    @property
    def working_memory(self) -> dict[str, str]:
        return dict(self._working_memory)

    @property
    def long_term_memory(self) -> dict[str, str]:
        return dict(self._long_term_memory)

    @property
    def user_profile(self) -> dict[str, str]:
        return dict(self._user_profile)

    @property
    def task_state(self) -> TaskState:
        return self._task_state

    @property
    def invariants(self) -> dict[str, str]:
        return dict(self._invariants)

    @property
    def current_branch(self) -> str:
        return self._current_branch

    def remember_working(self, key: str, value: str) -> None:
        self._working_memory[_validate_memory_key(key)] = value
        self._save_messages()

    def remember_long_term(self, key: str, value: str) -> None:
        self._long_term_memory[_validate_memory_key(key)] = value
        self._save_messages()

    def forget_working(self, key: str) -> None:
        self._working_memory.pop(key, None)
        self._save_messages()

    def forget_long_term(self, key: str) -> None:
        self._long_term_memory.pop(key, None)
        self._save_messages()

    def remember_profile(self, key: str, value: str) -> None:
        self._user_profile[_validate_memory_key(key)] = value
        self._save_messages()

    def forget_profile(self, key: str) -> None:
        self._user_profile.pop(key, None)
        self._save_messages()

    def start_task(self, title: str) -> None:
        validate_task_transition(self._task_state.stage, "planning")
        self._task_state = create_task_state(
            stage="planning",
            title=title,
            current_step="Сформировать план учебной задачи.",
            expected_action="Подготовить или уточнить план.",
        )
        self._save_messages()

    def update_task_state(
        self,
        *,
        stage: str | None = None,
        current_step: str | None = None,
        expected_action: str | None = None,
        title: str | None = None,
        paused: bool | None = None,
    ) -> None:
        next_stage = self._task_state.stage if stage is None else stage
        validate_task_transition(self._task_state.stage, next_stage)
        self._task_state = create_task_state(
            stage=next_stage,
            title=self._task_state.title if title is None else title,
            current_step=(
                self._task_state.current_step
                if current_step is None
                else current_step
            ),
            expected_action=(
                self._task_state.expected_action
                if expected_action is None
                else expected_action
            ),
            paused=self._task_state.paused if paused is None else paused,
        )
        self._save_messages()

    def approve_task(self) -> None:
        validate_task_transition(
            self._task_state.stage,
            "execution",
            approved=True,
        )
        self._task_state = create_task_state(
            stage="execution",
            title=self._task_state.title,
            current_step="Выполнить утвержденный план.",
            expected_action="Продолжить выполнение задачи.",
            paused=False,
        )
        self._save_messages()

    def pause_task(self) -> None:
        self.update_task_state(paused=True)

    def resume_task(self) -> None:
        self.update_task_state(paused=False)

    def remember_invariant(self, key: str, value: str) -> None:
        self._invariants[_validate_memory_key(key)] = value
        self._save_messages()

    def forget_invariant(self, key: str) -> None:
        self._invariants.pop(key, None)
        self._save_messages()

    def run_turn(
        self,
        user_message: str,
        *,
        max_completion_tokens: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AgentResponse:
        self._messages.append({"role": "user", "content": user_message})
        invariant_conflict = self._find_invariant_conflict(user_message)
        if invariant_conflict is not None:
            response_text = _format_invariant_refusal(invariant_conflict)
            self._messages.append({"role": "assistant", "content": response_text})
            self._save_messages()
            token_report = TokenReport(
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
            )
            return AgentResponse(text=response_text, token_report=token_report)

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
                facts=self._facts,
            )
        )
        memory_messages = self._build_memory_messages()
        profile_messages = self._build_profile_messages()
        task_messages = self._build_task_messages()
        invariant_messages = self._build_invariant_messages()
        if not self._system_prompt:
            return [
                *profile_messages,
                *task_messages,
                *invariant_messages,
                *memory_messages,
                *context_messages,
            ]

        return [
            {"role": "system", "content": self._system_prompt},
            *profile_messages,
            *task_messages,
            *invariant_messages,
            *memory_messages,
            *context_messages,
        ]

    def _build_invariant_messages(self) -> list[Message]:
        if not self._invariants:
            return []

        return [
            {
                "role": "system",
                "content": (
                    "Инварианты Study Coach Agent. "
                    "Считай эти правила жесткими ограничениями. "
                    "Перед ответом проверь, не нарушает ли предложение один из инвариантов. "
                    "Если запрос конфликтует с инвариантом, откажись от конфликтующей части и кратко объясни, какой инвариант мешает:\n\n"
                    + _format_memory_map(self._invariants)
                ),
            }
        ]

    def _build_task_messages(self) -> list[Message]:
        if self._task_state.stage == "idle" and not self._task_state.title:
            return []

        return [
            {
                "role": "system",
                "content": (
                    "Формализованное состояние текущей учебной задачи. "
                    "Продолжай работу с этого состояния без повторного объяснения уже известных вводных:\n\n"
                    f"- stage: {self._task_state.stage}\n"
                    f"- title: {self._task_state.title or '(not set)'}\n"
                    f"- current_step: {self._task_state.current_step or '(not set)'}\n"
                    f"- expected_action: {self._task_state.expected_action or '(not set)'}\n"
                    f"- paused: {self._task_state.paused}"
                ),
            }
        ]

    def _build_profile_messages(self) -> list[Message]:
        if not self._user_profile:
            return []

        return [
            {
                "role": "system",
                "content": (
                    "Профиль пользователя Study Coach Agent. "
                    "Автоматически адаптируй стиль, язык, формат и ограничения ответа под этот профиль:\n\n"
                    + _format_memory_map(self._user_profile)
                ),
            }
        ]

    def _build_memory_messages(self) -> list[Message]:
        sections = []
        if self._working_memory:
            sections.append(
                "Рабочая память текущей учебной задачи:\n"
                + _format_memory_map(self._working_memory)
            )
        if self._long_term_memory:
            sections.append(
                "Долговременная память пользователя и ассистента:\n"
                + _format_memory_map(self._long_term_memory)
            )
        if not sections:
            return []

        return [
            {
                "role": "system",
                "content": (
                    "Явные слои памяти Study Coach Agent. "
                    "Учитывай эти данные в ответе, но не смешивай их с текущим "
                    "диалогом:\n\n"
                    + "\n\n".join(sections)
                ),
            }
        ]

    def _compress_context(self) -> None:
        result = self._context_strategy.compress(
            ContextState(
                messages=self._messages,
                summary=self._summary,
                facts=self._facts,
            ),
            self._chat_completions_api,
            self._model,
        )
        if result.changed:
            self._messages = _copy_messages(result.state.messages)
            self._summary = result.state.summary
            self._facts = dict(result.state.facts or {})
        self._branches[self._current_branch] = _copy_messages(self._messages)

    def _save_messages(self) -> None:
        if self._memory is not None:
            self._memory.save_state(
                AgentMemoryState(
                    messages=self._messages,
                    summary=self._summary,
                    facts=self._facts,
                    working_memory=self._working_memory,
                    long_term_memory=self._long_term_memory,
                    user_profile=self._user_profile,
                    task_state=self._task_state,
                    invariants=self._invariants,
                    branches=self._branches,
                    checkpoints=self._checkpoints,
                    current_branch=self._current_branch,
                )
            )

    def save_checkpoint(self, name: str) -> None:
        _validate_name(name, "checkpoint")
        self._checkpoints[name] = _copy_messages(self._messages)
        self._save_messages()

    def create_branch(self, name: str, checkpoint: str | None = None) -> None:
        _validate_name(name, "branch")
        if name in self._branches:
            raise ValueError(f"Branch already exists: {name}")
        if checkpoint is None:
            messages = self._messages
        else:
            if checkpoint not in self._checkpoints:
                raise ValueError(f"Unknown checkpoint: {checkpoint}")
            messages = self._checkpoints[checkpoint]

        self._branches[name] = _copy_messages(messages)
        self.switch_branch(name)

    def switch_branch(self, name: str) -> None:
        if name not in self._branches:
            raise ValueError(f"Unknown branch: {name}")
        self._branches[self._current_branch] = _copy_messages(self._messages)
        self._current_branch = name
        self._messages = _copy_messages(self._branches[name])
        self._save_messages()

    def list_branches(self) -> list[str]:
        return sorted(self._branches)

    def list_checkpoints(self) -> list[str]:
        return sorted(self._checkpoints)

    def _find_invariant_conflict(self, user_message: str) -> tuple[str, str] | None:
        lowered_message = user_message.lower()
        conflict_words = (
            "наруш",
            "игнор",
            "обойди",
            "ignore",
            "violate",
            "break",
            "bypass",
        )
        if not any(word in lowered_message for word in conflict_words):
            return None

        for key, value in sorted(self._invariants.items()):
            if key.lower() in lowered_message:
                return key, value

        return None


def _copy_messages(messages: list[Message]) -> list[Message]:
    return [message.copy() for message in messages]


def _copy_message_map(value: dict[str, list[Message]]) -> dict[str, list[Message]]:
    return {
        key: _copy_messages(messages)
        for key, messages in value.items()
    }


def _validate_name(name: str, label: str) -> None:
    if not name or any(character.isspace() for character in name):
        raise ValueError(f"{label} name must be non-empty and contain no spaces.")


def _with_memory_layer_overrides(
    state: AgentMemoryState,
    *,
    working_memory: dict[str, str] | None,
    long_term_memory: dict[str, str] | None,
    user_profile: dict[str, str] | None,
    task_state: TaskState | None,
    invariants: dict[str, str] | None,
) -> AgentMemoryState:
    if (
        working_memory is None
        and long_term_memory is None
        and user_profile is None
        and task_state is None
        and invariants is None
    ):
        return state

    return AgentMemoryState(
        messages=state.messages,
        summary=state.summary,
        facts=state.facts,
        working_memory=(
            state.working_memory if working_memory is None else working_memory
        ),
        long_term_memory=(
            state.long_term_memory if long_term_memory is None else long_term_memory
        ),
        user_profile=state.user_profile if user_profile is None else user_profile,
        task_state=state.task_state if task_state is None else task_state,
        invariants=state.invariants if invariants is None else invariants,
        branches=state.branches,
        checkpoints=state.checkpoints,
        current_branch=state.current_branch,
    )


def _validate_memory_key(key: str) -> str:
    if not key or any(character.isspace() for character in key):
        raise ValueError("Memory key must be non-empty and contain no spaces.")
    return key


def _format_memory_map(memory: dict[str, str]) -> str:
    return "\n".join(f"- {key}: {value}" for key, value in sorted(memory.items()))


def _format_invariant_refusal(invariant: tuple[str, str]) -> str:
    key, value = invariant
    return (
        f"Не могу выполнить эту часть запроса: она нарушает инвариант `{key}`. "
        f"Ограничение: {value}"
    )


def _load_memory(memory: AgentMemory | None) -> AgentMemoryState:
    if memory is None:
        return AgentMemoryState(messages=[])
    return memory.load_state()
