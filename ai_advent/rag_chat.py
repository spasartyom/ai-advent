import json
from dataclasses import asdict, dataclass
from pathlib import Path

from ai_advent.chat import Message
from ai_advent.rag import GroundedRagAnswer, RagResponder


@dataclass(frozen=True)
class RagTaskMemory:
    goal: str = ""
    clarifications: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    terms: dict[str, str] | None = None

    def with_goal(self, goal: str) -> "RagTaskMemory":
        return RagTaskMemory(
            goal=goal,
            clarifications=self.clarifications,
            constraints=self.constraints,
            terms=dict(self.terms or {}),
        )

    def add_clarification(self, clarification: str) -> "RagTaskMemory":
        return RagTaskMemory(
            goal=self.goal,
            clarifications=(*self.clarifications, clarification),
            constraints=self.constraints,
            terms=dict(self.terms or {}),
        )

    def add_constraint(self, constraint: str) -> "RagTaskMemory":
        return RagTaskMemory(
            goal=self.goal,
            clarifications=self.clarifications,
            constraints=(*self.constraints, constraint),
            terms=dict(self.terms or {}),
        )

    def set_term(self, key: str, value: str) -> "RagTaskMemory":
        terms = dict(self.terms or {})
        terms[key] = value
        return RagTaskMemory(
            goal=self.goal,
            clarifications=self.clarifications,
            constraints=self.constraints,
            terms=terms,
        )


@dataclass(frozen=True)
class RagChatState:
    messages: tuple[Message, ...] = ()
    task_memory: RagTaskMemory = RagTaskMemory()


@dataclass(frozen=True)
class RagChatScenario:
    name: str
    messages: tuple[str, ...]
    expected: str


class JsonRagChatMemory:
    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def load(self) -> RagChatState:
        if not self._path.exists():
            return RagChatState()
        payload = json.loads(self._path.read_text(encoding="utf-8"))
        raw_messages = payload.get("messages", [])
        if not isinstance(raw_messages, list):
            raise ValueError("RAG chat memory messages must be a list.")

        messages: list[Message] = []
        for item in raw_messages:
            if not isinstance(item, dict):
                raise ValueError("RAG chat memory message must be an object.")
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError("RAG chat memory messages need role and content.")
            messages.append({"role": role, "content": content})

        task_payload = payload.get("task_memory", {})
        if not isinstance(task_payload, dict):
            task_payload = {}
        raw_terms = task_payload.get("terms", {})
        if not isinstance(raw_terms, dict):
            raw_terms = {}
        return RagChatState(
            messages=tuple(messages),
            task_memory=RagTaskMemory(
                goal=str(task_payload.get("goal", "")),
                clarifications=tuple(
                    str(item) for item in task_payload.get("clarifications", [])
                ),
                constraints=tuple(
                    str(item) for item in task_payload.get("constraints", [])
                ),
                terms={
                    str(key): str(value)
                    for key, value in raw_terms.items()
                },
            ),
        )

    def save(self, state: RagChatState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "messages": list(state.messages),
            "task_memory": asdict(state.task_memory),
        }
        self._path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


class RagChatSession:
    def __init__(
        self,
        responder: RagResponder,
        *,
        memory: JsonRagChatMemory | None = None,
        state: RagChatState | None = None,
        keep_last: int = 12,
        candidate_k: int | None = None,
        min_score: float = 0.2,
        rewrite_query: bool = True,
    ) -> None:
        if keep_last <= 0:
            raise ValueError("keep_last must be greater than zero.")
        self._responder = responder
        self._memory = memory
        self._state = state or (memory.load() if memory is not None else RagChatState())
        self._keep_last = keep_last
        self._candidate_k = candidate_k
        self._min_score = min_score
        self._rewrite_query = rewrite_query

    @property
    def state(self) -> RagChatState:
        return self._state

    def set_goal(self, goal: str) -> None:
        self._state = RagChatState(
            messages=self._state.messages,
            task_memory=self._state.task_memory.with_goal(goal),
        )
        self._save()

    def add_clarification(self, clarification: str) -> None:
        self._state = RagChatState(
            messages=self._state.messages,
            task_memory=self._state.task_memory.add_clarification(clarification),
        )
        self._save()

    def add_constraint(self, constraint: str) -> None:
        self._state = RagChatState(
            messages=self._state.messages,
            task_memory=self._state.task_memory.add_constraint(constraint),
        )
        self._save()

    def set_term(self, key: str, value: str) -> None:
        self._state = RagChatState(
            messages=self._state.messages,
            task_memory=self._state.task_memory.set_term(key, value),
        )
        self._save()

    def ask(self, question: str) -> GroundedRagAnswer:
        answer = self._responder.answer_with_citations(
            question,
            candidate_k=self._candidate_k,
            min_score=self._min_score,
            rewrite_query=self._rewrite_query,
            conversation_context=format_rag_chat_context(
                self._state,
                keep_last=self._keep_last,
            ),
        )
        self._state = RagChatState(
            messages=(
                *self._state.messages,
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer.answer},
            ),
            task_memory=self._state.task_memory,
        )
        self._save()
        return answer

    def _save(self) -> None:
        if self._memory is not None:
            self._memory.save(self._state)


def format_rag_chat_context(state: RagChatState, *, keep_last: int = 12) -> str:
    terms = state.task_memory.terms or {}
    lines = ["Task memory:"]
    lines.append(f"goal: {state.task_memory.goal or '(not set)'}")
    lines.append(
        "clarifications: "
        + ("; ".join(state.task_memory.clarifications) or "(empty)")
    )
    lines.append(
        "constraints: "
        + ("; ".join(state.task_memory.constraints) or "(empty)")
    )
    lines.append(
        "terms: "
        + (
            "; ".join(f"{key}={value}" for key, value in sorted(terms.items()))
            or "(empty)"
        )
    )
    lines.append("Recent dialog:")
    recent_messages = state.messages[-keep_last:]
    if not recent_messages:
        lines.append("(empty)")
    for message in recent_messages:
        lines.append(f"{message['role']}: {message['content']}")
    return "\n".join(lines)


RAG_CHAT_SCENARIOS = [
    RagChatScenario(
        name="task-state-study-session",
        messages=(
            "/goal разобраться с task state CLI",
            "Какие стадии задачи есть?",
            "/constraint отвечай коротко и со ссылками",
            "Как перейти из planning в execution?",
            "/clarify пользователь хочет увидеть запрет прямого перехода",
            "Почему /task stage execution не подходит?",
            "Что делать после execution?",
            "Как завершить задачу корректно?",
            "Повтори цель и ограничения этого диалога.",
            "Дай финальную шпаргалку по task state.",
        ),
        expected="Assistant should preserve the task-state goal and keep citing sources.",
    ),
    RagChatScenario(
        name="rag-indexing-study-session",
        messages=(
            "/goal подготовить демо Week 5 RAG",
            "Как собрать индекс?",
            "/term structure structure-aware chunking",
            "Чем fixed отличается от structure?",
            "/constraint всегда показывай chunk_id",
            "Как запустить первый RAG-запрос?",
            "Что добавил День 23?",
            "Что делает grounded режим Дня 24?",
            "Что значит слабая релевантность?",
            "Собери итоговый demo flow.",
        ),
        expected="Assistant should preserve the RAG demo goal and answer with sources each turn.",
    ),
]


def format_rag_chat_scenarios() -> list[str]:
    lines = ["RAG chat scenarios:"]
    for scenario in RAG_CHAT_SCENARIOS:
        lines.append(f"- {scenario.name}")
        lines.append(f"  expected: {scenario.expected}")
        for index, message in enumerate(scenario.messages, start=1):
            lines.append(f"  {index}. {message}")
    return lines
