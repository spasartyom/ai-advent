import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ai_advent.chat import Message
from ai_advent.task import TaskState, create_task_state, task_state_from_dict


@dataclass(frozen=True)
class AgentMemoryState:
    messages: list[Message]
    summary: str = ""
    facts: dict[str, str] | None = None
    working_memory: dict[str, str] | None = None
    long_term_memory: dict[str, str] | None = None
    user_profile: dict[str, str] | None = None
    task_state: TaskState | None = None
    invariants: dict[str, str] | None = None
    branches: dict[str, list[Message]] | None = None
    checkpoints: dict[str, list[Message]] | None = None
    current_branch: str = "main"


class AgentMemory(Protocol):
    def load_state(self) -> AgentMemoryState: ...

    def save_state(self, state: AgentMemoryState) -> None: ...


class JsonFileMemory:
    """Persistent agent memory backed by a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load_state(self) -> AgentMemoryState:
        if not self._path.exists():
            return AgentMemoryState(messages=[])

        with self._path.open("r", encoding="utf-8") as memory_file:
            data = json.load(memory_file)

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("Memory file must contain a messages list.")

        summary = data.get("summary", "")
        if not isinstance(summary, str):
            raise ValueError("Memory file summary must be a string.")

        facts = data.get("facts", {})
        if not isinstance(facts, dict):
            raise ValueError("Memory file facts must be an object.")

        working_memory = data.get("working_memory", {})
        if not isinstance(working_memory, dict):
            raise ValueError("Memory file working_memory must be an object.")

        long_term_memory = data.get("long_term_memory", {})
        if not isinstance(long_term_memory, dict):
            raise ValueError("Memory file long_term_memory must be an object.")

        user_profile = data.get("user_profile", {})
        if not isinstance(user_profile, dict):
            raise ValueError("Memory file user_profile must be an object.")

        task_state = data.get("task_state", {})
        if not isinstance(task_state, dict):
            raise ValueError("Memory file task_state must be an object.")

        invariants = data.get("invariants", {})
        if not isinstance(invariants, dict):
            raise ValueError("Memory file invariants must be an object.")

        branches = data.get("branches", {})
        if not isinstance(branches, dict):
            raise ValueError("Memory file branches must be an object.")

        checkpoints = data.get("checkpoints", {})
        if not isinstance(checkpoints, dict):
            raise ValueError("Memory file checkpoints must be an object.")

        current_branch = data.get("current_branch", "main")
        if not isinstance(current_branch, str):
            raise ValueError("Memory file current_branch must be a string.")

        return AgentMemoryState(
            messages=[_validate_message(message) for message in messages],
            summary=summary,
            facts=_validate_facts(facts),
            working_memory=_validate_string_map(working_memory, "working_memory"),
            long_term_memory=_validate_string_map(long_term_memory, "long_term_memory"),
            user_profile=_validate_string_map(user_profile, "user_profile"),
            task_state=task_state_from_dict(task_state),
            invariants=_validate_string_map(invariants, "invariants"),
            branches=_validate_message_map(branches, "branch"),
            checkpoints=_validate_message_map(checkpoints, "checkpoint"),
            current_branch=current_branch,
        )

    def save_state(self, state: AgentMemoryState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": state.summary,
            "facts": dict(state.facts or {}),
            "working_memory": dict(state.working_memory or {}),
            "long_term_memory": dict(state.long_term_memory or {}),
            "user_profile": dict(state.user_profile or {}),
            "task_state": (state.task_state or create_task_state()).to_dict(),
            "invariants": dict(state.invariants or {}),
            "messages": [message.copy() for message in state.messages],
            "branches": _copy_message_map(state.branches or {}),
            "checkpoints": _copy_message_map(state.checkpoints or {}),
            "current_branch": state.current_branch,
        }

        with self._path.open("w", encoding="utf-8") as memory_file:
            json.dump(payload, memory_file, ensure_ascii=False, indent=2)
            memory_file.write("\n")

    def load_messages(self) -> list[Message]:
        return self.load_state().messages

    def save_messages(self, messages: list[Message]) -> None:
        self.save_state(AgentMemoryState(messages=messages))


def _validate_message(message: object) -> Message:
    if not isinstance(message, dict):
        raise ValueError("Each memory message must be an object.")

    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError("Each memory message must contain role and content strings.")

    return {"role": role, "content": content}


def _validate_facts(facts: dict[object, object]) -> dict[str, str]:
    return _validate_string_map(facts, "facts")


def _validate_string_map(
    value: dict[object, object],
    label: str,
) -> dict[str, str]:
    validated: dict[str, str] = {}
    for key, item in value.items():
        if not isinstance(key, str) or not isinstance(item, str):
            raise ValueError(
                f"Memory file {label} must contain string keys and values."
            )
        validated[key] = item
    return validated


def _validate_message_map(
    value: dict[object, object],
    label: str,
) -> dict[str, list[Message]]:
    validated: dict[str, list[Message]] = {}
    for key, messages in value.items():
        if not isinstance(key, str) or not isinstance(messages, list):
            raise ValueError(f"Each memory {label} must contain a messages list.")
        validated[key] = [_validate_message(message) for message in messages]
    return validated


def _copy_message_map(
    value: dict[str, list[Message]],
) -> dict[str, list[Message]]:
    return {
        key: [message.copy() for message in messages]
        for key, messages in value.items()
    }
