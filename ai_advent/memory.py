import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ai_advent.chat import Message


@dataclass(frozen=True)
class AgentMemoryState:
    messages: list[Message]
    summary: str = ""


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

        return AgentMemoryState(
            messages=[_validate_message(message) for message in messages],
            summary=summary,
        )

    def save_state(self, state: AgentMemoryState) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": state.summary,
            "messages": [message.copy() for message in state.messages],
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
