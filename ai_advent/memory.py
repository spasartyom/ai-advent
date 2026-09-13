import json
from pathlib import Path
from typing import Protocol

from ai_advent.chat import Message


class AgentMemory(Protocol):
    def load_messages(self) -> list[Message]: ...

    def save_messages(self, messages: list[Message]) -> None: ...


class JsonFileMemory:
    """Persistent agent memory backed by a JSON file."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def path(self) -> Path:
        return self._path

    def load_messages(self) -> list[Message]:
        if not self._path.exists():
            return []

        with self._path.open("r", encoding="utf-8") as memory_file:
            data = json.load(memory_file)

        messages = data.get("messages", [])
        if not isinstance(messages, list):
            raise ValueError("Memory file must contain a messages list.")

        return [_validate_message(message) for message in messages]

    def save_messages(self, messages: list[Message]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"messages": [message.copy() for message in messages]}

        with self._path.open("w", encoding="utf-8") as memory_file:
            json.dump(payload, memory_file, ensure_ascii=False, indent=2)
            memory_file.write("\n")


def _validate_message(message: object) -> Message:
    if not isinstance(message, dict):
        raise ValueError("Each memory message must be an object.")

    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        raise ValueError("Each memory message must contain role and content strings.")

    return {"role": role, "content": content}
