import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_advent.memory import AgentMemoryState, JsonFileMemory


class JsonFileMemoryTests(unittest.TestCase):
    def test_missing_file_loads_empty_messages(self) -> None:
        with TemporaryDirectory() as directory:
            memory = JsonFileMemory(Path(directory) / "missing.json")

            self.assertEqual(memory.load_messages(), [])

    def test_save_and_load_messages(self) -> None:
        with TemporaryDirectory() as directory:
            memory = JsonFileMemory(Path(directory) / "nested" / "memory.json")
            messages = [
                {"role": "user", "content": "Привет"},
                {"role": "assistant", "content": "Короткий ответ"},
            ]

            memory.save_messages(messages)

            self.assertEqual(memory.load_messages(), messages)

    def test_save_and_load_state_with_summary(self) -> None:
        with TemporaryDirectory() as directory:
            memory = JsonFileMemory(Path(directory) / "memory.json")
            state = AgentMemoryState(
                summary="Пользователь строит CLI-агента.",
                facts={"goal": "build agent"},
                messages=[
                    {"role": "user", "content": "Привет"},
                    {"role": "assistant", "content": "Короткий ответ"},
                ],
                branches={
                    "main": [
                        {"role": "user", "content": "Привет"},
                    ],
                },
                checkpoints={
                    "base": [
                        {"role": "user", "content": "Привет"},
                    ],
                },
                current_branch="main",
            )

            memory.save_state(state)

            self.assertEqual(memory.load_state(), state)

    def test_old_memory_file_without_summary_loads_empty_summary(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps(
                    {
                        "messages": [
                            {"role": "user", "content": "Old memory"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            self.assertEqual(memory.load_state().summary, "")
            self.assertEqual(memory.load_state().facts, {})
            self.assertEqual(memory.load_state().branches, {})
            self.assertEqual(memory.load_state().checkpoints, {})

    def test_load_rejects_invalid_messages_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps({"messages": [{"role": "user"}]}),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_messages()

    def test_load_rejects_invalid_summary_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps({"summary": [], "messages": []}),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_state()

    def test_load_rejects_invalid_facts_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps({"facts": [], "messages": []}),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_state()


if __name__ == "__main__":
    unittest.main()
