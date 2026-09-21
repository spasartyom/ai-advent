import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_advent.memory import AgentMemoryState, JsonFileMemory
from ai_advent.task import create_task_state


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
                working_memory={"lesson_topic": "Python decorators"},
                long_term_memory={"preferred_language": "ru"},
                user_profile={"answer_style": "short, then practice"},
                task_state=create_task_state(
                    stage="execution",
                    title="Learn decorators",
                    current_step="Solve logging decorator exercise",
                    expected_action="Submit solution",
                    paused=True,
                ),
                invariants={"no_full_solution": "Do not give full exercise solution before user attempt."},
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
            self.assertEqual(memory.load_state().working_memory, {})
            self.assertEqual(memory.load_state().long_term_memory, {})
            self.assertEqual(memory.load_state().user_profile, {})
            self.assertEqual(memory.load_state().task_state, create_task_state())
            self.assertEqual(memory.load_state().invariants, {})
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

    def test_load_rejects_invalid_working_memory_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps({"working_memory": {"topic": 123}, "messages": []}),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_state()

    def test_load_rejects_invalid_long_term_memory_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps({"long_term_memory": {"language": []}, "messages": []}),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_state()

    def test_load_rejects_invalid_user_profile_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps({"user_profile": {"answer_style": []}, "messages": []}),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_state()

    def test_load_rejects_invalid_task_state_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps({"task_state": [], "messages": []}),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_state()

    def test_load_rejects_unknown_task_stage(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps(
                    {
                        "task_state": {
                            "stage": "skipped",
                            "title": "Bad task",
                            "current_step": "",
                            "expected_action": "",
                            "paused": False,
                        },
                        "messages": [],
                    }
                ),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_state()

    def test_load_rejects_invalid_invariants_shape(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"
            path.write_text(
                json.dumps({"invariants": {"no_full_solution": []}, "messages": []}),
                encoding="utf-8",
            )
            memory = JsonFileMemory(path)

            with self.assertRaises(ValueError):
                memory.load_state()


if __name__ == "__main__":
    unittest.main()
