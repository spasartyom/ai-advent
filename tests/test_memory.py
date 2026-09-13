import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_advent.memory import JsonFileMemory


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


if __name__ == "__main__":
    unittest.main()
