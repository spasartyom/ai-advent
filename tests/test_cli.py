import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from ai_advent.cli import parse_args, run_agent


class FakeAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def run_turn(self, message: str) -> object:
        self.messages.append(message)
        return SimpleNamespace(text=f"agent answer to {message}")


class CliTests(unittest.TestCase):
    def test_agent_command_is_available(self) -> None:
        args = parse_args(["agent"])

        self.assertEqual(args.command, "agent")
        self.assertIsNone(args.memory_file)

    def test_agent_command_accepts_memory_file(self) -> None:
        args = parse_args(["agent", "--memory-file", "custom-memory.json"])

        self.assertEqual(args.memory_file, "custom-memory.json")

    def test_run_agent_reads_user_messages_until_exit(self) -> None:
        agent = FakeAgent()
        inputs = iter(["Hello", "/exit"])
        output = io.StringIO()

        with redirect_stdout(output):
            run_agent(agent, "test-model", None, lambda _: next(inputs))

        self.assertEqual(agent.messages, ["Hello"])
        self.assertIn("AI Advent agent", output.getvalue())
        self.assertIn("Assistant: agent answer to Hello", output.getvalue())
        self.assertIn("Bye!", output.getvalue())


if __name__ == "__main__":
    unittest.main()
