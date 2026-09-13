import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from ai_advent.cli import parse_args, read_paste_block, run_agent
from ai_advent.tokens import TokenReport


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

    def test_agent_command_accepts_token_options(self) -> None:
        args = parse_args(["agent", "--show-tokens"])

        self.assertTrue(args.show_tokens)

    def test_run_agent_reads_user_messages_until_exit(self) -> None:
        agent = FakeAgent()
        inputs = iter(["Hello", "/exit"])
        output = io.StringIO()

        with redirect_stdout(output):
            run_agent(agent, "test-model", None, read_input=lambda _: next(inputs))

        self.assertEqual(agent.messages, ["Hello"])
        self.assertIn("AI Advent agent", output.getvalue())
        self.assertIn("Assistant: agent answer to Hello", output.getvalue())
        self.assertIn("Bye!", output.getvalue())

    def test_run_agent_sends_paste_block_as_one_message(self) -> None:
        agent = FakeAgent()
        inputs = iter(["/paste", "line one", "line two", "/send", "/exit"])
        output = io.StringIO()

        with redirect_stdout(output):
            run_agent(agent, "test-model", None, read_input=lambda _: next(inputs))

        self.assertEqual(agent.messages, ["line one\nline two"])
        self.assertIn("Paste multiline text", output.getvalue())

    def test_read_paste_block_returns_lines_until_send(self) -> None:
        inputs = iter(["first", "second", "/send"])
        output = io.StringIO()

        with redirect_stdout(output):
            message = read_paste_block(lambda _: next(inputs))

        self.assertEqual(message, "first\nsecond")

    def test_run_agent_prints_token_report_when_enabled(self) -> None:
        class TokenAgent(FakeAgent):
            def run_turn(self, message: str) -> object:
                self.messages.append(message)
                return SimpleNamespace(
                    text="agent answer",
                    token_report=TokenReport(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                    ),
                )

        agent = TokenAgent()
        inputs = iter(["Hello", "/exit"])
        output = io.StringIO()

        with redirect_stdout(output):
            run_agent(
                agent,
                "test-model",
                None,
                show_tokens=True,
                read_input=lambda _: next(inputs),
            )

        self.assertIn("Tokens:", output.getvalue())
        self.assertIn("prompt_tokens: 10", output.getvalue())
        self.assertIn("completion_tokens: 5", output.getvalue())
        self.assertIn("total_tokens: 15", output.getvalue())


if __name__ == "__main__":
    unittest.main()
