import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from ai_advent.cli import (
    build_context_strategy,
    handle_branch_command,
    handle_invariant_command,
    handle_memory_command,
    handle_profile_command,
    handle_task_command,
    parse_args,
    read_paste_block,
    run_agent,
)
from ai_advent.context import (
    FullContextStrategy,
    SlidingWindowContextStrategy,
    StickyFactsContextStrategy,
    SummaryContextStrategy,
)
from ai_advent.task import create_task_state
from ai_advent.tokens import TokenReport


class FakeAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.working_memory: dict[str, str] = {}
        self.long_term_memory: dict[str, str] = {}
        self.user_profile: dict[str, str] = {}
        self.task_state = create_task_state()
        self.invariants: dict[str, str] = {}

    def run_turn(self, message: str) -> object:
        self.messages.append(message)
        return SimpleNamespace(text=f"agent answer to {message}")

    def remember_working(self, key: str, value: str) -> None:
        self.working_memory[key] = value

    def remember_long_term(self, key: str, value: str) -> None:
        self.long_term_memory[key] = value

    def forget_working(self, key: str) -> None:
        self.working_memory.pop(key, None)

    def forget_long_term(self, key: str) -> None:
        self.long_term_memory.pop(key, None)

    def remember_profile(self, key: str, value: str) -> None:
        self.user_profile[key] = value

    def forget_profile(self, key: str) -> None:
        self.user_profile.pop(key, None)

    def start_task(self, title: str) -> None:
        self.task_state = create_task_state(
            stage="planning",
            title=title,
            current_step="Сформировать план учебной задачи.",
            expected_action="Подготовить или уточнить план.",
        )

    def update_task_state(
        self,
        *,
        stage: str | None = None,
        current_step: str | None = None,
        expected_action: str | None = None,
        title: str | None = None,
        paused: bool | None = None,
    ) -> None:
        self.task_state = create_task_state(
            stage=self.task_state.stage if stage is None else stage,
            title=self.task_state.title if title is None else title,
            current_step=(
                self.task_state.current_step
                if current_step is None
                else current_step
            ),
            expected_action=(
                self.task_state.expected_action
                if expected_action is None
                else expected_action
            ),
            paused=self.task_state.paused if paused is None else paused,
        )

    def pause_task(self) -> None:
        self.update_task_state(paused=True)

    def resume_task(self) -> None:
        self.update_task_state(paused=False)

    def remember_invariant(self, key: str, value: str) -> None:
        self.invariants[key] = value

    def forget_invariant(self, key: str) -> None:
        self.invariants.pop(key, None)


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

    def test_agent_command_accepts_context_strategy_options(self) -> None:
        args = parse_args(
            [
                "agent",
                "--context-strategy",
                "summary",
                "--keep-last",
                "4",
                "--branch",
                "option_a",
            ]
        )

        self.assertEqual(args.context_strategy, "summary")
        self.assertEqual(args.keep_last, 4)
        self.assertEqual(args.branch, "option_a")

    def test_build_context_strategy_returns_full_strategy(self) -> None:
        self.assertIsInstance(build_context_strategy("full", 10), FullContextStrategy)

    def test_build_context_strategy_returns_summary_strategy(self) -> None:
        self.assertIsInstance(
            build_context_strategy("summary", 10),
            SummaryContextStrategy,
        )

    def test_build_context_strategy_returns_sliding_window_strategy(self) -> None:
        self.assertIsInstance(
            build_context_strategy("sliding-window", 10),
            SlidingWindowContextStrategy,
        )

    def test_build_context_strategy_returns_facts_strategy(self) -> None:
        self.assertIsInstance(
            build_context_strategy("facts", 10),
            StickyFactsContextStrategy,
        )

    def test_branch_command_creates_checkpoint(self) -> None:
        class BranchAgent(FakeAgent):
            current_branch = "main"

            def __init__(self) -> None:
                super().__init__()
                self.checkpoints: list[str] = []

            def save_checkpoint(self, name: str) -> None:
                self.checkpoints.append(name)

        agent = BranchAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_branch_command(agent, "/checkpoint base")

        self.assertTrue(handled)
        self.assertEqual(agent.checkpoints, ["base"])

    def test_memory_command_saves_working_memory(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_memory_command(
                agent,
                "/memory set working lesson_topic Python decorators",
            )

        self.assertTrue(handled)
        self.assertEqual(agent.working_memory, {"lesson_topic": "Python decorators"})
        self.assertIn("Working memory saved: lesson_topic", output.getvalue())

    def test_memory_command_saves_quoted_value_without_quotes(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_memory_command(
                agent,
                '/memory set working current_exercise "write a logging decorator"',
            )

        self.assertTrue(handled)
        self.assertEqual(
            agent.working_memory,
            {"current_exercise": "write a logging decorator"},
        )

    def test_memory_command_saves_long_term_memory(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_memory_command(
                agent,
                "/memory set long preferred_language ru",
            )

        self.assertTrue(handled)
        self.assertEqual(agent.long_term_memory, {"preferred_language": "ru"})
        self.assertIn("Long-term memory saved: preferred_language", output.getvalue())

    def test_run_agent_handles_memory_command_without_model_turn(self) -> None:
        agent = FakeAgent()
        inputs = iter(
            [
                "/memory set working lesson_topic Python decorators",
                "/exit",
            ]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            run_agent(agent, "test-model", None, read_input=lambda _: next(inputs))

        self.assertEqual(agent.messages, [])
        self.assertEqual(agent.working_memory, {"lesson_topic": "Python decorators"})

    def test_profile_command_saves_user_profile(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_profile_command(
                agent,
                "/profile set answer_style short then practice",
            )

        self.assertTrue(handled)
        self.assertEqual(agent.user_profile, {"answer_style": "short then practice"})
        self.assertIn("Profile saved: answer_style", output.getvalue())

    def test_profile_command_saves_quoted_value_without_quotes(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_profile_command(
                agent,
                '/profile set answer_style "short, then practice"',
            )

        self.assertTrue(handled)
        self.assertEqual(agent.user_profile, {"answer_style": "short, then practice"})

    def test_profile_command_shows_user_profile(self) -> None:
        agent = FakeAgent()
        agent.user_profile["language"] = "ru"
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_profile_command(agent, "/profile show")

        self.assertTrue(handled)
        self.assertIn("User profile:", output.getvalue())
        self.assertIn("language: ru", output.getvalue())

    def test_run_agent_handles_profile_command_without_model_turn(self) -> None:
        agent = FakeAgent()
        inputs = iter(["/profile set language ru", "/exit"])
        output = io.StringIO()

        with redirect_stdout(output):
            run_agent(agent, "test-model", None, read_input=lambda _: next(inputs))

        self.assertEqual(agent.messages, [])
        self.assertEqual(agent.user_profile, {"language": "ru"})

    def test_task_command_starts_task(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_task_command(agent, '/task start "Learn decorators"')

        self.assertTrue(handled)
        self.assertEqual(agent.task_state.stage, "planning")
        self.assertEqual(agent.task_state.title, "Learn decorators")
        self.assertIn("Task state:", output.getvalue())

    def test_task_command_updates_stage_step_and_expected_action(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handle_task_command(agent, "/task start decorators")
            handle_task_command(agent, "/task stage execution")
            handle_task_command(agent, '/task step "Solve exercise"')
            handled = handle_task_command(agent, '/task expect "Submit solution"')

        self.assertTrue(handled)
        self.assertEqual(agent.task_state.stage, "execution")
        self.assertEqual(agent.task_state.current_step, "Solve exercise")
        self.assertEqual(agent.task_state.expected_action, "Submit solution")

    def test_task_command_pauses_and_resumes_task(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handle_task_command(agent, "/task start decorators")
            handle_task_command(agent, "/task pause")
            self.assertTrue(agent.task_state.paused)
            handled = handle_task_command(agent, "/task resume")

        self.assertTrue(handled)
        self.assertFalse(agent.task_state.paused)

    def test_run_agent_handles_task_command_without_model_turn(self) -> None:
        agent = FakeAgent()
        inputs = iter(['/task start "Learn decorators"', "/exit"])
        output = io.StringIO()

        with redirect_stdout(output):
            run_agent(agent, "test-model", None, read_input=lambda _: next(inputs))

        self.assertEqual(agent.messages, [])
        self.assertEqual(agent.task_state.title, "Learn decorators")

    def test_invariant_command_saves_invariant(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_invariant_command(
                agent,
                '/invariant add no_full_solution "Do not give full solution"',
            )

        self.assertTrue(handled)
        self.assertEqual(
            agent.invariants,
            {"no_full_solution": "Do not give full solution"},
        )
        self.assertIn("Invariant saved: no_full_solution", output.getvalue())

    def test_invariant_command_shows_invariants(self) -> None:
        agent = FakeAgent()
        agent.invariants["no_full_solution"] = "Do not give full solution"
        output = io.StringIO()

        with redirect_stdout(output):
            handled = handle_invariant_command(agent, "/invariant show")

        self.assertTrue(handled)
        self.assertIn("Invariants:", output.getvalue())
        self.assertIn("no_full_solution: Do not give full solution", output.getvalue())

    def test_run_agent_handles_invariant_command_without_model_turn(self) -> None:
        agent = FakeAgent()
        inputs = iter(
            [
                '/invariant add no_full_solution "Do not give full solution"',
                "/exit",
            ]
        )
        output = io.StringIO()

        with redirect_stdout(output):
            run_agent(agent, "test-model", None, read_input=lambda _: next(inputs))

        self.assertEqual(agent.messages, [])
        self.assertEqual(
            agent.invariants,
            {"no_full_solution": "Do not give full solution"},
        )

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
