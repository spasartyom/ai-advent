import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from ai_advent.cli import (
    build_context_strategy,
    build_mcp_env,
    handle_branch_command,
    handle_invariant_command,
    handle_lesson_command,
    handle_memory_command,
    handle_note_command,
    handle_profile_command,
    handle_reminder_command,
    handle_task_command,
    parse_args,
    parse_tool_arguments,
    print_orchestration_result,
    print_pipeline_result,
    print_registered_tools,
    print_mcp_tool_result,
    print_mcp_tools,
    read_paste_block,
    run_agent,
    run_worker,
)
from ai_advent.context import (
    FullContextStrategy,
    SlidingWindowContextStrategy,
    StickyFactsContextStrategy,
    SummaryContextStrategy,
)
from ai_advent.mcp_client import McpTool, McpToolResult
from ai_advent.orchestration import OrchestrationResult, OrchestrationStep, RegisteredTool
from ai_advent.pipeline import PipelineResult, PipelineStep
from ai_advent.task import create_task_state, validate_task_transition
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
        validate_task_transition(self.task_state.stage, "planning")
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
        next_stage = self.task_state.stage if stage is None else stage
        validate_task_transition(self.task_state.stage, next_stage)
        self.task_state = create_task_state(
            stage=next_stage,
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

    def approve_task(self) -> None:
        validate_task_transition(
            self.task_state.stage,
            "execution",
            approved=True,
        )
        self.task_state = create_task_state(
            stage="execution",
            title=self.task_state.title,
            current_step="Выполнить утвержденный план.",
            expected_action="Продолжить выполнение задачи.",
            paused=False,
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

    def test_mcp_list_tools_command_is_available(self) -> None:
        args = parse_args(["mcp", "list-tools"])

        self.assertEqual(args.command, "mcp")
        self.assertEqual(args.mcp_command, "list-tools")
        self.assertIsNone(args.url)
        self.assertIsNone(args.server_command)

    def test_mcp_list_tools_command_accepts_url_and_stdio_command(self) -> None:
        args = parse_args(
            [
                "mcp",
                "list-tools",
                "--url",
                "http://127.0.0.1:8000/mcp",
                "--server-command",
                "python server.py",
            ]
        )

        self.assertEqual(args.url, "http://127.0.0.1:8000/mcp")
        self.assertEqual(args.server_command, "python server.py")

    def test_mcp_call_tool_command_is_available(self) -> None:
        args = parse_args(
            [
                "mcp",
                "call-tool",
                "get_lesson",
                "--arguments",
                '{"topic": "mcp"}',
            ]
        )

        self.assertEqual(args.command, "mcp")
        self.assertEqual(args.mcp_command, "call-tool")
        self.assertEqual(args.tool_name, "get_lesson")
        self.assertEqual(args.arguments, '{"topic": "mcp"}')

    def test_mcp_run_pipeline_command_is_available(self) -> None:
        args = parse_args(
            [
                "mcp",
                "run-pipeline",
                "mcp",
                "--notes-dir",
                "demo-notes",
            ]
        )

        self.assertEqual(args.command, "mcp")
        self.assertEqual(args.mcp_command, "run-pipeline")
        self.assertEqual(args.query, "mcp")
        self.assertEqual(args.notes_dir, "demo-notes")

    def test_mcp_orchestrate_command_is_available(self) -> None:
        args = parse_args(
            [
                "mcp",
                "orchestrate",
                "mcp",
                "--notes-dir",
                "demo-notes",
                "--scheduler-file",
                "demo-scheduler.json",
                "--remind-in",
                "60",
            ]
        )

        self.assertEqual(args.command, "mcp")
        self.assertEqual(args.mcp_command, "orchestrate")
        self.assertEqual(args.query, "mcp")
        self.assertEqual(args.notes_dir, "demo-notes")
        self.assertEqual(args.scheduler_file, "demo-scheduler.json")
        self.assertEqual(args.remind_in, 60)

    def test_worker_command_accepts_scheduler_options(self) -> None:
        args = parse_args(
            [
                "worker",
                "--once",
                "--interval",
                "0.5",
                "--scheduler-file",
                "demo-scheduler.json",
            ]
        )

        self.assertEqual(args.command, "worker")
        self.assertTrue(args.once)
        self.assertEqual(args.interval, 0.5)
        self.assertEqual(args.scheduler_file, "demo-scheduler.json")

    def test_index_build_command_is_available(self) -> None:
        args = parse_args(
            [
                "index",
                "build",
                "README.md",
                "ai_advent",
                "--output",
                ".ai-advent/test-index.json",
                "--strategy",
                "fixed",
                "--offline-embeddings",
            ]
        )

        self.assertEqual(args.command, "index")
        self.assertEqual(args.index_command, "build")
        self.assertEqual(args.paths, ["README.md", "ai_advent"])
        self.assertEqual(args.output, ".ai-advent/test-index.json")
        self.assertEqual(args.strategy, "fixed")
        self.assertTrue(args.offline_embeddings)

    def test_index_build_command_accepts_ollama_embeddings(self) -> None:
        args = parse_args(
            [
                "index",
                "build",
                "README.md",
                "--embedding-provider",
                "ollama",
                "--embedding-model",
                "nomic-embed-text",
                "--ollama-url",
                "http://127.0.0.1:11434",
            ]
        )

        self.assertEqual(args.embedding_provider, "ollama")
        self.assertEqual(args.embedding_model, "nomic-embed-text")
        self.assertEqual(args.ollama_url, "http://127.0.0.1:11434")

    def test_rag_ask_command_is_available(self) -> None:
        args = parse_args(
            [
                "rag",
                "ask",
                "Какие команды управляют task state?",
                "--index",
                ".ai-advent/document-index.json",
                "--top-k",
                "3",
                "--embedding-provider",
                "ollama",
                "--embedding-model",
                "nomic-embed-text",
            ]
        )

        self.assertEqual(args.command, "rag")
        self.assertEqual(args.rag_command, "ask")
        self.assertEqual(args.question, "Какие команды управляют task state?")
        self.assertEqual(args.index, ".ai-advent/document-index.json")
        self.assertEqual(args.top_k, 3)
        self.assertEqual(args.embedding_provider, "ollama")

    def test_rag_compare_command_is_available(self) -> None:
        args = parse_args(["rag", "compare", "Что делает pipeline Дня 19?"])

        self.assertEqual(args.command, "rag")
        self.assertEqual(args.rag_command, "compare")
        self.assertEqual(args.question, "Что делает pipeline Дня 19?")

    def test_rag_eval_questions_command_is_available(self) -> None:
        args = parse_args(["rag", "eval-questions"])

        self.assertEqual(args.command, "rag")
        self.assertEqual(args.rag_command, "eval-questions")

    def test_build_mcp_env_includes_scheduler_file_when_present(self) -> None:
        self.assertEqual(
            build_mcp_env("demo-scheduler.json"),
            {"AI_ADVENT_SCHEDULER_FILE": "demo-scheduler.json"},
        )
        self.assertEqual(
            build_mcp_env(notes_dir="demo-notes"),
            {"AI_ADVENT_NOTES_DIR": "demo-notes"},
        )
        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(build_mcp_env(None), {})

    def test_parse_tool_arguments_requires_json_object(self) -> None:
        self.assertEqual(parse_tool_arguments('{"topic": "mcp"}'), {"topic": "mcp"})

        with self.assertRaises(ValueError):
            parse_tool_arguments('["mcp"]')

    def test_print_mcp_tools_outputs_tool_details(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            print_mcp_tools(
                [
                    McpTool(
                        name="get_lesson",
                        title="Get study lesson",
                        description="Return a lesson.",
                        input_schema={"type": "object"},
                    )
                ]
            )

        printed = output.getvalue()
        self.assertIn("MCP tools: 1", printed)
        self.assertIn("get_lesson: Get study lesson", printed)
        self.assertIn("description: Return a lesson.", printed)
        self.assertIn("input_schema: {'type': 'object'}", printed)

    def test_print_mcp_tool_result_outputs_content(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            print_mcp_tool_result(
                McpToolResult(
                    tool_name="get_lesson",
                    content_text="MCP lesson",
                    structured_content=None,
                    is_error=False,
                )
            )

        printed = output.getvalue()
        self.assertIn("MCP tool result: get_lesson", printed)
        self.assertIn("is_error: False", printed)
        self.assertIn("MCP lesson", printed)

    def test_lesson_command_calls_mcp_tool_and_uses_result_in_agent_turn(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()

        with patch(
            "ai_advent.cli.call_tool_sync",
            return_value=McpToolResult(
                tool_name="get_lesson",
                content_text="MCP explains external tools.",
                structured_content=None,
                is_error=False,
            ),
        ) as call_tool, redirect_stdout(output):
            handled = handle_lesson_command(agent, "/lesson mcp")

        self.assertTrue(handled)
        call_tool.assert_called_once_with("get_lesson", {"topic": "mcp"})
        self.assertEqual(len(agent.messages), 1)
        self.assertIn("MCP explains external tools.", agent.messages[0])
        self.assertIn("Assistant: agent answer", output.getvalue())

    def test_remind_command_calls_create_reminder_tool(self) -> None:
        output = io.StringIO()

        with patch(
            "ai_advent.cli.call_tool_sync",
            return_value=McpToolResult(
                tool_name="create_reminder",
                content_text="created",
                structured_content={"id": "abc"},
                is_error=False,
            ),
        ) as call_tool, redirect_stdout(output):
            handled = handle_reminder_command("/remind 5 Review MCP")

        self.assertTrue(handled)
        call_tool.assert_called_once_with(
            "create_reminder",
            {
                "title": "Review MCP",
                "due_in_seconds": 5,
            },
        )
        self.assertIn("MCP tool result: create_reminder", output.getvalue())

    def test_note_command_runs_study_note_pipeline(self) -> None:
        output = io.StringIO()
        result = PipelineResult(
            query="mcp",
            steps=[
                PipelineStep(
                    tool_name="search_lessons",
                    arguments={"query": "mcp"},
                    result=McpToolResult(
                        "search_lessons",
                        "",
                        [{"topic": "mcp"}],
                        False,
                    ),
                )
            ],
            saved_path="demo-notes/study-note-mcp.md",
            summary="# Study note",
        )

        with patch(
            "ai_advent.cli.run_study_note_pipeline",
            return_value=result,
        ) as run_pipeline, redirect_stdout(output):
            handled = handle_note_command("/note mcp")

        self.assertTrue(handled)
        run_pipeline.assert_called_once()
        self.assertIn("MCP pipeline: study note for mcp", output.getvalue())
        self.assertIn("Saved note: demo-notes/study-note-mcp.md", output.getvalue())

    def test_print_pipeline_result_outputs_steps_and_saved_path(self) -> None:
        output = io.StringIO()
        result = PipelineResult(
            query="mcp",
            steps=[
                PipelineStep(
                    tool_name="save_note",
                    arguments={"title": "Study note: mcp"},
                    result=McpToolResult("save_note", "", None, False),
                )
            ],
            saved_path="demo-notes/study-note-mcp.md",
            summary="# Study note",
        )

        with redirect_stdout(output):
            print_pipeline_result(result)

        printed = output.getvalue()
        self.assertIn("1. save_note", printed)
        self.assertIn("Saved note: demo-notes/study-note-mcp.md", printed)

    def test_print_registered_tools_groups_tools_by_server(self) -> None:
        output = io.StringIO()

        with redirect_stdout(output):
            print_registered_tools(
                [
                    RegisteredTool(
                        server_name="lessons",
                        tool=McpTool(
                            name="search_lessons",
                            title="Search lessons",
                            description=None,
                            input_schema={},
                        ),
                    ),
                    RegisteredTool(
                        server_name="notes",
                        tool=McpTool(
                            name="save_note",
                            title="Save note",
                            description=None,
                            input_schema={},
                        ),
                    ),
                ]
            )

        printed = output.getvalue()
        self.assertIn("lessons: search_lessons", printed)
        self.assertIn("notes: save_note", printed)

    def test_print_orchestration_result_outputs_server_tool_steps(self) -> None:
        output = io.StringIO()
        result = OrchestrationResult(
            query="mcp",
            steps=[
                OrchestrationStep(
                    server_name="lessons",
                    tool_name="search_lessons",
                    arguments={"query": "mcp"},
                    result=McpToolResult("search_lessons", "", None, False),
                )
            ],
            saved_path="demo-output/day20/study-note-mcp.md",
            reminder_id="reminder-1",
            summary="# Study note",
            scheduler_summary="Completed 1 due study reminder.",
        )

        with redirect_stdout(output):
            print_orchestration_result(result)

        printed = output.getvalue()
        self.assertIn("1. lessons.search_lessons", printed)
        self.assertIn("Saved note: demo-output/day20/study-note-mcp.md", printed)
        self.assertIn("Reminder id: reminder-1", printed)
        self.assertIn("Scheduler summary: Completed 1 due study reminder.", printed)

    def test_worker_once_calls_run_due_tasks_tool(self) -> None:
        args = parse_args(
            [
                "worker",
                "--once",
                "--scheduler-file",
                "demo-scheduler.json",
            ]
        )
        output = io.StringIO()

        with patch(
            "ai_advent.cli.call_tool_sync",
            return_value=McpToolResult(
                tool_name="run_due_tasks",
                content_text="No due reminders",
                structured_content=None,
                is_error=False,
            ),
        ) as call_tool, redirect_stdout(output):
            run_worker(args)

        call_tool.assert_called_once_with(
            "run_due_tasks",
            command=None,
            env={"AI_ADVENT_SCHEDULER_FILE": "demo-scheduler.json"},
        )
        self.assertIn("AI Advent worker", output.getvalue())

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
            handle_task_command(agent, "/task approve")
            handle_task_command(agent, '/task step "Solve exercise"')
            handled = handle_task_command(agent, '/task expect "Submit solution"')

        self.assertTrue(handled)
        self.assertEqual(agent.task_state.stage, "execution")
        self.assertEqual(agent.task_state.current_step, "Solve exercise")
        self.assertEqual(agent.task_state.expected_action, "Submit solution")

    def test_task_command_rejects_execution_before_approval(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()
        errors = io.StringIO()

        with redirect_stdout(output), redirect_stderr(errors):
            handle_task_command(agent, "/task start decorators")
            handled = handle_task_command(agent, "/task stage execution")

        self.assertTrue(handled)
        self.assertEqual(agent.task_state.stage, "planning")
        self.assertIn("before the plan is approved", errors.getvalue())

    def test_task_command_rejects_done_before_validation(self) -> None:
        agent = FakeAgent()
        output = io.StringIO()
        errors = io.StringIO()

        with redirect_stdout(output), redirect_stderr(errors):
            handle_task_command(agent, "/task start decorators")
            handle_task_command(agent, "/task approve")
            handled = handle_task_command(agent, "/task done")

        self.assertTrue(handled)
        self.assertEqual(agent.task_state.stage, "execution")
        self.assertIn("Cannot transition task from execution to done", errors.getvalue())

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
