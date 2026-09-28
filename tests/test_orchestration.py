import unittest

from ai_advent.mcp_client import McpToolResult
from ai_advent.orchestration import (
    McpOrchestrator,
    McpServerConfig,
    OrchestrationStep,
    run_study_orchestration_flow,
)


class FakeOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, tool_name: str, arguments: dict) -> OrchestrationStep:
        self.calls.append((tool_name, arguments))
        if tool_name == "search_lessons":
            result = McpToolResult(
                tool_name,
                "",
                {"result": [{"topic": "mcp", "content": "MCP connects tools."}]},
                False,
            )
            return OrchestrationStep("lessons", tool_name, arguments, result)
        if tool_name == "summarize_note":
            assert "mcp: MCP connects tools." in arguments["content"]
            result = McpToolResult(
                tool_name,
                "",
                {"result": "# Study note\n\nMCP summary"},
                False,
            )
            return OrchestrationStep("notes", tool_name, arguments, result)
        if tool_name == "save_note":
            assert arguments["content"] == "# Study note\n\nMCP summary"
            result = McpToolResult(
                tool_name,
                "",
                {"result": {"path": "demo-output/day20/study-note-mcp.md"}},
                False,
            )
            return OrchestrationStep("notes", tool_name, arguments, result)
        if tool_name == "create_reminder":
            assert (
                arguments["note"]
                == "Review saved study note: demo-output/day20/study-note-mcp.md"
            )
            result = McpToolResult(
                tool_name,
                "",
                {"result": {"id": "reminder-1"}},
                False,
            )
            return OrchestrationStep("scheduler", tool_name, arguments, result)
        if tool_name == "run_due_tasks":
            result = McpToolResult(
                tool_name,
                "",
                {"result": {"summary": "Completed 1 due study reminder(s)."}},
                False,
            )
            return OrchestrationStep("scheduler", tool_name, arguments, result)
        raise AssertionError(f"Unexpected tool: {tool_name}")


class OrchestrationTests(unittest.TestCase):
    def test_study_orchestration_flow_routes_steps_across_servers(self) -> None:
        orchestrator = FakeOrchestrator()

        result = run_study_orchestration_flow(
            "mcp",
            reminder_seconds=60,
            orchestrator=orchestrator,
        )

        self.assertEqual(
            [tool_name for tool_name, _ in orchestrator.calls],
            [
                "search_lessons",
                "summarize_note",
                "save_note",
                "create_reminder",
                "run_due_tasks",
            ],
        )
        self.assertEqual(
            [step.server_name for step in result.steps],
            ["lessons", "notes", "notes", "scheduler", "scheduler"],
        )
        self.assertEqual(result.saved_path, "demo-output/day20/study-note-mcp.md")
        self.assertEqual(result.reminder_id, "reminder-1")
        self.assertEqual(
            result.scheduler_summary,
            "Completed 1 due study reminder(s).",
        )

    def test_mcp_orchestrator_requires_registered_servers(self) -> None:
        with self.assertRaises(ValueError):
            McpOrchestrator([])

    def test_mcp_orchestrator_reports_unknown_tool(self) -> None:
        orchestrator = McpOrchestrator(
            [
                McpServerConfig(
                    name="empty",
                    command=["python", "-m", "missing"],
                    env={},
                )
            ]
        )
        orchestrator.discover_tools = lambda: []

        with self.assertRaises(ValueError):
            orchestrator.call_tool("missing_tool", {})


if __name__ == "__main__":
    unittest.main()
