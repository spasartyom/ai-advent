import unittest

from ai_advent.mcp_client import McpToolResult
from ai_advent.pipeline import run_study_note_pipeline


class PipelineTests(unittest.TestCase):
    def test_study_note_pipeline_passes_data_between_tools(self) -> None:
        calls: list[tuple[str, dict]] = []

        def fake_call(tool_name: str, arguments: dict) -> McpToolResult:
            calls.append((tool_name, arguments))
            if tool_name == "search_lessons":
                return McpToolResult(
                    tool_name=tool_name,
                    content_text="",
                    structured_content={
                        "result": [
                            {
                                "topic": "mcp",
                                "content": "MCP connects agents to tools.",
                            }
                        ]
                    },
                    is_error=False,
                )
            if tool_name == "summarize_note":
                self.assertIn("mcp: MCP connects agents to tools.", arguments["content"])
                return McpToolResult(
                    tool_name=tool_name,
                    content_text="",
                    structured_content={"result": "# Study note\n\nMCP summary"},
                    is_error=False,
                )
            if tool_name == "save_note":
                self.assertEqual(arguments["content"], "# Study note\n\nMCP summary")
                return McpToolResult(
                    tool_name=tool_name,
                    content_text="",
                    structured_content={"result": {"path": ".ai-advent/notes/mcp.md"}},
                    is_error=False,
                )
            raise AssertionError(f"Unexpected tool: {tool_name}")

        result = run_study_note_pipeline("mcp", caller=fake_call)

        self.assertEqual(
            [tool_name for tool_name, _ in calls],
            ["search_lessons", "summarize_note", "save_note"],
        )
        self.assertEqual(result.saved_path, ".ai-advent/notes/mcp.md")
        self.assertEqual(result.summary, "# Study note\n\nMCP summary")

    def test_study_note_pipeline_rejects_empty_query(self) -> None:
        with self.assertRaises(ValueError):
            run_study_note_pipeline(" ")


if __name__ == "__main__":
    unittest.main()
