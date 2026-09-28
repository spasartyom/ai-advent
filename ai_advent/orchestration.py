import sys
from dataclasses import dataclass
from typing import Any

from ai_advent.mcp_client import McpTool, McpToolResult, call_tool_sync, list_tools_sync
from ai_advent.pipeline import _format_search_payload, _raise_for_tool_error, _tool_payload


@dataclass(frozen=True)
class McpServerConfig:
    name: str
    command: list[str]
    env: dict[str, str]


@dataclass(frozen=True)
class RegisteredTool:
    server_name: str
    tool: McpTool


@dataclass(frozen=True)
class OrchestrationStep:
    server_name: str
    tool_name: str
    arguments: dict[str, Any]
    result: McpToolResult


@dataclass(frozen=True)
class OrchestrationResult:
    query: str
    steps: list[OrchestrationStep]
    saved_path: str
    reminder_id: str
    summary: str
    scheduler_summary: str


class McpOrchestrator:
    def __init__(self, servers: list[McpServerConfig]) -> None:
        if not servers:
            raise ValueError("At least one MCP server must be registered.")
        self._servers = {server.name: server for server in servers}
        self._routes: dict[str, str] = {}

    @property
    def servers(self) -> dict[str, McpServerConfig]:
        return dict(self._servers)

    def discover_tools(self) -> list[RegisteredTool]:
        registered_tools: list[RegisteredTool] = []
        routes: dict[str, str] = {}
        for server in self._servers.values():
            tools = list_tools_sync(command=server.command, env=server.env)
            for tool in tools:
                if tool.name not in routes:
                    routes[tool.name] = server.name
                registered_tools.append(
                    RegisteredTool(
                        server_name=server.name,
                        tool=tool,
                    )
                )
        self._routes = routes
        return registered_tools

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> OrchestrationStep:
        if tool_name not in self._routes:
            self.discover_tools()
        server_name = self._routes.get(tool_name)
        if server_name is None:
            raise ValueError(f"No registered MCP server provides tool {tool_name!r}.")

        server = self._servers[server_name]
        result = call_tool_sync(
            tool_name,
            arguments,
            command=server.command,
            env=server.env,
        )
        return OrchestrationStep(
            server_name=server_name,
            tool_name=tool_name,
            arguments=arguments,
            result=result,
        )


def build_default_orchestrator(
    *,
    scheduler_file: str | None = None,
    notes_dir: str | None = None,
) -> McpOrchestrator:
    scheduler_env = {}
    if scheduler_file is not None:
        scheduler_env["AI_ADVENT_SCHEDULER_FILE"] = scheduler_file

    notes_env = {}
    if notes_dir is not None:
        notes_env["AI_ADVENT_NOTES_DIR"] = notes_dir

    return McpOrchestrator(
        [
            McpServerConfig(
                name="lessons",
                command=[sys.executable, "-m", "ai_advent.mcp_servers.lessons"],
                env={},
            ),
            McpServerConfig(
                name="notes",
                command=[sys.executable, "-m", "ai_advent.mcp_servers.notes"],
                env=notes_env,
            ),
            McpServerConfig(
                name="scheduler",
                command=[sys.executable, "-m", "ai_advent.mcp_servers.scheduler"],
                env=scheduler_env,
            ),
        ]
    )


def run_study_orchestration_flow(
    query: str,
    *,
    reminder_seconds: int = 0,
    orchestrator: McpOrchestrator | None = None,
) -> OrchestrationResult:
    if not query.strip():
        raise ValueError("Orchestration query cannot be empty.")
    if reminder_seconds < 0:
        raise ValueError("reminder_seconds must be zero or greater.")

    active_orchestrator = orchestrator or build_default_orchestrator()
    steps: list[OrchestrationStep] = []

    search_step = active_orchestrator.call_tool("search_lessons", {"query": query})
    steps.append(search_step)
    _raise_for_tool_error(search_step.result)
    lesson_text = _format_search_payload(_tool_payload(search_step.result))

    summarize_arguments = {
        "title": f"Study note: {query}",
        "content": lesson_text,
    }
    summarize_step = active_orchestrator.call_tool(
        "summarize_note",
        summarize_arguments,
    )
    steps.append(summarize_step)
    _raise_for_tool_error(summarize_step.result)
    summary = str(_tool_payload(summarize_step.result))

    save_arguments = {
        "title": f"Study note: {query}",
        "content": summary,
    }
    save_step = active_orchestrator.call_tool("save_note", save_arguments)
    steps.append(save_step)
    _raise_for_tool_error(save_step.result)
    save_payload = _tool_payload(save_step.result)
    saved_path = (
        str(save_payload.get("path"))
        if isinstance(save_payload, dict) and save_payload.get("path")
        else save_step.result.content_text
    )

    reminder_arguments = {
        "title": f"Review {query}",
        "due_in_seconds": reminder_seconds,
        "note": f"Review saved study note: {saved_path}",
    }
    reminder_step = active_orchestrator.call_tool(
        "create_reminder",
        reminder_arguments,
    )
    steps.append(reminder_step)
    _raise_for_tool_error(reminder_step.result)
    reminder_payload = _tool_payload(reminder_step.result)
    reminder_id = (
        str(reminder_payload.get("id"))
        if isinstance(reminder_payload, dict) and reminder_payload.get("id")
        else ""
    )

    run_due_step = active_orchestrator.call_tool("run_due_tasks", {})
    steps.append(run_due_step)
    _raise_for_tool_error(run_due_step.result)
    run_due_payload = _tool_payload(run_due_step.result)
    scheduler_summary = (
        str(run_due_payload.get("summary"))
        if isinstance(run_due_payload, dict) and run_due_payload.get("summary")
        else run_due_step.result.content_text
    )

    return OrchestrationResult(
        query=query,
        steps=steps,
        saved_path=saved_path,
        reminder_id=reminder_id,
        summary=summary,
        scheduler_summary=scheduler_summary,
    )
