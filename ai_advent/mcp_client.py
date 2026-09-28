import asyncio
import sys
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class McpTool:
    name: str
    title: str | None
    description: str | None
    input_schema: dict[str, Any]


def default_study_server_command() -> list[str]:
    return [sys.executable, "-m", "ai_advent.mcp_servers.study"]


def list_tools_sync(
    *,
    url: str | None = None,
    command: list[str] | None = None,
) -> list[McpTool]:
    return asyncio.run(list_tools(url=url, command=command))


async def list_tools(
    *,
    url: str | None = None,
    command: list[str] | None = None,
) -> list[McpTool]:
    try:
        from mcp import Client, StdioServerParameters
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MCP SDK is not installed. Run: python -m pip install -e .",
        ) from error

    if url is not None and command is not None:
        raise ValueError("Use either an MCP URL or a stdio command, not both.")

    if url is not None:
        server: str | StdioServerParameters = url
    else:
        server_command = command or default_study_server_command()
        if not server_command:
            raise ValueError("MCP stdio command cannot be empty.")
        server = StdioServerParameters(
            command=server_command[0],
            args=server_command[1:],
        )

    tools: list[McpTool] = []
    async with Client(server) as client:
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(_normalize_tool(tool) for tool in page.tools)
            if page.next_cursor is None:
                return tools
            cursor = page.next_cursor


def _normalize_tool(tool: Any) -> McpTool:
    input_schema = _read_tool_attr(tool, "input_schema", "inputSchema") or {}
    return McpTool(
        name=_read_tool_attr(tool, "name"),
        title=_read_tool_attr(tool, "title"),
        description=_read_tool_attr(tool, "description"),
        input_schema=input_schema,
    )


def _read_tool_attr(tool: Any, *names: str) -> Any:
    for name in names:
        if hasattr(tool, name):
            return getattr(tool, name)
        if isinstance(tool, dict) and name in tool:
            return tool[name]
    return None
