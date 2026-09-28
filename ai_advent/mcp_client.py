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


@dataclass(frozen=True)
class McpToolResult:
    tool_name: str
    content_text: str
    structured_content: Any
    is_error: bool


def default_study_server_command() -> list[str]:
    return [sys.executable, "-m", "ai_advent.mcp_servers.study"]


def list_tools_sync(
    *,
    url: str | None = None,
    command: list[str] | None = None,
) -> list[McpTool]:
    return asyncio.run(list_tools(url=url, command=command))


def call_tool_sync(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    url: str | None = None,
    command: list[str] | None = None,
) -> McpToolResult:
    return asyncio.run(
        call_tool(
            tool_name,
            arguments,
            url=url,
            command=command,
        )
    )


async def list_tools(
    *,
    url: str | None = None,
    command: list[str] | None = None,
) -> list[McpTool]:
    server = _build_server(url=url, command=command)

    tools: list[McpTool] = []
    async with _mcp_client(server) as client:
        cursor: str | None = None
        while True:
            page = await client.list_tools(cursor=cursor)
            tools.extend(_normalize_tool(tool) for tool in page.tools)
            if page.next_cursor is None:
                return tools
            cursor = page.next_cursor


async def call_tool(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    *,
    url: str | None = None,
    command: list[str] | None = None,
) -> McpToolResult:
    server = _build_server(url=url, command=command)

    async with _mcp_client(server) as client:
        result = await client.call_tool(tool_name, arguments or {})
        return McpToolResult(
            tool_name=tool_name,
            content_text=_content_to_text(_read_result_attr(result, "content") or []),
            structured_content=_read_result_attr(
                result,
                "structured_content",
                "structuredContent",
            ),
            is_error=bool(_read_result_attr(result, "is_error", "isError")),
        )


def _build_server(
    *,
    url: str | None = None,
    command: list[str] | None = None,
) -> Any:
    try:
        from mcp import StdioServerParameters
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MCP SDK is not installed. Run: python -m pip install -e .",
        ) from error

    if url is not None and command is not None:
        raise ValueError("Use either an MCP URL or a stdio command, not both.")

    if url is not None:
        return url

    server_command = command or default_study_server_command()
    if not server_command:
        raise ValueError("MCP stdio command cannot be empty.")
    return StdioServerParameters(
        command=server_command[0],
        args=server_command[1:],
    )


def _mcp_client(server: Any) -> Any:
    try:
        from mcp import Client
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "MCP SDK is not installed. Run: python -m pip install -e .",
        ) from error
    return Client(server)


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


def _read_result_attr(result: Any, *names: str) -> Any:
    return _read_tool_attr(result, *names)


def _content_to_text(content: list[Any]) -> str:
    texts: list[str] = []
    for block in content:
        text = _read_tool_attr(block, "text")
        if text is not None:
            texts.append(str(text))
        elif isinstance(block, dict):
            texts.append(str(block))
    return "\n".join(texts)
