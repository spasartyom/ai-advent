import json
from dataclasses import dataclass
from typing import Any, Callable

from ai_advent.mcp_client import McpToolResult, call_tool_sync


ToolCaller = Callable[[str, dict[str, Any]], McpToolResult]


@dataclass(frozen=True)
class PipelineStep:
    tool_name: str
    arguments: dict[str, Any]
    result: McpToolResult


@dataclass(frozen=True)
class PipelineResult:
    query: str
    steps: list[PipelineStep]
    saved_path: str
    summary: str


def run_study_note_pipeline(
    query: str,
    *,
    caller: ToolCaller | None = None,
) -> PipelineResult:
    if not query.strip():
        raise ValueError("Pipeline query cannot be empty.")

    call = caller or _call_default_tool
    steps: list[PipelineStep] = []

    search_arguments = {"query": query}
    search_result = call("search_lessons", search_arguments)
    steps.append(PipelineStep("search_lessons", search_arguments, search_result))
    _raise_for_tool_error(search_result)
    search_payload = _tool_payload(search_result)
    lesson_text = _format_search_payload(search_payload)

    summarize_arguments = {
        "title": f"Study note: {query}",
        "content": lesson_text,
    }
    summarize_result = call("summarize_note", summarize_arguments)
    steps.append(PipelineStep("summarize_note", summarize_arguments, summarize_result))
    _raise_for_tool_error(summarize_result)
    summary = str(_tool_payload(summarize_result))

    save_arguments = {
        "title": f"Study note: {query}",
        "content": summary,
    }
    save_result = call("save_note", save_arguments)
    steps.append(PipelineStep("save_note", save_arguments, save_result))
    _raise_for_tool_error(save_result)
    save_payload = _tool_payload(save_result)
    saved_path = (
        str(save_payload.get("path"))
        if isinstance(save_payload, dict) and save_payload.get("path")
        else save_result.content_text
    )

    return PipelineResult(
        query=query,
        steps=steps,
        saved_path=saved_path,
        summary=summary,
    )


def _call_default_tool(tool_name: str, arguments: dict[str, Any]) -> McpToolResult:
    return call_tool_sync(tool_name, arguments)


def _raise_for_tool_error(result: McpToolResult) -> None:
    if result.is_error:
        raise RuntimeError(f"MCP tool {result.tool_name} failed: {result.content_text}")


def _tool_payload(result: McpToolResult) -> Any:
    if isinstance(result.structured_content, dict) and "result" in result.structured_content:
        return result.structured_content["result"]
    if result.structured_content is not None:
        return result.structured_content

    try:
        return json.loads(result.content_text)
    except json.JSONDecodeError:
        return result.content_text


def _format_search_payload(payload: Any) -> str:
    if isinstance(payload, list):
        if not payload:
            return "No lessons found."
        lines: list[str] = []
        for item in payload:
            if isinstance(item, dict):
                topic = item.get("topic", "unknown")
                content = item.get("content", "")
                lines.append(f"{topic}: {content}")
            else:
                lines.append(str(item))
        return "\n".join(lines)
    return str(payload)
