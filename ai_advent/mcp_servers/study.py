from mcp.server import MCPServer


mcp = MCPServer(
    "AI Advent Study Tools",
    instructions="Expose small study-coach tools for the AI Advent MCP week.",
)

LESSONS = {
    "memory": "Agent memory can be split into short-term dialog, working memory for the current task, and long-term memory for stable facts.",
    "task_state": "A Study Coach task state tracks stage, title, current step, expected action, and whether the task is paused.",
    "mcp": "Model Context Protocol lets an agent discover and call external tools through a standard client-server protocol.",
}


@mcp.tool(title="List study lessons")
def list_lessons() -> list[str]:
    """Return available AI Advent study lesson topics."""
    return sorted(LESSONS)


@mcp.tool(title="Get study lesson")
def get_lesson(topic: str) -> str:
    """Return a short lesson for the requested topic."""
    return LESSONS.get(
        topic,
        f"No local lesson found for {topic!r}. Available topics: {', '.join(sorted(LESSONS))}.",
    )


if __name__ == "__main__":
    mcp.run()
