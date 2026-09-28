from mcp.server import MCPServer

from ai_advent.study_content import (
    get_lesson_content,
    list_lesson_topics,
    search_lesson_content,
)


mcp = MCPServer(
    "AI Advent Lesson Tools",
    instructions="Expose lesson discovery tools for Study Coach orchestration.",
)


@mcp.tool(title="List study lessons")
def list_lessons() -> list[str]:
    """Return available AI Advent study lesson topics."""
    return list_lesson_topics()


@mcp.tool(title="Get study lesson")
def get_lesson(topic: str) -> str:
    """Return a short lesson for the requested topic."""
    return get_lesson_content(topic)


@mcp.tool(title="Search study lessons")
def search_lessons(query: str) -> list[dict]:
    """Search local AI Advent study lessons by topic or content."""
    return search_lesson_content(query)


if __name__ == "__main__":
    mcp.run()
