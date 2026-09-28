import os
from pathlib import Path

from mcp.server import MCPServer

from ai_advent.scheduler import (
    DEFAULT_SCHEDULER_FILE,
    create_reminder as create_study_reminder,
    list_reminders as list_study_reminders,
    run_due_tasks as run_due_study_tasks,
)
from ai_advent.study_content import (
    DEFAULT_NOTES_DIR,
    get_lesson_content,
    list_lesson_topics,
    save_study_note,
    search_lesson_content,
    summarize_study_note,
)

mcp = MCPServer(
    "AI Advent Study Tools",
    instructions="Expose small study-coach tools for the AI Advent MCP week.",
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


@mcp.tool(title="Summarize study note")
def summarize_note(title: str, content: str) -> str:
    """Create a short deterministic study summary from note content."""
    return summarize_study_note(title, content)


@mcp.tool(title="Save study note")
def save_note(title: str, content: str) -> dict:
    """Save a study note to a Markdown file and return its path."""
    return save_study_note(_notes_dir(), title=title, content=content)


@mcp.tool(title="Create study reminder")
def create_reminder(title: str, due_in_seconds: int = 0, note: str = "") -> dict:
    """Create a delayed study reminder and persist it to JSON storage."""
    return create_study_reminder(
        _scheduler_path(),
        title=title,
        due_in_seconds=due_in_seconds,
        note=note,
    )


@mcp.tool(title="List study reminders")
def list_reminders() -> dict:
    """Return persisted study reminders grouped by status."""
    return list_study_reminders(_scheduler_path())


@mcp.tool(title="Run due study tasks")
def run_due_tasks() -> dict:
    """Complete due reminders and return an aggregate scheduler summary."""
    return run_due_study_tasks(_scheduler_path())


def _scheduler_path() -> Path:
    return Path(os.getenv("AI_ADVENT_SCHEDULER_FILE", DEFAULT_SCHEDULER_FILE))


def _notes_dir() -> Path:
    return Path(os.getenv("AI_ADVENT_NOTES_DIR", DEFAULT_NOTES_DIR))


if __name__ == "__main__":
    mcp.run()
