import os
import re
from pathlib import Path

from mcp.server import MCPServer

from ai_advent.scheduler import (
    DEFAULT_SCHEDULER_FILE,
    create_reminder as create_study_reminder,
    list_reminders as list_study_reminders,
    run_due_tasks as run_due_study_tasks,
)


DEFAULT_NOTES_DIR = ".ai-advent/notes"

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


@mcp.tool(title="Search study lessons")
def search_lessons(query: str) -> list[dict]:
    """Search local AI Advent study lessons by topic or content."""
    normalized_query = query.strip().lower()
    if not normalized_query:
        return []

    matches: list[dict] = []
    for topic, content in sorted(LESSONS.items()):
        searchable = f"{topic} {content}".lower()
        if normalized_query in searchable:
            matches.append(
                {
                    "topic": topic,
                    "content": content,
                }
            )
    return matches


@mcp.tool(title="Summarize study note")
def summarize_note(title: str, content: str) -> str:
    """Create a short deterministic study summary from note content."""
    compact_content = " ".join(content.split())
    if not compact_content:
        return f"# {title}\n\nNo source content was provided."

    sentences = re.split(r"(?<=[.!?])\s+", compact_content)
    summary = " ".join(sentences[:2]).strip()
    return (
        f"# {title}\n\n"
        f"## Summary\n\n{summary}\n\n"
        "## Practice\n\nExplain the idea in your own words and write one small example."
    )


@mcp.tool(title="Save study note")
def save_note(title: str, content: str) -> dict:
    """Save a study note to a Markdown file and return its path."""
    notes_dir = _notes_dir()
    notes_dir.mkdir(parents=True, exist_ok=True)
    path = notes_dir / f"{_slugify(title)}.md"
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "bytes": len(content.encode("utf-8")),
    }


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


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "study-note"


if __name__ == "__main__":
    mcp.run()
