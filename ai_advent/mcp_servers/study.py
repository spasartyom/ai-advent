import os
from pathlib import Path

from mcp.server import MCPServer

from ai_advent.scheduler import (
    DEFAULT_SCHEDULER_FILE,
    create_reminder as create_study_reminder,
    list_reminders as list_study_reminders,
    run_due_tasks as run_due_study_tasks,
)


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


if __name__ == "__main__":
    mcp.run()
