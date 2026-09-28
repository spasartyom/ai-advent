import os
from pathlib import Path

from mcp.server import MCPServer

from ai_advent.study_content import (
    DEFAULT_NOTES_DIR,
    save_study_note,
    summarize_study_note,
)


mcp = MCPServer(
    "AI Advent Note Tools",
    instructions="Expose note processing and persistence tools for Study Coach orchestration.",
)


@mcp.tool(title="Summarize study note")
def summarize_note(title: str, content: str) -> str:
    """Create a short deterministic study summary from note content."""
    return summarize_study_note(title, content)


@mcp.tool(title="Save study note")
def save_note(title: str, content: str) -> dict:
    """Save a study note to a Markdown file and return its path."""
    return save_study_note(_notes_dir(), title=title, content=content)


def _notes_dir() -> Path:
    return Path(os.getenv("AI_ADVENT_NOTES_DIR", DEFAULT_NOTES_DIR))


if __name__ == "__main__":
    mcp.run()
