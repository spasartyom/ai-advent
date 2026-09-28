import re
from pathlib import Path


LESSONS = {
    "memory": "Agent memory can be split into short-term dialog, working memory for the current task, and long-term memory for stable facts.",
    "task_state": "A Study Coach task state tracks stage, title, current step, expected action, and whether the task is paused.",
    "mcp": "Model Context Protocol lets an agent discover and call external tools through a standard client-server protocol.",
}
DEFAULT_NOTES_DIR = ".ai-advent/notes"


def list_lesson_topics() -> list[str]:
    return sorted(LESSONS)


def get_lesson_content(topic: str) -> str:
    return LESSONS.get(
        topic,
        f"No local lesson found for {topic!r}. Available topics: {', '.join(list_lesson_topics())}.",
    )


def search_lesson_content(query: str) -> list[dict]:
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


def summarize_study_note(title: str, content: str) -> str:
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


def save_study_note(notes_dir: Path, *, title: str, content: str) -> dict:
    notes_dir.mkdir(parents=True, exist_ok=True)
    path = notes_dir / f"{slugify(title)}.md"
    path.write_text(content, encoding="utf-8")
    return {
        "path": str(path),
        "bytes": len(content.encode("utf-8")),
    }


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "study-note"
