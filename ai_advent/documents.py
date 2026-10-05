from dataclasses import dataclass
from pathlib import Path


TEXT_FILE_SUFFIXES = {".md", ".markdown", ".py", ".txt", ".rst"}
SKIPPED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
}


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


def load_documents(paths: list[str | Path]) -> list[Document]:
    documents: list[Document] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_dir():
            documents.extend(_load_directory(path))
            continue
        if path.is_file() and _is_text_file(path):
            documents.append(_load_file(path))

    return sorted(documents, key=lambda document: document.source)


def _load_directory(path: Path) -> list[Document]:
    documents: list[Document] = []
    for child in sorted(path.rglob("*")):
        if any(part in SKIPPED_DIRS for part in child.parts):
            continue
        if child.is_file() and _is_text_file(child):
            documents.append(_load_file(child))
    return documents


def _load_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    return Document(
        source=str(path),
        title=path.name,
        text=text,
    )


def _is_text_file(path: Path) -> bool:
    return path.suffix.lower() in TEXT_FILE_SUFFIXES
