from dataclasses import dataclass

from ai_advent.documents import Document


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    source: str
    title: str
    section: str
    text: str


@dataclass(frozen=True)
class ChunkingStats:
    strategy: str
    chunk_count: int
    average_characters: float
    minimum_characters: int
    maximum_characters: int


def chunk_fixed_size(
    documents: list[Document],
    *,
    chunk_size: int = 1200,
    overlap: int = 150,
) -> list[Chunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")
    if overlap < 0:
        raise ValueError("overlap must be zero or greater.")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    chunks: list[Chunk] = []
    for document in documents:
        text = document.text.strip()
        if not text:
            continue

        start = 0
        index = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(document.source, "fixed", index),
                        source=document.source,
                        title=document.title,
                        section=f"characters {start}-{end}",
                        text=chunk_text,
                    )
                )
                index += 1
            if end == len(text):
                break
            start = end - overlap
    return chunks


def chunk_by_structure(
    documents: list[Document],
    *,
    max_chunk_size: int = 1800,
) -> list[Chunk]:
    if max_chunk_size <= 0:
        raise ValueError("max_chunk_size must be greater than zero.")

    chunks: list[Chunk] = []
    for document in documents:
        sections = _split_markdown_sections(document)
        if not sections and document.source.endswith(".py"):
            sections = _split_python_sections(document)
        if not sections:
            sections = [(document.title, document.text)]

        index = 0
        for section, text in sections:
            for piece in _split_long_text(text.strip(), max_chunk_size):
                if not piece:
                    continue
                chunks.append(
                    Chunk(
                        chunk_id=_chunk_id(document.source, "structure", index),
                        source=document.source,
                        title=document.title,
                        section=section,
                        text=piece,
                    )
                )
                index += 1
    return chunks


def compare_chunking_strategies(
    documents: list[Document],
    *,
    fixed_chunk_size: int = 1200,
    fixed_overlap: int = 150,
    structure_max_chunk_size: int = 1800,
) -> list[ChunkingStats]:
    fixed_chunks = chunk_fixed_size(
        documents,
        chunk_size=fixed_chunk_size,
        overlap=fixed_overlap,
    )
    structure_chunks = chunk_by_structure(
        documents,
        max_chunk_size=structure_max_chunk_size,
    )
    return [
        describe_chunks("fixed", fixed_chunks),
        describe_chunks("structure", structure_chunks),
    ]


def describe_chunks(strategy: str, chunks: list[Chunk]) -> ChunkingStats:
    sizes = [len(chunk.text) for chunk in chunks]
    if not sizes:
        return ChunkingStats(
            strategy=strategy,
            chunk_count=0,
            average_characters=0.0,
            minimum_characters=0,
            maximum_characters=0,
        )
    return ChunkingStats(
        strategy=strategy,
        chunk_count=len(chunks),
        average_characters=sum(sizes) / len(sizes),
        minimum_characters=min(sizes),
        maximum_characters=max(sizes),
    )


def _split_markdown_sections(document: Document) -> list[tuple[str, str]]:
    if not document.source.lower().endswith((".md", ".markdown", ".rst")):
        return []

    sections: list[tuple[str, list[str]]] = []
    current_title = document.title
    current_lines: list[str] = []
    for line in document.text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            if current_lines:
                sections.append((current_title, current_lines))
            current_title = stripped.lstrip("#").strip() or document.title
            current_lines = [line]
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines)) for title, lines in sections]


def _split_python_sections(document: Document) -> list[tuple[str, str]]:
    sections: list[tuple[str, list[str]]] = []
    current_title = document.title
    current_lines: list[str] = []
    for line in document.text.splitlines():
        stripped = line.lstrip()
        is_top_level_definition = (
            line == stripped
            and (stripped.startswith("def ") or stripped.startswith("class "))
        )
        if is_top_level_definition and current_lines:
            sections.append((current_title, current_lines))
            current_title = stripped.split(":", 1)[0]
            current_lines = [line]
            continue
        if is_top_level_definition:
            current_title = stripped.split(":", 1)[0]
        current_lines.append(line)
    if current_lines:
        sections.append((current_title, current_lines))

    return [(title, "\n".join(lines)) for title, lines in sections]


def _split_long_text(text: str, max_chunk_size: int) -> list[str]:
    if len(text) <= max_chunk_size:
        return [text]

    pieces: list[str] = []
    paragraphs = text.split("\n\n")
    current = ""
    for paragraph in paragraphs:
        next_text = paragraph if not current else f"{current}\n\n{paragraph}"
        if len(next_text) <= max_chunk_size:
            current = next_text
            continue
        if current:
            pieces.append(current)
        if len(paragraph) <= max_chunk_size:
            current = paragraph
        else:
            pieces.extend(
                paragraph[start : start + max_chunk_size]
                for start in range(0, len(paragraph), max_chunk_size)
            )
            current = ""
    if current:
        pieces.append(current)
    return [piece.strip() for piece in pieces if piece.strip()]


def _chunk_id(source: str, strategy: str, index: int) -> str:
    normalized = source.replace("/", "-").replace("\\", "-").strip("-")
    return f"{normalized}:{strategy}:{index}"
