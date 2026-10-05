from dataclasses import dataclass

from ai_advent.chunking import (
    Chunk,
    ChunkingStats,
    chunk_by_structure,
    chunk_fixed_size,
    compare_chunking_strategies,
)
from ai_advent.documents import Document, load_documents
from ai_advent.embeddings import EmbeddingsAPI, create_embeddings
from ai_advent.vector_index import IndexedChunk, VectorIndex


@dataclass(frozen=True)
class IndexBuildResult:
    index: VectorIndex
    document_count: int
    total_characters: int
    saved_path: str


def build_document_index(
    paths: list[str],
    *,
    embeddings_api: EmbeddingsAPI,
    embedding_model: str,
    output_path: str,
    strategy: str = "structure",
    fixed_chunk_size: int = 1200,
    fixed_overlap: int = 150,
    structure_max_chunk_size: int = 1800,
) -> IndexBuildResult:
    documents = load_documents(paths)
    if not documents:
        raise ValueError("No supported text documents found.")

    comparison = compare_chunking_strategies(
        documents,
        fixed_chunk_size=fixed_chunk_size,
        fixed_overlap=fixed_overlap,
        structure_max_chunk_size=structure_max_chunk_size,
    )
    chunks = _chunk_documents(
        documents,
        strategy=strategy,
        fixed_chunk_size=fixed_chunk_size,
        fixed_overlap=fixed_overlap,
        structure_max_chunk_size=structure_max_chunk_size,
    )
    if not chunks:
        raise ValueError("No chunks were produced from the selected documents.")

    embeddings = create_embeddings(
        embeddings_api,
        embedding_model,
        [chunk.text for chunk in chunks],
    ).embeddings
    if len(embeddings) != len(chunks):
        raise ValueError("Embedding provider returned a different number of vectors.")

    index = VectorIndex(
        embedding_model=embedding_model,
        chunking_strategy=strategy,
        chunks=[
            IndexedChunk(chunk=chunk, embedding=embedding)
            for chunk, embedding in zip(chunks, embeddings)
        ],
        comparison=comparison,
    )
    index.save(output_path)
    return IndexBuildResult(
        index=index,
        document_count=len(documents),
        total_characters=sum(len(document.text) for document in documents),
        saved_path=output_path,
    )


def _chunk_documents(
    documents: list[Document],
    *,
    strategy: str,
    fixed_chunk_size: int,
    fixed_overlap: int,
    structure_max_chunk_size: int,
) -> list[Chunk]:
    if strategy == "fixed":
        return chunk_fixed_size(
            documents,
            chunk_size=fixed_chunk_size,
            overlap=fixed_overlap,
        )
    if strategy == "structure":
        return chunk_by_structure(
            documents,
            max_chunk_size=structure_max_chunk_size,
        )
    raise ValueError("strategy must be fixed or structure.")


def format_chunking_comparison(comparison: list[ChunkingStats]) -> list[str]:
    lines = ["Chunking comparison:"]
    for stats in comparison:
        lines.append(
            "  "
            f"{stats.strategy}: "
            f"{stats.chunk_count} chunks, "
            f"avg {stats.average_characters:.1f} chars, "
            f"min {stats.minimum_characters}, "
            f"max {stats.maximum_characters}"
        )
    return lines
