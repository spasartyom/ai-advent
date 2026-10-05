import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ai_advent.chunking import Chunk, ChunkingStats


@dataclass(frozen=True)
class IndexedChunk:
    chunk: Chunk
    embedding: list[float]


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class VectorIndex:
    embedding_model: str
    chunking_strategy: str
    chunks: list[IndexedChunk]
    comparison: list[ChunkingStats]

    def search(self, query_embedding: list[float], *, top_k: int = 5) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        results = [
            SearchResult(
                chunk=indexed_chunk.chunk,
                score=cosine_similarity(query_embedding, indexed_chunk.embedding),
            )
            for indexed_chunk in self.chunks
        ]
        return sorted(results, key=lambda result: result.score, reverse=True)[:top_k]

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "embedding_model": self.embedding_model,
            "chunking_strategy": self.chunking_strategy,
            "comparison": [asdict(item) for item in self.comparison],
            "chunks": [
                {
                    "chunk_id": indexed_chunk.chunk.chunk_id,
                    "source": indexed_chunk.chunk.source,
                    "title": indexed_chunk.chunk.title,
                    "section": indexed_chunk.chunk.section,
                    "text": indexed_chunk.chunk.text,
                    "embedding": indexed_chunk.embedding,
                }
                for indexed_chunk in self.chunks
            ],
        }
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "VectorIndex":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VectorIndex":
        chunks = [
            IndexedChunk(
                chunk=Chunk(
                    chunk_id=str(item["chunk_id"]),
                    source=str(item["source"]),
                    title=str(item["title"]),
                    section=str(item["section"]),
                    text=str(item["text"]),
                ),
                embedding=[float(value) for value in item["embedding"]],
            )
            for item in payload.get("chunks", [])
        ]
        comparison = [
            ChunkingStats(
                strategy=str(item["strategy"]),
                chunk_count=int(item["chunk_count"]),
                average_characters=float(item["average_characters"]),
                minimum_characters=int(item["minimum_characters"]),
                maximum_characters=int(item["maximum_characters"]),
            )
            for item in payload.get("comparison", [])
        ]
        return cls(
            embedding_model=str(payload.get("embedding_model", "")),
            chunking_strategy=str(payload.get("chunking_strategy", "")),
            chunks=chunks,
            comparison=comparison,
        )


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("Embeddings must have the same dimensions.")
    left_length = math.sqrt(sum(value * value for value in left))
    right_length = math.sqrt(sum(value * value for value in right))
    if left_length == 0 or right_length == 0:
        return 0.0
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot_product / (left_length * right_length)
