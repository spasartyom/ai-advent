import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from ai_advent.chunking import chunk_by_structure, chunk_fixed_size
from ai_advent.documents import Document, load_documents
from ai_advent.embeddings import HashEmbeddingAPI, OllamaEmbeddingAPI
from ai_advent.indexing import build_document_index
from ai_advent.vector_index import VectorIndex


class IndexingTests(unittest.TestCase):
    def test_load_documents_reads_supported_files_recursively(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "README.md").write_text("# Title\n\nContent", encoding="utf-8")
            package = root / "package"
            package.mkdir()
            (package / "module.py").write_text("def run():\n    pass\n", encoding="utf-8")
            (package / "data.bin").write_text("ignored", encoding="utf-8")

            documents = load_documents([root])

        self.assertEqual([document.title for document in documents], ["README.md", "module.py"])

    def test_fixed_chunking_uses_overlap_and_metadata(self) -> None:
        document = Document(
            source="README.md",
            title="README.md",
            text="abcdefghijklmnopqrstuvwxyz",
        )

        chunks = chunk_fixed_size([document], chunk_size=10, overlap=2)

        self.assertEqual([chunk.text for chunk in chunks], ["abcdefghij", "ijklmnopqr", "qrstuvwxyz"])
        self.assertEqual(chunks[0].source, "README.md")
        self.assertEqual(chunks[0].section, "characters 0-10")

    def test_structure_chunking_splits_markdown_headings(self) -> None:
        document = Document(
            source="README.md",
            title="README.md",
            text="# Intro\n\nHello\n\n## Usage\n\nRun it",
        )

        chunks = chunk_by_structure([document])

        self.assertEqual([chunk.section for chunk in chunks], ["Intro", "Usage"])

    def test_build_document_index_saves_json_with_embeddings_and_comparison(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "README.md"
            output = root / "index.json"
            source.write_text(
                "# Intro\n\nAI agents use memory.\n\n## CLI\n\nRun ai-advent agent.",
                encoding="utf-8",
            )

            result = build_document_index(
                [str(source)],
                embeddings_api=HashEmbeddingAPI(dimensions=8),
                embedding_model="hash-test",
                output_path=str(output),
                strategy="structure",
                structure_max_chunk_size=80,
            )

            payload = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(result.document_count, 1)
        self.assertEqual(payload["embedding_model"], "hash-test")
        self.assertEqual(payload["chunking_strategy"], "structure")
        self.assertEqual(len(payload["comparison"]), 2)
        self.assertGreaterEqual(len(payload["chunks"]), 1)
        self.assertIn("source", payload["chunks"][0])
        self.assertIn("section", payload["chunks"][0])
        self.assertEqual(len(payload["chunks"][0]["embedding"]), 8)

    def test_vector_index_can_search_loaded_index(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "notes.txt"
            output = root / "index.json"
            source.write_text("alpha beta gamma\n\nscheduler reminder task", encoding="utf-8")
            build_document_index(
                [str(source)],
                embeddings_api=HashEmbeddingAPI(dimensions=8),
                embedding_model="hash-test",
                output_path=str(output),
                strategy="fixed",
                fixed_chunk_size=20,
                fixed_overlap=0,
            )

            index = VectorIndex.load(output)
            query_embedding = HashEmbeddingAPI(dimensions=8).create(
                model="hash-test",
                input=["scheduler"],
            )["data"][0]["embedding"]
            results = index.search(query_embedding, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertGreaterEqual(results[0].score, -1.0)

    def test_ollama_embedding_api_uses_batch_embed_endpoint(self) -> None:
        class FakeResponse:
            def __enter__(self) -> "FakeResponse":
                return self

            def __exit__(self, *args: object) -> None:
                return None

            def read(self) -> bytes:
                return json.dumps(
                    {
                        "embeddings": [
                            [0.1, 0.2],
                            [0.3, 0.4],
                        ]
                    }
                ).encode("utf-8")

        captured: dict[str, object] = {}

        def fake_urlopen(http_request: object, timeout: int) -> FakeResponse:
            captured["url"] = http_request.full_url
            captured["data"] = json.loads(http_request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse()

        with patch("ai_advent.embeddings.request.urlopen", fake_urlopen):
            response = OllamaEmbeddingAPI("localhost:11434").create(
                model="nomic-embed-text",
                input=["first", "second"],
            )

        self.assertEqual(captured["url"], "http://localhost:11434/api/embed")
        self.assertEqual(
            captured["data"],
            {
                "model": "nomic-embed-text",
                "input": ["first", "second"],
            },
        )
        self.assertEqual(captured["timeout"], 120)
        self.assertEqual(
            response,
            {
                "data": [
                    {"embedding": [0.1, 0.2], "index": 0},
                    {"embedding": [0.3, 0.4], "index": 1},
                ]
            },
        )


if __name__ == "__main__":
    unittest.main()
