import unittest
from types import SimpleNamespace

from ai_advent.chunking import Chunk
from ai_advent.rag import RagResponder, build_rag_prompt
from ai_advent.vector_index import IndexedChunk, SearchResult, VectorIndex


class FakeEmbeddingsAPI:
    def create(self, **kwargs: object) -> object:
        inputs = kwargs["input"]
        assert isinstance(inputs, list)
        return {
            "data": [
                {
                    "embedding": _embedding_for_text(str(text)),
                    "index": index,
                }
                for index, text in enumerate(inputs)
            ]
        }


class FakeChatCompletionsAPI:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"answer-{len(self.requests)}"),
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=11,
                completion_tokens=12,
                total_tokens=23,
            ),
        )


class RagTests(unittest.TestCase):
    def test_rag_answer_retrieves_context_and_calls_chat_model(self) -> None:
        chat_api = FakeChatCompletionsAPI()
        responder = RagResponder(
            index=_test_index(),
            embeddings_api=FakeEmbeddingsAPI(),
            embedding_model="fake-embedding",
            chat_completions_api=chat_api,
            chat_model="fake-chat",
            top_k=1,
        )

        answer = responder.answer("How does task state work?")

        self.assertEqual(answer.answer, "answer-1")
        self.assertEqual(answer.prompt_tokens, 11)
        self.assertEqual(len(answer.retrieved), 1)
        self.assertEqual(answer.retrieved[0].chunk.source, "task.py")
        self.assertEqual(chat_api.requests[0]["model"], "fake-chat")
        messages = chat_api.requests[0]["messages"]
        self.assertIn("task state stores stage", messages[1]["content"])
        self.assertIn("chunk-task", messages[1]["content"])

    def test_rag_compare_calls_without_rag_and_with_rag(self) -> None:
        chat_api = FakeChatCompletionsAPI()
        responder = RagResponder(
            index=_test_index(),
            embeddings_api=FakeEmbeddingsAPI(),
            embedding_model="fake-embedding",
            chat_completions_api=chat_api,
            chat_model="fake-chat",
            top_k=1,
        )

        comparison = responder.compare("Which commands manage memory?")

        self.assertEqual(comparison.without_rag.text, "answer-1")
        self.assertEqual(comparison.with_rag.answer, "answer-2")
        self.assertEqual(len(chat_api.requests), 2)
        self.assertEqual(chat_api.requests[0]["messages"][1]["content"], "Which commands manage memory?")
        self.assertIn("Найденный контекст", chat_api.requests[1]["messages"][1]["content"])

    def test_build_rag_prompt_formats_sources(self) -> None:
        chunk = Chunk(
            chunk_id="chunk-1",
            source="README.md",
            title="README.md",
            section="Day 22",
            text="RAG uses retrieved chunks.",
        )

        prompt = build_rag_prompt(
            "What is RAG?",
            [SearchResult(chunk=chunk, score=0.75)],
        )

        self.assertIn("What is RAG?", prompt)
        self.assertIn("source: README.md", prompt)
        self.assertIn("section: Day 22", prompt)
        self.assertIn("chunk_id: chunk-1", prompt)
        self.assertIn("RAG uses retrieved chunks.", prompt)

    def test_rag_rejects_empty_question(self) -> None:
        responder = RagResponder(
            index=_test_index(),
            embeddings_api=FakeEmbeddingsAPI(),
            embedding_model="fake-embedding",
            chat_completions_api=FakeChatCompletionsAPI(),
            chat_model="fake-chat",
        )

        with self.assertRaises(ValueError):
            responder.answer(" ")


def _test_index() -> VectorIndex:
    task_chunk = Chunk(
        chunk_id="chunk-task",
        source="task.py",
        title="task.py",
        section="TaskState",
        text="task state stores stage and expected action",
    )
    memory_chunk = Chunk(
        chunk_id="chunk-memory",
        source="memory.py",
        title="memory.py",
        section="Memory",
        text="memory stores messages and long term facts",
    )
    return VectorIndex(
        embedding_model="fake-embedding",
        chunking_strategy="structure",
        chunks=[
            IndexedChunk(
                chunk=task_chunk,
                embedding=_embedding_for_text(task_chunk.text),
            ),
            IndexedChunk(
                chunk=memory_chunk,
                embedding=_embedding_for_text(memory_chunk.text),
            ),
        ],
        comparison=[],
    )


def _embedding_for_text(text: str) -> list[float]:
    lower = text.lower()
    return [
        1.0 if "task" in lower else 0.0,
        1.0 if "memory" in lower else 0.0,
        1.0 if "command" in lower else 0.0,
    ]


if __name__ == "__main__":
    unittest.main()
