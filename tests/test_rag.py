import unittest
from types import SimpleNamespace

from ai_advent.chunking import Chunk
from ai_advent.rag import (
    RagResponder,
    build_citations,
    build_grounded_rag_prompt,
    build_rag_prompt,
    filter_and_rerank_results,
)
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


class RewriteAwareFakeChatCompletionsAPI(FakeChatCompletionsAPI):
    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        system_message = messages[0]["content"]
        if "переписываешь вопросы" in system_message:
            content = "memory command"
        else:
            content = f"answer-{len(self.requests)}"
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
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

    def test_filter_and_rerank_results_applies_threshold_and_keyword_overlap(self) -> None:
        weak = SearchResult(
            chunk=Chunk(
                chunk_id="weak",
                source="weak.md",
                title="weak.md",
                section="Weak",
                text="unrelated topic",
            ),
            score=0.1,
        )
        lexical_match = SearchResult(
            chunk=Chunk(
                chunk_id="match",
                source="match.md",
                title="match.md",
                section="Match",
                text="memory command details",
            ),
            score=0.5,
        )
        semantic_only = SearchResult(
            chunk=Chunk(
                chunk_id="semantic",
                source="semantic.md",
                title="semantic.md",
                section="Semantic",
                text="general details",
            ),
            score=0.5,
        )

        results = filter_and_rerank_results(
            "memory command",
            [weak, semantic_only, lexical_match],
            top_k=2,
            min_score=0.2,
        )

        self.assertEqual([result.chunk.chunk_id for result in results], ["match", "semantic"])

    def test_compare_retrieval_rewrites_query_and_reports_improved_candidates(self) -> None:
        chat_api = RewriteAwareFakeChatCompletionsAPI()
        responder = RagResponder(
            index=_test_index(),
            embeddings_api=FakeEmbeddingsAPI(),
            embedding_model="fake-embedding",
            chat_completions_api=chat_api,
            chat_model="fake-chat",
            top_k=1,
        )

        comparison = responder.compare_retrieval(
            "Which commands manage memory?",
            candidate_k=2,
            min_score=0.1,
        )

        self.assertEqual(comparison.baseline.search_query, "Which commands manage memory?")
        self.assertEqual(comparison.improved.search_query, "memory command")
        self.assertEqual(comparison.improved.candidate_count, 2)
        self.assertEqual(comparison.improved.retrieved[0].chunk.chunk_id, "chunk-memory")
        self.assertEqual(len(chat_api.requests), 3)

    def test_grounded_answer_includes_citations_and_grounded_prompt(self) -> None:
        chat_api = FakeChatCompletionsAPI()
        responder = RagResponder(
            index=_test_index(),
            embeddings_api=FakeEmbeddingsAPI(),
            embedding_model="fake-embedding",
            chat_completions_api=chat_api,
            chat_model="fake-chat",
            top_k=1,
        )

        answer = responder.answer_with_citations(
            "How does task state work?",
            min_score=0.1,
        )

        self.assertEqual(answer.answer, "answer-1")
        self.assertFalse(answer.refused_for_low_relevance)
        self.assertEqual(len(answer.citations), 1)
        self.assertEqual(answer.citations[0].source, "task.py")
        self.assertEqual(answer.citations[0].chunk_id, "chunk-task")
        self.assertIn("task state stores stage", answer.citations[0].quote)
        messages = chat_api.requests[0]["messages"]
        self.assertIn("Источники", messages[0]["content"])
        self.assertIn("Разрешенные источники и цитаты", messages[1]["content"])

    def test_grounded_answer_refuses_low_relevance_without_calling_model(self) -> None:
        chat_api = FakeChatCompletionsAPI()
        responder = RagResponder(
            index=_test_index(),
            embeddings_api=FakeEmbeddingsAPI(),
            embedding_model="fake-embedding",
            chat_completions_api=chat_api,
            chat_model="fake-chat",
            top_k=1,
        )

        answer = responder.answer_with_citations(
            "How does task state work?",
            min_score=1.1,
        )

        self.assertTrue(answer.refused_for_low_relevance)
        self.assertIn("Не знаю", answer.answer)
        self.assertEqual(answer.citations, [])
        self.assertEqual(chat_api.requests, [])

    def test_build_citations_and_grounded_prompt_use_chunk_metadata(self) -> None:
        chunk = Chunk(
            chunk_id="chunk-1",
            source="README.md",
            title="README.md",
            section="Day 24",
            text="RAG answers must include sources and quotes from retrieved chunks.",
        )
        result = SearchResult(chunk=chunk, score=0.9)

        citations = build_citations([result])
        prompt = build_grounded_rag_prompt("What must RAG include?", [result], citations)

        self.assertEqual(citations[0].source, "README.md")
        self.assertEqual(citations[0].section, "Day 24")
        self.assertEqual(citations[0].chunk_id, "chunk-1")
        self.assertIn("sources and quotes", citations[0].quote)
        self.assertIn("README.md | Day 24 | chunk-1", prompt)

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
