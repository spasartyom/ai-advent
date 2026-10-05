from dataclasses import dataclass

from ai_advent.chat import ChatCompletionsAPI, ChatResponse, create_chat_completion
from ai_advent.embeddings import EmbeddingsAPI, create_embeddings
from ai_advent.vector_index import SearchResult, VectorIndex


RAG_SYSTEM_PROMPT = (
    "Ты Study Coach Agent с RAG-контекстом. "
    "Отвечай по-русски, кратко и по делу. "
    "Используй найденные фрагменты как основную опору. "
    "Если контекст не содержит ответа, явно скажи, что в найденных фрагментах нет достаточной информации."
)

NO_RAG_SYSTEM_PROMPT = (
    "Ты Study Coach Agent. "
    "Отвечай по-русски, кратко и по делу, используя только общие знания и вопрос пользователя."
)


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    retrieved: list[SearchResult]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class RagComparison:
    question: str
    without_rag: ChatResponse
    with_rag: RagAnswer


class RagResponder:
    def __init__(
        self,
        *,
        index: VectorIndex,
        embeddings_api: EmbeddingsAPI,
        embedding_model: str,
        chat_completions_api: ChatCompletionsAPI,
        chat_model: str,
        top_k: int = 5,
    ) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")
        self._index = index
        self._embeddings_api = embeddings_api
        self._embedding_model = embedding_model
        self._chat_completions_api = chat_completions_api
        self._chat_model = chat_model
        self._top_k = top_k

    def answer(self, question: str) -> RagAnswer:
        if not question.strip():
            raise ValueError("question cannot be empty.")

        query_embedding = create_embeddings(
            self._embeddings_api,
            self._embedding_model,
            [question],
        ).embeddings[0]
        retrieved = self._index.search(query_embedding, top_k=self._top_k)
        response = create_chat_completion(
            self._chat_completions_api,
            self._chat_model,
            [
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_rag_prompt(question, retrieved),
                },
            ],
        )
        return RagAnswer(
            question=question,
            answer=response.text,
            retrieved=retrieved,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )

    def compare(self, question: str) -> RagComparison:
        if not question.strip():
            raise ValueError("question cannot be empty.")

        without_rag = create_chat_completion(
            self._chat_completions_api,
            self._chat_model,
            [
                {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
        )
        with_rag = self.answer(question)
        return RagComparison(
            question=question,
            without_rag=without_rag,
            with_rag=with_rag,
        )


def build_rag_prompt(question: str, retrieved: list[SearchResult]) -> str:
    context = "\n\n".join(
        _format_retrieved_chunk(index, result)
        for index, result in enumerate(retrieved, start=1)
    )
    if not context:
        context = "(no retrieved context)"

    return (
        "Ответь на вопрос пользователя, используя найденный контекст.\n"
        "После ответа кратко перечисли использованные источники в формате "
        "`source | section | chunk_id`.\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Найденный контекст:\n{context}"
    )


def _format_retrieved_chunk(index: int, result: SearchResult) -> str:
    chunk = result.chunk
    return (
        f"[{index}] score={result.score:.4f}\n"
        f"source: {chunk.source}\n"
        f"title: {chunk.title}\n"
        f"section: {chunk.section}\n"
        f"chunk_id: {chunk.chunk_id}\n"
        f"text:\n{chunk.text}"
    )
