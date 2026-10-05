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

GROUNDING_SYSTEM_PROMPT = (
    "Ты Study Coach Agent с RAG-контекстом и строгими анти-галлюцинационными правилами. "
    "Отвечай только на основе найденных фрагментов. "
    "Верни ответ в трех блоках: `Ответ`, `Источники`, `Цитаты`. "
    "В источниках используй только source, section и chunk_id из контекста. "
    "В цитатах используй только короткие фрагменты из контекста."
)


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    retrieved: list[SearchResult]
    search_query: str = ""
    candidate_count: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


@dataclass(frozen=True)
class RagComparison:
    question: str
    without_rag: ChatResponse
    with_rag: RagAnswer


@dataclass(frozen=True)
class RagRetrievalComparison:
    question: str
    baseline: RagAnswer
    improved: RagAnswer


@dataclass(frozen=True)
class RagCitation:
    source: str
    section: str
    chunk_id: str
    quote: str


@dataclass(frozen=True)
class GroundedRagAnswer:
    question: str
    answer: str
    retrieved: list[SearchResult]
    citations: list[RagCitation]
    search_query: str = ""
    candidate_count: int = 0
    refused_for_low_relevance: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


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

    def answer(
        self,
        question: str,
        *,
        candidate_k: int | None = None,
        min_score: float | None = None,
        rewrite_query: bool = False,
    ) -> RagAnswer:
        if not question.strip():
            raise ValueError("question cannot be empty.")
        active_candidate_k = candidate_k or self._top_k
        if active_candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        search_query = self.rewrite_query(question) if rewrite_query else question
        query_embedding = create_embeddings(
            self._embeddings_api,
            self._embedding_model,
            [search_query],
        ).embeddings[0]
        candidates = self._index.search(query_embedding, top_k=active_candidate_k)
        retrieved = filter_and_rerank_results(
            question,
            candidates,
            top_k=self._top_k,
            min_score=min_score,
        )
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
            search_query=search_query,
            candidate_count=len(candidates),
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

    def compare_retrieval(
        self,
        question: str,
        *,
        candidate_k: int,
        min_score: float,
    ) -> RagRetrievalComparison:
        baseline = self.answer(question)
        improved = self.answer(
            question,
            candidate_k=candidate_k,
            min_score=min_score,
            rewrite_query=True,
        )
        return RagRetrievalComparison(
            question=question,
            baseline=baseline,
            improved=improved,
        )

    def rewrite_query(self, question: str) -> str:
        response = create_chat_completion(
            self._chat_completions_api,
            self._chat_model,
            [
                {
                    "role": "system",
                    "content": (
                        "Ты переписываешь вопросы для поиска по локальному индексу. "
                        "Верни только короткий поисковый запрос без пояснений."
                    ),
                },
                {"role": "user", "content": question},
            ],
        )
        rewritten = response.text.strip()
        return rewritten or question

    def answer_with_citations(
        self,
        question: str,
        *,
        candidate_k: int | None = None,
        min_score: float = 0.2,
        rewrite_query: bool = False,
    ) -> GroundedRagAnswer:
        if not question.strip():
            raise ValueError("question cannot be empty.")
        active_candidate_k = candidate_k or self._top_k
        if active_candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        search_query = self.rewrite_query(question) if rewrite_query else question
        query_embedding = create_embeddings(
            self._embeddings_api,
            self._embedding_model,
            [search_query],
        ).embeddings[0]
        candidates = self._index.search(query_embedding, top_k=active_candidate_k)
        retrieved = filter_and_rerank_results(
            question,
            candidates,
            top_k=self._top_k,
            min_score=min_score,
        )
        citations = build_citations(retrieved)
        if not retrieved:
            return GroundedRagAnswer(
                question=question,
                answer=(
                    "Не знаю: в локальном индексе не нашлось достаточно "
                    "релевантного контекста. Уточните вопрос или соберите индекс "
                    "по более подходящим документам."
                ),
                retrieved=[],
                citations=[],
                search_query=search_query,
                candidate_count=len(candidates),
                refused_for_low_relevance=True,
            )

        response = create_chat_completion(
            self._chat_completions_api,
            self._chat_model,
            [
                {"role": "system", "content": GROUNDING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_grounded_rag_prompt(question, retrieved, citations),
                },
            ],
        )
        return GroundedRagAnswer(
            question=question,
            answer=response.text,
            retrieved=retrieved,
            citations=citations,
            search_query=search_query,
            candidate_count=len(candidates),
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            total_tokens=response.total_tokens,
        )


def filter_and_rerank_results(
    question: str,
    candidates: list[SearchResult],
    *,
    top_k: int,
    min_score: float | None,
) -> list[SearchResult]:
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    filtered = [
        result
        for result in candidates
        if min_score is None or result.score >= min_score
    ]
    return sorted(
        filtered,
        key=lambda result: _rerank_score(question, result),
        reverse=True,
    )[:top_k]


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


def build_grounded_rag_prompt(
    question: str,
    retrieved: list[SearchResult],
    citations: list[RagCitation],
) -> str:
    context = "\n\n".join(
        _format_retrieved_chunk(index, result)
        for index, result in enumerate(retrieved, start=1)
    )
    citation_block = "\n".join(
        f"- {citation.source} | {citation.section} | {citation.chunk_id}: {citation.quote}"
        for citation in citations
    )
    return (
        "Ответь на вопрос пользователя строго по найденному контексту.\n"
        "Обязательно включи три блока: `Ответ`, `Источники`, `Цитаты`.\n"
        "Не добавляй факты, которых нет в контексте.\n\n"
        f"Вопрос:\n{question}\n\n"
        f"Разрешенные источники и цитаты:\n{citation_block}\n\n"
        f"Найденный контекст:\n{context}"
    )


def build_citations(retrieved: list[SearchResult]) -> list[RagCitation]:
    return [
        RagCitation(
            source=result.chunk.source,
            section=result.chunk.section,
            chunk_id=result.chunk.chunk_id,
            quote=_short_quote(result.chunk.text),
        )
        for result in retrieved
    ]


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


def _rerank_score(question: str, result: SearchResult) -> float:
    question_tokens = set(_tokens(question))
    chunk_tokens = set(_tokens(result.chunk.text))
    if not question_tokens:
        return result.score
    overlap = len(question_tokens & chunk_tokens) / len(question_tokens)
    return result.score + overlap * 0.05


def _tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in "".join(
            character if character.isalnum() else " "
            for character in text
        ).split()
    ]


def _short_quote(text: str, max_length: int = 220) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_length:
        return compact
    return compact[: max_length - 3].rstrip() + "..."
