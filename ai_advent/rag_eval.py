from dataclasses import dataclass


@dataclass(frozen=True)
class ControlQuestion:
    question: str
    expected: str
    expected_sources: list[str]


CONTROL_QUESTIONS = [
    ControlQuestion(
        question="Какие команды CLI управляют task state?",
        expected="Ответ должен перечислить команды /task status, start, stage, approve, step, expect, title, pause, resume и done.",
        expected_sources=["README.md", "AGENTS.md"],
    ),
    ControlQuestion(
        question="Какие состояния задачи разрешены в Study Coach Agent?",
        expected="Ответ должен назвать idle, planning, execution, validation и done.",
        expected_sources=["ai_advent/task.py", "README.md", "AGENTS.md"],
    ),
    ControlQuestion(
        question="Как работает /lesson TOPIC?",
        expected="Ответ должен объяснить вызов MCP get_lesson, печать результата и передачу материала в Agent.run_turn.",
        expected_sources=["README.md", "AGENTS.md", "ai_advent/cli.py"],
    ),
    ControlQuestion(
        question="Что сохраняет JsonFileMemory?",
        expected="Ответ должен упомянуть messages, summary, facts, memory layers, profile, task_state, invariants, branches и checkpoints.",
        expected_sources=["ai_advent/memory.py", "AGENTS.md"],
    ),
    ControlQuestion(
        question="Какие стратегии context management есть в проекте?",
        expected="Ответ должен назвать full, summary, sliding-window, facts и branch.",
        expected_sources=["ai_advent/context.py", "README.md", "AGENTS.md"],
    ),
    ControlQuestion(
        question="Что делает pipeline Дня 19?",
        expected="Ответ должен описать цепочку search_lessons -> summarize_note -> save_note.",
        expected_sources=["ai_advent/pipeline.py", "README.md", "AGENTS.md"],
    ),
    ControlQuestion(
        question="Какие MCP-серверы участвуют в orchestration Дня 20?",
        expected="Ответ должен назвать lessons, notes и scheduler servers.",
        expected_sources=["ai_advent/orchestration.py", "README.md", "AGENTS.md"],
    ),
    ControlQuestion(
        question="Какие две стратегии chunking реализованы для индексации документов?",
        expected="Ответ должен сравнить fixed-size и structure-aware chunking.",
        expected_sources=["ai_advent/chunking.py", "README.md"],
    ),
    ControlQuestion(
        question="Какие метаданные сохраняются у каждого чанка?",
        expected="Ответ должен назвать source, title, section и chunk_id.",
        expected_sources=["ai_advent/vector_index.py", "README.md"],
    ),
    ControlQuestion(
        question="Как собрать индекс через Ollama embeddings?",
        expected="Ответ должен привести --embedding-provider ollama, модель nomic-embed-text и при необходимости --ollama-url.",
        expected_sources=["README.md", "ai_advent/cli.py", "ai_advent/embeddings.py"],
    ),
]


def format_control_questions() -> list[str]:
    lines = ["RAG control questions:"]
    for index, item in enumerate(CONTROL_QUESTIONS, start=1):
        lines.append(f"{index}. {item.question}")
        lines.append(f"   expected: {item.expected}")
        lines.append(f"   expected_sources: {', '.join(item.expected_sources)}")
    return lines
