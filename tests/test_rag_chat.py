import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from ai_advent.rag import GroundedRagAnswer
from ai_advent.rag_chat import (
    JsonRagChatMemory,
    RagChatSession,
    RagChatState,
    RagTaskMemory,
    format_rag_chat_context,
    format_rag_chat_scenarios,
)


class FakeResponder:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def answer_with_citations(self, question: str, **kwargs: object) -> GroundedRagAnswer:
        self.calls.append({"question": question, **kwargs})
        return GroundedRagAnswer(
            question=question,
            answer=f"grounded answer to {question}",
            retrieved=[],
            citations=[],
            search_query=question,
            candidate_count=0,
        )


class RagChatTests(unittest.TestCase):
    def test_rag_chat_session_tracks_task_memory_and_passes_context(self) -> None:
        responder = FakeResponder()
        session = RagChatSession(responder, keep_last=4)

        session.set_goal("understand RAG")
        session.add_clarification("focus on citations")
        session.add_constraint("answer briefly")
        session.set_term("RAG", "retrieval augmented generation")
        answer = session.ask("What is the goal?")

        self.assertEqual(answer.answer, "grounded answer to What is the goal?")
        call = responder.calls[0]
        self.assertIn("goal: understand RAG", call["conversation_context"])
        self.assertIn("focus on citations", call["conversation_context"])
        self.assertIn("answer briefly", call["conversation_context"])
        self.assertIn("RAG=retrieval augmented generation", call["conversation_context"])
        self.assertEqual(len(session.state.messages), 2)

    def test_json_rag_chat_memory_round_trips_state(self) -> None:
        with TemporaryDirectory() as directory:
            memory = JsonRagChatMemory(Path(directory) / "rag-chat.json")
            state = RagChatState(
                messages=(
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi"},
                ),
                task_memory=RagTaskMemory(
                    goal="demo",
                    clarifications=("one",),
                    constraints=("cite sources",),
                    terms={"chunk": "retrieved text"},
                ),
            )

            memory.save(state)
            loaded = memory.load()

        self.assertEqual(loaded, state)

    def test_format_rag_chat_context_keeps_recent_messages(self) -> None:
        state = RagChatState(
            messages=(
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
                {"role": "user", "content": "three"},
            ),
            task_memory=RagTaskMemory(goal="demo"),
        )

        context = format_rag_chat_context(state, keep_last=2)

        self.assertIn("goal: demo", context)
        self.assertNotIn("user: one", context)
        self.assertIn("assistant: two", context)
        self.assertIn("user: three", context)

    def test_format_rag_chat_scenarios_lists_two_long_scenarios(self) -> None:
        lines = format_rag_chat_scenarios()
        scenario_headers = [line for line in lines if line.startswith("- ")]

        self.assertEqual(len(scenario_headers), 2)
        self.assertTrue(any("task-state-study-session" in line for line in lines))
        self.assertTrue(any("rag-indexing-study-session" in line for line in lines))


if __name__ == "__main__":
    unittest.main()
