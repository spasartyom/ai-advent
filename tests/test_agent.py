import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from ai_advent.agent import Agent, DEFAULT_SYSTEM_PROMPT
from ai_advent.context import StickyFactsContextStrategy, SummaryContextStrategy
from ai_advent.memory import AgentMemoryState, JsonFileMemory


class FakeChatCompletionsAPI:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        number = len(self.requests)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=f"answer-{number}"),
                ),
            ],
            usage=SimpleNamespace(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
            ),
        )


class FailingChatCompletionsAPI:
    def create(self, **kwargs: object) -> object:
        raise RuntimeError("API failed")


class AgentTests(unittest.TestCase):
    def test_agent_sends_user_message_to_selected_model(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model")

        response = agent.run_turn("Hello")

        self.assertEqual(response.text, "answer-1")
        self.assertEqual(
            api.requests,
            [
                {
                    "model": "test-model",
                    "messages": [
                        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": "Hello"},
                    ],
                },
            ],
        )

    def test_agent_keeps_context_inside_the_entity(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model")

        agent.run_turn("First message")
        agent.run_turn("Follow-up message")

        self.assertEqual(
            api.requests[1],
            {
                "model": "test-model",
                "messages": [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "answer-1"},
                    {"role": "user", "content": "Follow-up message"},
                ],
            },
        )

    def test_agent_exposes_a_copy_of_messages(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model")

        agent.run_turn("Hello")
        messages = agent.messages
        messages.append({"role": "user", "content": "mutated outside"})
        messages[0]["content"] = "changed outside"

        self.assertEqual(len(agent.messages), 2)
        self.assertEqual(agent.messages[0]["content"], "Hello")

    def test_agent_copies_initial_messages(self) -> None:
        api = FakeChatCompletionsAPI()
        initial_messages = [{"role": "user", "content": "Previous"}]
        agent = Agent(api, "test-model", messages=initial_messages)

        initial_messages[0]["content"] = "changed outside"
        agent.run_turn("Next")

        self.assertEqual(
            api.requests[0]["messages"][1],
            {"role": "user", "content": "Previous"},
        )

    def test_agent_does_not_store_system_prompt_in_history(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model")

        agent.run_turn("Hello")

        self.assertEqual(
            agent.messages,
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "answer-1"},
            ],
        )

    def test_agent_allows_custom_system_prompt(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model", system_prompt="Answer in one sentence.")

        agent.run_turn("Hello")

        self.assertEqual(
            api.requests[0]["messages"][0],
            {"role": "system", "content": "Answer in one sentence."},
        )

    def test_agent_can_disable_system_prompt(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model", system_prompt="")

        agent.run_turn("Hello")

        self.assertEqual(
            api.requests[0]["messages"],
            [{"role": "user", "content": "Hello"}],
        )

    def test_agent_returns_usage_metadata(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model")

        response = agent.run_turn("Hello")

        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 20)
        self.assertEqual(response.total_tokens, 30)
        self.assertIsNotNone(response.token_report)
        assert response.token_report is not None
        self.assertEqual(response.token_report.prompt_tokens, 10)
        self.assertEqual(response.token_report.completion_tokens, 20)
        self.assertEqual(response.token_report.total_tokens, 30)

    def test_agent_reports_usage_for_each_successful_turn(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model")

        first_response = agent.run_turn("Hello")
        second_response = agent.run_turn("Again")

        self.assertIsNotNone(first_response.token_report)
        self.assertIsNotNone(second_response.token_report)
        assert first_response.token_report is not None
        assert second_response.token_report is not None
        self.assertEqual(first_response.token_report.total_tokens, 30)
        self.assertEqual(second_response.token_report.total_tokens, 30)

    def test_agent_reports_missing_usage_as_none(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model")

        api.create = lambda **kwargs: SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="answer without usage"),
                ),
            ],
            usage=None,
        )
        response = agent.run_turn("Hello")

        self.assertIsNotNone(response.token_report)
        assert response.token_report is not None
        self.assertIsNone(response.token_report.prompt_tokens)
        self.assertIsNone(response.token_report.completion_tokens)
        self.assertIsNone(response.token_report.total_tokens)

    def test_agent_saves_messages_to_configured_memory(self) -> None:
        api = FakeChatCompletionsAPI()

        with TemporaryDirectory() as directory:
            memory = JsonFileMemory(Path(directory) / "memory.json")
            agent = Agent(api, "test-model", memory=memory)

            agent.run_turn("Remember me")

            self.assertEqual(
                memory.load_messages(),
                [
                    {"role": "user", "content": "Remember me"},
                    {"role": "assistant", "content": "answer-1"},
                ],
            )

    def test_agent_loads_messages_from_configured_memory(self) -> None:
        api = FakeChatCompletionsAPI()

        with TemporaryDirectory() as directory:
            memory = JsonFileMemory(Path(directory) / "memory.json")
            memory.save_messages(
                [
                    {"role": "user", "content": "My name is Anton"},
                    {"role": "assistant", "content": "Got it"},
                ]
            )
            agent = Agent(api, "test-model", memory=memory)

            agent.run_turn("What is my name?")

            self.assertEqual(
                api.requests[0]["messages"],
                [
                    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
                    {"role": "user", "content": "My name is Anton"},
                    {"role": "assistant", "content": "Got it"},
                    {"role": "user", "content": "What is my name?"},
                ],
            )

    def test_agent_loads_summary_from_configured_memory(self) -> None:
        api = FakeChatCompletionsAPI()

        with TemporaryDirectory() as directory:
            memory = JsonFileMemory(Path(directory) / "memory.json")
            memory.save_state(
                state=AgentMemoryState(
                    summary="User name is Anton.",
                    messages=[{"role": "user", "content": "What do you remember?"}],
                )
            )
            agent = Agent(
                api,
                "test-model",
                memory=memory,
                context_strategy=SummaryContextStrategy(keep_last=5),
            )

            agent.run_turn("Continue")

            self.assertIn(
                "User name is Anton.",
                api.requests[0]["messages"][1]["content"],
            )

    def test_agent_compresses_summary_context_after_successful_turn(self) -> None:
        api = FakeChatCompletionsAPI()

        with TemporaryDirectory() as directory:
            memory = JsonFileMemory(Path(directory) / "memory.json")
            agent = Agent(
                api,
                "test-model",
                memory=memory,
                context_strategy=SummaryContextStrategy(keep_last=2),
            )

            agent.run_turn("First")
            agent.run_turn("Second")

            self.assertEqual(agent.summary, "answer-3")
            self.assertEqual(
                agent.messages,
                [
                    {"role": "user", "content": "Second"},
                    {"role": "assistant", "content": "answer-2"},
                ],
            )
            self.assertEqual(memory.load_state().summary, "answer-3")
            self.assertEqual(len(api.requests), 3)

            agent.run_turn("Third")

            self.assertIn("answer-3", api.requests[3]["messages"][1]["content"])

    def test_agent_updates_facts_context_after_successful_turn(self) -> None:
        class FactsApi(FakeChatCompletionsAPI):
            def create(self, **kwargs: object) -> object:
                self.requests.append(kwargs)
                number = len(self.requests)
                if number == 2:
                    content = '{"goal": "build agent"}'
                else:
                    content = f"answer-{number}"
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(content=content),
                        ),
                    ],
                    usage=SimpleNamespace(
                        prompt_tokens=10,
                        completion_tokens=20,
                        total_tokens=30,
                    ),
                )

        api = FactsApi()
        agent = Agent(
            api,
            "test-model",
            context_strategy=StickyFactsContextStrategy(keep_last=2),
        )

        agent.run_turn("My goal is building an agent.")

        self.assertEqual(agent.facts, {"goal": "build agent"})

    def test_agent_saves_explicit_memory_layers(self) -> None:
        api = FakeChatCompletionsAPI()

        with TemporaryDirectory() as directory:
            memory = JsonFileMemory(Path(directory) / "memory.json")
            agent = Agent(api, "test-model", memory=memory)

            agent.remember_working("lesson_topic", "Python decorators")
            agent.remember_long_term("preferred_language", "ru")

            state = memory.load_state()
            self.assertEqual(
                state.working_memory,
                {"lesson_topic": "Python decorators"},
            )
            self.assertEqual(state.long_term_memory, {"preferred_language": "ru"})

    def test_agent_includes_memory_layers_in_request(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(
            api,
            "test-model",
            working_memory={"lesson_topic": "Python decorators"},
            long_term_memory={"preferred_language": "ru"},
        )

        agent.run_turn("Continue lesson")

        messages = api.requests[0]["messages"]
        self.assertEqual(
            messages[0],
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        )
        self.assertIn("Рабочая память", messages[1]["content"])
        self.assertIn("lesson_topic: Python decorators", messages[1]["content"])
        self.assertIn("preferred_language: ru", messages[1]["content"])
        self.assertEqual(
            messages[2],
            {"role": "user", "content": "Continue lesson"},
        )

    def test_agent_creates_and_switches_branches_from_checkpoint(self) -> None:
        api = FakeChatCompletionsAPI()
        agent = Agent(api, "test-model")

        agent.run_turn("Base")
        agent.save_checkpoint("base")
        agent.create_branch("option_a", "base")
        agent.run_turn("A")
        agent.switch_branch("main")
        agent.create_branch("option_b", "base")
        agent.run_turn("B")

        self.assertEqual(agent.current_branch, "option_b")
        self.assertIn("main", agent.list_branches())
        self.assertIn("option_a", agent.list_branches())
        self.assertIn("option_b", agent.list_branches())
        self.assertEqual(
            agent.messages[-2:],
            [
                {"role": "user", "content": "B"},
                {"role": "assistant", "content": "answer-3"},
            ],
        )

        agent.switch_branch("option_a")

        self.assertEqual(
            agent.messages[-2:],
            [
                {"role": "user", "content": "A"},
                {"role": "assistant", "content": "answer-2"},
            ],
        )

    def test_agent_rolls_back_user_message_when_api_fails(self) -> None:
        agent = Agent(FailingChatCompletionsAPI(), "test-model")

        with self.assertRaises(RuntimeError):
            agent.run_turn("This turn will fail")

        self.assertEqual(agent.messages, [])


if __name__ == "__main__":
    unittest.main()
