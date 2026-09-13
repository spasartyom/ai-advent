import unittest
from types import SimpleNamespace

from ai_advent.agent import Agent, DEFAULT_SYSTEM_PROMPT


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


if __name__ == "__main__":
    unittest.main()
