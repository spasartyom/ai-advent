import unittest
from types import SimpleNamespace

from ai_advent.chat import ChatSession


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
        )


class ChatSessionTests(unittest.TestCase):
    def test_first_message_is_sent_to_selected_model(self) -> None:
        api = FakeChatCompletionsAPI()
        session = ChatSession(api, "test-model")

        answer = session.ask("Hello")

        self.assertEqual(answer, "answer-1")
        self.assertEqual(
            api.requests,
            [
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            ],
        )

    def test_next_message_sends_message_history(self) -> None:
        api = FakeChatCompletionsAPI()
        session = ChatSession(api, "test-model")

        session.ask("First message")
        session.ask("Follow-up message")

        self.assertEqual(
            api.requests[1],
            {
                "model": "test-model",
                "messages": [
                    {"role": "user", "content": "First message"},
                    {"role": "assistant", "content": "answer-1"},
                    {"role": "user", "content": "Follow-up message"},
                ],
            },
        )

    def test_generation_controls_are_sent_to_api(self) -> None:
        api = FakeChatCompletionsAPI()
        session = ChatSession(api, "test-model")

        answer = session.ask(
            "Hello",
            max_completion_tokens=120,
            stop=["###END###"],
        )

        self.assertEqual(answer, "answer-1")
        self.assertEqual(
            api.requests,
            [
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_completion_tokens": 120,
                    "stop": ["###END###"],
                },
            ],
        )

    def test_temperature_is_sent_to_api(self) -> None:
        api = FakeChatCompletionsAPI()
        session = ChatSession(api, "test-model")

        answer = session.ask("Hello", temperature=0.7)

        self.assertEqual(answer, "answer-1")
        self.assertEqual(
            api.requests,
            [
                {
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "temperature": 0.7,
                },
            ],
        )

if __name__ == "__main__":
    unittest.main()
