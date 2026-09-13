import unittest
from types import SimpleNamespace

from ai_advent.context import (
    ContextState,
    FullContextStrategy,
    SummaryContextStrategy,
    summarize_messages,
)


class FakeChatCompletionsAPI:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def create(self, **kwargs: object) -> object:
        self.requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="updated summary"),
                ),
            ],
            usage=None,
        )


class ContextStrategyTests(unittest.TestCase):
    def test_full_context_strategy_returns_all_messages(self) -> None:
        strategy = FullContextStrategy()
        state = ContextState(
            messages=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        )

        self.assertEqual(strategy.build_messages(state), state.messages)

    def test_summary_strategy_adds_summary_before_recent_messages(self) -> None:
        strategy = SummaryContextStrategy(keep_last=2)
        state = ContextState(
            summary="User prefers concise answers.",
            messages=[
                {"role": "user", "content": "one"},
                {"role": "assistant", "content": "two"},
            ],
        )

        messages = strategy.build_messages(state)

        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("User prefers concise answers.", messages[0]["content"])
        self.assertEqual(messages[1:], state.messages)

    def test_summary_strategy_compresses_old_messages(self) -> None:
        api = FakeChatCompletionsAPI()
        strategy = SummaryContextStrategy(keep_last=2)
        state = ContextState(
            summary="existing summary",
            messages=[
                {"role": "user", "content": "old one"},
                {"role": "assistant", "content": "old two"},
                {"role": "user", "content": "recent one"},
                {"role": "assistant", "content": "recent two"},
            ],
        )

        result = strategy.compress(state, api, "test-model")

        self.assertTrue(result.changed)
        self.assertEqual(result.state.summary, "updated summary")
        self.assertEqual(
            result.state.messages,
            [
                {"role": "user", "content": "recent one"},
                {"role": "assistant", "content": "recent two"},
            ],
        )
        self.assertIn("existing summary", api.requests[0]["messages"][1]["content"])
        self.assertIn("old one", api.requests[0]["messages"][1]["content"])

    def test_summary_strategy_keeps_short_history_unchanged(self) -> None:
        api = FakeChatCompletionsAPI()
        strategy = SummaryContextStrategy(keep_last=3)
        state = ContextState(messages=[{"role": "user", "content": "short"}])

        result = strategy.compress(state, api, "test-model")

        self.assertFalse(result.changed)
        self.assertEqual(result.state, state)
        self.assertEqual(api.requests, [])

    def test_summary_strategy_rejects_invalid_keep_last(self) -> None:
        with self.assertRaises(ValueError):
            SummaryContextStrategy(keep_last=0)

    def test_summarize_messages_returns_stripped_response_text(self) -> None:
        api = FakeChatCompletionsAPI()

        summary = summarize_messages(
            api,
            "test-model",
            "",
            [{"role": "user", "content": "remember лимон-47"}],
        )

        self.assertEqual(summary, "updated summary")
        self.assertEqual(api.requests[0]["model"], "test-model")
        self.assertEqual(api.requests[0]["max_completion_tokens"], 700)


if __name__ == "__main__":
    unittest.main()
