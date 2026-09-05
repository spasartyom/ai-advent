import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from ai_advent.cli import (
    DAY2_MAX_COMPLETION_TOKENS,
    build_city_json_prompt,
    run_format_comparison,
)


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


class Day2FormatComparisonTests(unittest.TestCase):
    def test_city_json_prompt_describes_expected_response(self) -> None:
        prompt = build_city_json_prompt("Расскажи про город Сызрань.")

        self.assertIn("Расскажи про город Сызрань.", prompt)
        self.assertIn("строго в JSON", prompt)
        self.assertIn('"summary"', prompt)
        self.assertIn('"facts"', prompt)
        self.assertIn('"year_of_foundation"', prompt)
        self.assertIn('"population"', prompt)
        self.assertIn("не длиннее 120 слов", prompt)
        self.assertIn("после закрывающей фигурной скобки", prompt)

    def test_format_comparison_sends_plain_and_controlled_requests(self) -> None:
        api = FakeChatCompletionsAPI()
        output = io.StringIO()

        with redirect_stdout(output):
            run_format_comparison(api, "test-model", "Расскажи про город Сызрань.")

        self.assertEqual(len(api.requests), 1)
        self.assertEqual(api.requests[0]["model"], "test-model")
        self.assertEqual(
            api.requests[0]["max_completion_tokens"],
            DAY2_MAX_COMPLETION_TOKENS,
        )
        self.assertNotIn("stop", api.requests[0])
        self.assertIn("Без ограничений", output.getvalue())
        self.assertIn("С ограничениями", output.getvalue())


if __name__ == "__main__":
    unittest.main()
