import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from ai_advent.cli import (
    DAY2_MAX_COMPLETION_TOKENS,
    DAY3_MAX_COMPLETION_TOKENS,
    build_city_json_prompt,
    build_expert_group_prompt,
    build_prompt_generation_prompt,
    build_reasoning_comparison_prompt,
    build_step_by_step_prompt,
    run_format_comparison,
    run_reasoning_comparison,
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

        self.assertEqual(len(api.requests), 2)
        self.assertEqual(api.requests[0]["model"], "test-model")
        self.assertEqual(
            api.requests[0]["messages"],
            [
                {
                    "role": "user",
                    "content": "Расскажи про город Сызрань.",
                },
            ],
        )
        self.assertNotIn("max_completion_tokens", api.requests[0])
        self.assertEqual(
            api.requests[1]["max_completion_tokens"],
            DAY2_MAX_COMPLETION_TOKENS,
        )
        self.assertNotIn("stop", api.requests[1])
        self.assertIn("Без ограничений", output.getvalue())
        self.assertIn("С ограничениями", output.getvalue())


class Day3ReasoningComparisonTests(unittest.TestCase):
    def test_reasoning_prompts_describe_expected_strategies(self) -> None:
        task = "Составь идеальный завтрак."

        self.assertIn("Решай пошагово", build_step_by_step_prompt(task))
        self.assertIn("Составь хороший промпт", build_prompt_generation_prompt(task))
        self.assertIn("Нутрициолог", build_expert_group_prompt(task))
        self.assertIn("Культурный аналитик", build_expert_group_prompt(task))

    def test_reasoning_comparison_prompt_includes_all_answers(self) -> None:
        prompt = build_reasoning_comparison_prompt(
            "task",
            "direct",
            "step",
            "generated prompt",
            "generated answer",
            "experts",
        )

        self.assertIn("task", prompt)
        self.assertIn("direct", prompt)
        self.assertIn("step", prompt)
        self.assertIn("generated prompt", prompt)
        self.assertIn("generated answer", prompt)
        self.assertIn("experts", prompt)
        self.assertIn("лучший способ", prompt)

    def test_reasoning_comparison_sends_expected_requests(self) -> None:
        api = FakeChatCompletionsAPI()
        output = io.StringIO()

        with redirect_stdout(output):
            run_reasoning_comparison(api, "test-model", "Составь идеальный завтрак.")

        self.assertEqual(len(api.requests), 6)
        self.assertEqual(
            api.requests[0]["messages"],
            [{"role": "user", "content": "Составь идеальный завтрак."}],
        )
        self.assertIn("Решай пошагово", api.requests[1]["messages"][0]["content"])
        self.assertIn(
            "Составь хороший промпт",
            api.requests[2]["messages"][0]["content"],
        )
        self.assertEqual(api.requests[3]["messages"][0]["content"], "answer-3")
        self.assertIn("Нутрициолог", api.requests[4]["messages"][0]["content"])
        self.assertIn("Сравни четыре решения", api.requests[5]["messages"][0]["content"])
        self.assertTrue(
            all(
                request["max_completion_tokens"] == DAY3_MAX_COMPLETION_TOKENS
                for request in api.requests
            )
        )
        self.assertIn("Прямой ответ", output.getvalue())
        self.assertIn("Группа экспертов", output.getvalue())


if __name__ == "__main__":
    unittest.main()
