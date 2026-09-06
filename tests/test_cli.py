import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

from ai_advent.cli import (
    DAY2_MAX_COMPLETION_TOKENS,
    DAY3_MAX_COMPLETION_TOKENS,
    DAY4_MAX_COMPLETION_TOKENS,
    DAY4_TEMPERATURES,
    DEFAULT_DAY5_PROMPT,
    DAY5_MAX_TOKENS,
    build_model_comparison_prompt,
    build_city_json_prompt,
    build_expert_group_prompt,
    build_prompt_generation_prompt,
    build_reasoning_comparison_prompt,
    build_step_by_step_prompt,
    build_temperature_comparison_prompt,
    calculate_zai_cost_usd,
    ModelComparisonResult,
    run_format_comparison,
    run_model_comparison,
    run_reasoning_comparison,
    run_temperature_comparison,
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
            usage=SimpleNamespace(
                prompt_tokens=100,
                completion_tokens=200,
                total_tokens=300,
            ),
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


class Day4TemperatureComparisonTests(unittest.TestCase):
    def test_temperature_comparison_prompt_includes_answers_and_criteria(self) -> None:
        prompt = build_temperature_comparison_prompt(
            "Опиши кафе.",
            {
                0.0: "answer zero",
                0.7: "answer medium",
                1.2: "answer high",
            },
        )

        self.assertIn("Опиши кафе.", prompt)
        self.assertIn("temperature = 0", prompt)
        self.assertIn("temperature = 0.7", prompt)
        self.assertIn("temperature = 1.2", prompt)
        self.assertIn("точности", prompt)
        self.assertIn("креативности", prompt)
        self.assertIn("разнообразию", prompt)

    def test_temperature_comparison_sends_expected_requests(self) -> None:
        api = FakeChatCompletionsAPI()
        output = io.StringIO()

        with redirect_stdout(output):
            run_temperature_comparison(api, "test-model", "Опиши кафе.")

        self.assertEqual(len(api.requests), 4)
        for index, temperature in enumerate(DAY4_TEMPERATURES):
            self.assertEqual(
                api.requests[index]["messages"],
                [{"role": "user", "content": "Опиши кафе."}],
            )
            self.assertEqual(api.requests[index]["temperature"], temperature)
            self.assertEqual(
                api.requests[index]["max_completion_tokens"],
                DAY4_MAX_COMPLETION_TOKENS,
            )

        self.assertNotIn("temperature", api.requests[3])
        self.assertIn("Сравни ответы", api.requests[3]["messages"][0]["content"])
        self.assertIn("temperature = 0", output.getvalue())
        self.assertIn("temperature = 0.7", output.getvalue())
        self.assertIn("temperature = 1.2", output.getvalue())


class Day5ModelComparisonTests(unittest.TestCase):
    def test_default_prompt_asks_for_mvp_specification(self) -> None:
        self.assertIn("AI Advent Tracker", DEFAULT_DAY5_PROMPT)
        self.assertIn("спецификацию MVP", DEFAULT_DAY5_PROMPT)
        self.assertIn("5 ключевых функций", DEFAULT_DAY5_PROMPT)
        self.assertIn("3 метрики успеха", DEFAULT_DAY5_PROMPT)
        self.assertIn("до 900 слов", DEFAULT_DAY5_PROMPT)
        self.assertIn("без дополнительных вопросов", DEFAULT_DAY5_PROMPT)

    def test_calculate_zai_cost_uses_input_and_output_prices(self) -> None:
        self.assertEqual(
            calculate_zai_cost_usd("glm-4.5-air", 1_000_000, 1_000_000),
            1.3,
        )
        self.assertEqual(calculate_zai_cost_usd("glm-4.5-flash", 100, 200), 0.0)
        self.assertIsNone(calculate_zai_cost_usd("unknown-model", 100, 200))
        self.assertIsNone(calculate_zai_cost_usd("glm-5", None, 200))

    def test_model_comparison_prompt_includes_answers_and_metrics(self) -> None:
        prompt = build_model_comparison_prompt(
            "Explain hallucinations.",
            [
                ModelComparisonResult(
                    model="glm-a",
                    answer="short answer",
                    elapsed_seconds=1.23,
                    prompt_tokens=10,
                    completion_tokens=20,
                    total_tokens=30,
                    cost_usd=0.0001,
                ),
            ],
        )

        self.assertIn("Explain hallucinations.", prompt)
        self.assertIn("glm-a", prompt)
        self.assertIn("short answer", prompt)
        self.assertIn("1.23", prompt)
        self.assertIn("$0.000100", prompt)
        self.assertIn("точность", prompt)
        self.assertIn("стоимость", prompt)

    def test_model_comparison_sends_same_prompt_to_all_models(self) -> None:
        api = FakeChatCompletionsAPI()
        output = io.StringIO()

        with redirect_stdout(output):
            run_model_comparison(
                api,
                "Explain hallucinations.",
                ("glm-weak", "glm-mid", "glm-strong"),
                "glm-judge",
            )

        self.assertEqual(len(api.requests), 4)
        for index, model in enumerate(("glm-weak", "glm-mid", "glm-strong")):
            self.assertEqual(api.requests[index]["model"], model)
            self.assertEqual(
                api.requests[index]["messages"],
                [{"role": "user", "content": "Explain hallucinations."}],
            )
            self.assertEqual(api.requests[index]["max_tokens"], DAY5_MAX_TOKENS)

        self.assertEqual(api.requests[3]["model"], "glm-judge")
        self.assertIn("Сравни ответы разных GLM-моделей", api.requests[3]["messages"][0]["content"])
        self.assertIn("Сводная таблица", output.getvalue())
        self.assertIn("glm-weak", output.getvalue())
        self.assertIn("Tokens: prompt=100, completion=200, total=300", output.getvalue())


if __name__ == "__main__":
    unittest.main()
