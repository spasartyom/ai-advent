import argparse
import os
import sys
from collections.abc import Callable

from ai_advent.chat import ChatSession

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False

try:
    from openai import APIError, OpenAI
except ModuleNotFoundError:
    APIError = Exception
    OpenAI = None

DEFAULT_MODEL = "gpt-5.6-luna"
EXIT_COMMANDS = {"/exit", "/quit"}
API_KEY_ENV = "AI_ADVENT_API_KEY"
BASE_URL_ENV = "AI_ADVENT_BASE_URL"
MODEL_ENV = "AI_ADVENT_MODEL"
DEFAULT_CITY_PROMPT = "Расскажи про город Казань"
DAY2_MAX_COMPLETION_TOKENS = 220
DEFAULT_DAY3_TASK = (
    "Составь идеальный завтрак для среднестатистического гражданина РФ."
)
DAY3_MAX_COMPLETION_TOKENS = 500
DEFAULT_DAY4_PROMPT = (
    "Придумай короткое описание нового кафе в центре города Минск, "
    "которое специализируется на завтраках."
)
DAY4_TEMPERATURES = (0.0, 0.7, 1.2)
DAY4_MAX_COMPLETION_TOKENS = 350


def build_city_json_prompt(question: str) -> str:
    return (
        f"{question}\n\n"
        "Ответь строго в JSON формате без Markdown и пояснений.\n"
        "JSON должен содержать поля:\n"
        '- "summary": string, общая короткая информация о городе;\n'
        '- "facts": string[], ровно 3 коротких факта;\n'
        '- "year_of_foundation": number, год основания города;\n'
        '- "population": number, численность населения.\n'
        "Ограничение длины: весь ответ должен быть не длиннее 120 слов.\n"
        "Условие завершения: заверши ответ сразу после закрывающей "
        "фигурной скобки JSON-объекта. После JSON не добавляй никакого текста."
    )


def build_step_by_step_prompt(task: str) -> str:
    return f"{task}\n\nРешай пошагово."


def build_prompt_generation_prompt(task: str) -> str:
    return (
        "Составь хороший промпт для решения аналитической задачи ниже.\n"
        "Промпт должен помочь получить обоснованный, практичный и культурно "
        "уместный ответ.\n\n"
        f"Задача:\n{task}\n\n"
        "Верни только готовый промпт без пояснений."
    )


def build_expert_group_prompt(task: str) -> str:
    return (
        "Реши задачу через группу экспертов.\n\n"
        f"Задача:\n{task}\n\n"
        "Эксперты:\n"
        "1. Нутрициолог — оценивает питательность и баланс.\n"
        "2. Культурный аналитик — оценивает соответствие привычкам страны.\n"
        "3. Практик — оценивает доступность, стоимость и простоту приготовления.\n"
        "4. Критик — ищет слабые места в предложениях.\n\n"
        "Каждый эксперт должен дать короткое решение. "
        "В конце дай общий финальный вариант завтрака."
    )


def build_reasoning_comparison_prompt(
    task: str,
    direct_answer: str,
    step_by_step_answer: str,
    generated_prompt: str,
    generated_prompt_answer: str,
    expert_group_answer: str,
) -> str:
    return (
        "Сравни четыре решения одной аналитической задачи и выбери наиболее "
        "точный способ.\n\n"
        f"Задача:\n{task}\n\n"
        f"1. Прямой ответ:\n{direct_answer}\n\n"
        f"2. Ответ с инструкцией решать пошагово:\n{step_by_step_answer}\n\n"
        f"3a. Промпт, созданный моделью:\n{generated_prompt}\n\n"
        f"3b. Ответ по созданному промпту:\n{generated_prompt_answer}\n\n"
        f"4. Ответ группы экспертов:\n{expert_group_answer}\n\n"
        "Критерии: культурная уместность для РФ, питательность, "
        "доступность продуктов, умеренная стоимость и практичность утром.\n"
        "Коротко опиши отличия и назови лучший способ."
    )


def build_temperature_comparison_prompt(
    prompt: str,
    answers_by_temperature: dict[float, str],
) -> str:
    answers = "\n\n".join(
        f"temperature = {temperature:g}:\n{answer}"
        for temperature, answer in answers_by_temperature.items()
    )
    return (
        "Сравни ответы модели на один и тот же запрос при разных значениях "
        "temperature.\n\n"
        f"Запрос:\n{prompt}\n\n"
        f"Ответы:\n{answers}\n\n"
        "Оцени каждый ответ по точности, креативности и разнообразию. "
        "В конце сформулируй, для каких задач лучше подходит temperature = 0, "
        "temperature = 0.7 и temperature = 1.2."
    )


def run_chat(
    session: ChatSession,
    model: str,
    base_url: str | None,
    read_input: Callable[[str], str] = input,
) -> None:
    print(f"AI Advent chat (model: {model})")
    if base_url:
        print(f"API base URL: {base_url}")
    print("Type /exit or /quit to stop.\n")

    while True:
        try:
            message = read_input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if message.lower() in EXIT_COMMANDS:
            print("Bye!")
            return

        if not message:
            continue

        try:
            answer = session.ask(message)
        except APIError as error:
            print(f"API error: {error}", file=sys.stderr)
            continue

        print(f"Assistant: {answer}\n")


def run_format_comparison(
    chat_completions_api: object,
    model: str,
    city_prompt: str = DEFAULT_CITY_PROMPT,
) -> None:
    plain_session = ChatSession(chat_completions_api, model)
    controlled_session = ChatSession(chat_completions_api, model)
    controlled_prompt = build_city_json_prompt(city_prompt)

    print(f"AI Advent day 2: response format comparison (model: {model})\n")
    print(f"Prompt: {city_prompt}\n")

    try:
        plain_answer = plain_session.ask(city_prompt)
        controlled_answer = controlled_session.ask(
            controlled_prompt,
            max_completion_tokens=DAY2_MAX_COMPLETION_TOKENS,
        )
    except APIError as error:
        print(f"API error: {error}", file=sys.stderr)
        raise SystemExit(1) from error

    print("=== Без ограничений ===")
    print(f"{plain_answer}\n")

    print("=== С ограничениями ===")
    print(f"{controlled_answer}\n")

    print("=== Сравнение ===")
    print("Без ограничений: свободная структура, произвольная длина и стиль.")
    print("С ограничениями: JSON-схема, лимит длины и явное условие завершения.")


def run_reasoning_comparison(
    chat_completions_api: object,
    model: str,
    task: str = DEFAULT_DAY3_TASK,
) -> None:
    direct_session = ChatSession(chat_completions_api, model)
    step_by_step_session = ChatSession(chat_completions_api, model)
    prompt_generation_session = ChatSession(chat_completions_api, model)
    generated_prompt_session = ChatSession(chat_completions_api, model)
    expert_group_session = ChatSession(chat_completions_api, model)
    comparison_session = ChatSession(chat_completions_api, model)

    print(f"AI Advent day 3: reasoning comparison (model: {model})\n")
    print(f"Task: {task}\n")

    try:
        print("=== 1. Прямой ответ ===")
        direct_answer = direct_session.ask(
            task,
            max_completion_tokens=DAY3_MAX_COMPLETION_TOKENS,
        )
        print(f"{direct_answer}\n")

        print("=== 2. Решай пошагово ===")
        step_by_step_answer = step_by_step_session.ask(
            build_step_by_step_prompt(task),
            max_completion_tokens=DAY3_MAX_COMPLETION_TOKENS,
        )
        print(f"{step_by_step_answer}\n")

        print("=== 3. Сначала промпт, затем решение ===")
        print("Generated prompt:")
        generated_prompt = prompt_generation_session.ask(
            build_prompt_generation_prompt(task),
            max_completion_tokens=DAY3_MAX_COMPLETION_TOKENS,
        )
        print(f"{generated_prompt}\n")

        print("Answer:")
        generated_prompt_answer = generated_prompt_session.ask(
            generated_prompt,
            max_completion_tokens=DAY3_MAX_COMPLETION_TOKENS,
        )
        print(f"{generated_prompt_answer}\n")

        print("=== 4. Группа экспертов ===")
        expert_group_answer = expert_group_session.ask(
            build_expert_group_prompt(task),
            max_completion_tokens=DAY3_MAX_COMPLETION_TOKENS,
        )
        print(f"{expert_group_answer}\n")

        print("=== Сравнение ===")
        comparison_answer = comparison_session.ask(
            build_reasoning_comparison_prompt(
                task,
                direct_answer,
                step_by_step_answer,
                generated_prompt,
                generated_prompt_answer,
                expert_group_answer,
            ),
            max_completion_tokens=DAY3_MAX_COMPLETION_TOKENS,
        )
        print(comparison_answer)
    except APIError as error:
        print(f"API error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def run_temperature_comparison(
    chat_completions_api: object,
    model: str,
    prompt: str = DEFAULT_DAY4_PROMPT,
) -> None:
    answers_by_temperature: dict[float, str] = {}
    comparison_session = ChatSession(chat_completions_api, model)

    print(f"AI Advent day 4: temperature comparison (model: {model})\n")
    print(f"Prompt: {prompt}\n")

    try:
        for temperature in DAY4_TEMPERATURES:
            print(f"=== temperature = {temperature:g} ===")
            session = ChatSession(chat_completions_api, model)
            answer = session.ask(
                prompt,
                max_completion_tokens=DAY4_MAX_COMPLETION_TOKENS,
                temperature=temperature,
            )
            answers_by_temperature[temperature] = answer
            print(f"{answer}\n")

        print("=== Сравнение ===")
        comparison_answer = comparison_session.ask(
            build_temperature_comparison_prompt(prompt, answers_by_temperature),
            max_completion_tokens=DAY4_MAX_COMPLETION_TOKENS,
        )
        print(comparison_answer)
    except APIError as error:
        print(f"API error: {error}", file=sys.stderr)
        raise SystemExit(1) from error


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Advent CLI")
    subparsers = parser.add_subparsers(dest="command")

    compare_parser = subparsers.add_parser(
        "compare-format",
        help="compare an unconstrained answer with a controlled JSON answer",
    )
    compare_parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_CITY_PROMPT,
        help="city question to send in both requests",
    )

    reasoning_parser = subparsers.add_parser(
        "compare-reasoning",
        help="compare several prompting strategies for one analytical task",
    )
    reasoning_parser.add_argument(
        "task",
        nargs="?",
        default=DEFAULT_DAY3_TASK,
        help="analytical task to solve with several prompting strategies",
    )

    temperature_parser = subparsers.add_parser(
        "compare-temperature",
        help="compare answers generated with different temperature values",
    )
    temperature_parser.add_argument(
        "prompt",
        nargs="?",
        default=DEFAULT_DAY4_PROMPT,
        help="prompt to send with several temperature values",
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    load_dotenv()

    if OpenAI is None:
        print(
            "OpenAI SDK is not installed. Run: python -m pip install -e .",
            file=sys.stderr,
        )
        raise SystemExit(1)

    api_key = os.getenv(API_KEY_ENV) or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(
            f"{API_KEY_ENV} is not set. Copy .env.example to .env and add your key.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    model = os.getenv(MODEL_ENV) or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
    base_url = os.getenv(BASE_URL_ENV) or os.getenv("OPENAI_BASE_URL")

    client = OpenAI(api_key=api_key, base_url=base_url)
    if args.command == "compare-format":
        run_format_comparison(client.chat.completions, model, args.prompt)
    elif args.command == "compare-reasoning":
        run_reasoning_comparison(client.chat.completions, model, args.task)
    elif args.command == "compare-temperature":
        run_temperature_comparison(client.chat.completions, model, args.prompt)
    else:
        session = ChatSession(client.chat.completions, model)
        run_chat(session, model, base_url)


if __name__ == "__main__":
    main()
