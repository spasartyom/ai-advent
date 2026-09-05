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
    else:
        session = ChatSession(client.chat.completions, model)
        run_chat(session, model, base_url)


if __name__ == "__main__":
    main()
