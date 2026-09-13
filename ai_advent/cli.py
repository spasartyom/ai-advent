import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from ai_advent.agent import Agent
from ai_advent.memory import JsonFileMemory

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
MEMORY_FILE_ENV = "AI_ADVENT_MEMORY_FILE"
DEFAULT_MEMORY_FILE = ".ai-advent/agent-memory.json"


def run_agent(
    agent: Agent,
    model: str,
    base_url: str | None,
    read_input: Callable[[str], str] = input,
) -> None:
    print(f"AI Advent agent (model: {model})")
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
            response = agent.run_turn(message)
        except APIError as error:
            print(f"API error: {error}", file=sys.stderr)
            continue

        print(f"Assistant: {response.text}\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI Advent CLI")
    subparsers = parser.add_subparsers(dest="command")

    agent_parser = subparsers.add_parser(
        "agent",
        help="run the week 2 AI agent",
    )
    agent_parser.add_argument(
        "--memory-file",
        default=None,
        help="JSON file used to persist agent messages",
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
    memory_file = getattr(
        args,
        "memory_file",
        None,
    ) or os.getenv(MEMORY_FILE_ENV, DEFAULT_MEMORY_FILE)
    memory = JsonFileMemory(Path(memory_file))
    agent = Agent(client.chat.completions, model, memory=memory)
    if args.command in {None, "agent"}:
        run_agent(agent, model, base_url)


if __name__ == "__main__":
    main()
