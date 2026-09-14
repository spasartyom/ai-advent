import argparse
import os
import sys
from collections.abc import Callable
from pathlib import Path

from ai_advent.agent import Agent
from ai_advent.context import (
    FullContextStrategy,
    SlidingWindowContextStrategy,
    StickyFactsContextStrategy,
    SummaryContextStrategy,
)
from ai_advent.memory import JsonFileMemory
from ai_advent.tokens import TokenReport

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
PASTE_COMMAND = "/paste"
SEND_COMMAND = "/send"
API_KEY_ENV = "AI_ADVENT_API_KEY"
BASE_URL_ENV = "AI_ADVENT_BASE_URL"
MODEL_ENV = "AI_ADVENT_MODEL"
MEMORY_FILE_ENV = "AI_ADVENT_MEMORY_FILE"
DEFAULT_MEMORY_FILE = ".ai-advent/agent-memory.json"
DEFAULT_CONTEXT_STRATEGY = "full"
DEFAULT_KEEP_LAST = 10


def run_agent(
    agent: Agent,
    model: str,
    base_url: str | None,
    *,
    show_tokens: bool = False,
    read_input: Callable[[str], str] = input,
) -> None:
    print(f"AI Advent agent (model: {model})")
    if base_url:
        print(f"API base URL: {base_url}")
    print("Type /exit or /quit to stop. Use /paste, then /send for multiline input.\n")

    while True:
        try:
            message = read_input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if message.lower() in EXIT_COMMANDS:
            print("Bye!")
            return

        if message.lower() == PASTE_COMMAND:
            message = read_paste_block(read_input)
            if message is None:
                print("Bye!")
                return

        if handle_branch_command(agent, message):
            continue

        if not message:
            continue

        try:
            response = agent.run_turn(message)
        except APIError as error:
            print(f"API error: {error}", file=sys.stderr)
            continue

        print(f"Assistant: {response.text}\n")
        if show_tokens and response.token_report is not None:
            print_token_report(response.token_report)


def handle_branch_command(agent: Agent, message: str) -> bool:
    parts = message.split()
    if not parts:
        return False

    command = parts[0].lower()
    try:
        if command == "/checkpoint":
            if len(parts) != 2:
                print("Usage: /checkpoint NAME")
                return True
            agent.save_checkpoint(parts[1])
            print(f"Checkpoint saved: {parts[1]}")
            return True

        if command == "/branch":
            return handle_branch_subcommand(agent, parts[1:])
    except ValueError as error:
        print(f"Branch error: {error}", file=sys.stderr)
        return True

    return False


def handle_branch_subcommand(agent: Agent, args: list[str]) -> bool:
    if not args:
        print("Usage: /branch list | create NAME [CHECKPOINT] | switch NAME | checkpoints")
        return True

    subcommand = args[0].lower()
    if subcommand == "list":
        print(f"Branches: {', '.join(agent.list_branches())}")
        print(f"Current branch: {agent.current_branch}")
        return True

    if subcommand == "checkpoints":
        checkpoints = agent.list_checkpoints()
        print(f"Checkpoints: {', '.join(checkpoints) if checkpoints else '(none)'}")
        return True

    if subcommand == "create":
        if len(args) not in {2, 3}:
            print("Usage: /branch create NAME [CHECKPOINT]")
            return True
        checkpoint = args[2] if len(args) == 3 else None
        agent.create_branch(args[1], checkpoint)
        print(f"Branch created and selected: {args[1]}")
        return True

    if subcommand == "switch":
        if len(args) != 2:
            print("Usage: /branch switch NAME")
            return True
        agent.switch_branch(args[1])
        print(f"Switched to branch: {args[1]}")
        return True

    print("Usage: /branch list | create NAME [CHECKPOINT] | switch NAME | checkpoints")
    return True


def read_paste_block(read_input: Callable[[str], str]) -> str | None:
    print(f"Paste multiline text. Finish with {SEND_COMMAND}.")
    lines: list[str] = []

    while True:
        try:
            line = read_input("")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return None

        if line.strip().lower() == SEND_COMMAND:
            return "\n".join(lines).strip()

        lines.append(line)


def print_token_report(report: TokenReport) -> None:
    print("Tokens:")
    print(f"  prompt_tokens: {_format_optional_int(report.prompt_tokens)}")
    print(f"  completion_tokens: {_format_optional_int(report.completion_tokens)}")
    print(f"  total_tokens: {_format_optional_int(report.total_tokens)}\n")


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
    agent_parser.add_argument(
        "--show-tokens",
        action="store_true",
        help="print API token usage after each agent response",
    )
    agent_parser.add_argument(
        "--context-strategy",
        choices=("full", "summary", "sliding-window", "facts", "branch"),
        default=DEFAULT_CONTEXT_STRATEGY,
        help="context management strategy",
    )
    agent_parser.add_argument(
        "--keep-last",
        type=int,
        default=DEFAULT_KEEP_LAST,
        help="number of recent messages to keep in compact context strategies",
    )
    agent_parser.add_argument(
        "--branch",
        default=None,
        help="initial branch name for branch context experiments",
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
    context_strategy = build_context_strategy(args.context_strategy, args.keep_last)
    agent = Agent(
        client.chat.completions,
        model,
        memory=memory,
        context_strategy=context_strategy,
        branch=args.branch,
    )
    if args.command in {None, "agent"}:
        run_agent(
            agent,
            model,
            base_url,
            show_tokens=getattr(args, "show_tokens", False),
        )


def build_context_strategy(
    strategy_name: str,
    keep_last: int,
) -> (
    FullContextStrategy
    | SummaryContextStrategy
    | SlidingWindowContextStrategy
    | StickyFactsContextStrategy
):
    if strategy_name == "summary":
        return SummaryContextStrategy(keep_last)
    if strategy_name == "sliding-window":
        return SlidingWindowContextStrategy(keep_last)
    if strategy_name == "facts":
        return StickyFactsContextStrategy(keep_last)

    return FullContextStrategy()


def _format_optional_int(value: int | None) -> str:
    return "n/a" if value is None else str(value)


if __name__ == "__main__":
    main()
