import argparse
import os
import shlex
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

        if handle_memory_command(agent, message):
            continue

        if handle_profile_command(agent, message):
            continue

        if handle_task_command(agent, message):
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


def handle_memory_command(agent: Agent, message: str) -> bool:
    parts = split_command(message)
    if not parts or parts[0].lower() != "/memory":
        return False

    args = parts[1:]
    if not args:
        print_memory_usage()
        return True

    subcommand = args[0].lower()
    try:
        if subcommand == "show":
            layer = args[1].lower() if len(args) == 2 else "all"
            if len(args) > 2:
                print_memory_usage()
                return True
            print_memory_layer(agent, layer)
            return True

        if subcommand == "set":
            if len(args) < 4:
                print("Usage: /memory set working|long KEY VALUE")
                return True
            set_memory_value(agent, args[1].lower(), args[2], " ".join(args[3:]))
            return True

        if subcommand in {"forget", "remove"}:
            if len(args) != 3:
                print("Usage: /memory forget working|long KEY")
                return True
            forget_memory_value(agent, args[1].lower(), args[2])
            return True
    except ValueError as error:
        print(f"Memory error: {error}", file=sys.stderr)
        return True

    print_memory_usage()
    return True


def print_memory_usage() -> None:
    print("Usage: /memory show [short|working|long|all]")
    print("       /memory set working|long KEY VALUE")
    print("       /memory forget working|long KEY")


def handle_profile_command(agent: Agent, message: str) -> bool:
    parts = split_command(message)
    if not parts or parts[0].lower() != "/profile":
        return False

    args = parts[1:]
    if not args:
        print_profile_usage()
        return True

    subcommand = args[0].lower()
    try:
        if subcommand == "show":
            if len(args) != 1:
                print_profile_usage()
                return True
            print_string_map("User profile", agent.user_profile)
            return True

        if subcommand == "set":
            if len(args) < 3:
                print("Usage: /profile set KEY VALUE")
                return True
            agent.remember_profile(args[1], " ".join(args[2:]))
            print(f"Profile saved: {args[1]}")
            return True

        if subcommand in {"forget", "remove"}:
            if len(args) != 2:
                print("Usage: /profile forget KEY")
                return True
            agent.forget_profile(args[1])
            print(f"Profile removed: {args[1]}")
            return True
    except ValueError as error:
        print(f"Profile error: {error}", file=sys.stderr)
        return True

    print_profile_usage()
    return True


def print_profile_usage() -> None:
    print("Usage: /profile show")
    print("       /profile set KEY VALUE")
    print("       /profile forget KEY")


def handle_task_command(agent: Agent, message: str) -> bool:
    parts = split_command(message)
    if not parts or parts[0].lower() != "/task":
        return False

    args = parts[1:]
    if not args:
        print_task_usage()
        return True

    subcommand = args[0].lower()
    try:
        if subcommand == "status":
            if len(args) != 1:
                print_task_usage()
                return True
            print_task_state(agent)
            return True

        if subcommand == "start":
            if len(args) < 2:
                print("Usage: /task start TITLE")
                return True
            agent.start_task(" ".join(args[1:]))
            print_task_state(agent)
            return True

        if subcommand == "stage":
            if len(args) != 2:
                print("Usage: /task stage idle|planning|execution|validation|done")
                return True
            agent.update_task_state(stage=args[1])
            print_task_state(agent)
            return True

        if subcommand == "step":
            if len(args) < 2:
                print("Usage: /task step CURRENT_STEP")
                return True
            agent.update_task_state(current_step=" ".join(args[1:]))
            print_task_state(agent)
            return True

        if subcommand == "expect":
            if len(args) < 2:
                print("Usage: /task expect EXPECTED_ACTION")
                return True
            agent.update_task_state(expected_action=" ".join(args[1:]))
            print_task_state(agent)
            return True

        if subcommand == "title":
            if len(args) < 2:
                print("Usage: /task title TITLE")
                return True
            agent.update_task_state(title=" ".join(args[1:]))
            print_task_state(agent)
            return True

        if subcommand == "pause":
            if len(args) != 1:
                print_task_usage()
                return True
            agent.pause_task()
            print_task_state(agent)
            return True

        if subcommand == "resume":
            if len(args) != 1:
                print_task_usage()
                return True
            agent.resume_task()
            print_task_state(agent)
            return True

        if subcommand == "done":
            if len(args) != 1:
                print_task_usage()
                return True
            agent.update_task_state(
                stage="done",
                current_step="Задача завершена.",
                expected_action="Нет ожидаемого действия.",
                paused=False,
            )
            print_task_state(agent)
            return True
    except ValueError as error:
        print(f"Task error: {error}", file=sys.stderr)
        return True

    print_task_usage()
    return True


def print_task_usage() -> None:
    print("Usage: /task status")
    print("       /task start TITLE")
    print("       /task stage idle|planning|execution|validation|done")
    print("       /task step CURRENT_STEP")
    print("       /task expect EXPECTED_ACTION")
    print("       /task title TITLE")
    print("       /task pause")
    print("       /task resume")
    print("       /task done")


def print_task_state(agent: Agent) -> None:
    task_state = agent.task_state
    print("Task state:")
    print(f"  stage: {task_state.stage}")
    print(f"  title: {task_state.title or '(not set)'}")
    print(f"  current_step: {task_state.current_step or '(not set)'}")
    print(f"  expected_action: {task_state.expected_action or '(not set)'}")
    print(f"  paused: {task_state.paused}")


def split_command(message: str) -> list[str]:
    try:
        return shlex.split(message)
    except ValueError:
        return message.split()


def print_memory_layer(agent: Agent, layer: str) -> None:
    if layer == "all":
        print_memory_layer(agent, "short")
        print_memory_layer(agent, "working")
        print_memory_layer(agent, "long")
        return

    if layer == "short":
        messages = agent.messages
        print(f"Short-term memory: {len(messages)} messages")
        for message in messages:
            print(f"  {message['role']}: {message['content']}")
        return

    if layer == "working":
        print_string_map("Working memory", agent.working_memory)
        return

    if layer in {"long", "long-term", "long_term"}:
        print_string_map("Long-term memory", agent.long_term_memory)
        return

    print_memory_usage()


def print_string_map(label: str, values: dict[str, str]) -> None:
    print(f"{label}:")
    if not values:
        print("  (empty)")
        return
    for key, value in sorted(values.items()):
        print(f"  {key}: {value}")


def set_memory_value(agent: Agent, layer: str, key: str, value: str) -> None:
    if layer == "working":
        agent.remember_working(key, value)
        print(f"Working memory saved: {key}")
        return
    if layer in {"long", "long-term", "long_term"}:
        agent.remember_long_term(key, value)
        print(f"Long-term memory saved: {key}")
        return
    raise ValueError("Memory layer must be working or long.")


def forget_memory_value(agent: Agent, layer: str, key: str) -> None:
    if layer == "working":
        agent.forget_working(key)
        print(f"Working memory removed: {key}")
        return
    if layer in {"long", "long-term", "long_term"}:
        agent.forget_long_term(key)
        print(f"Long-term memory removed: {key}")
        return
    raise ValueError("Memory layer must be working or long.")


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
