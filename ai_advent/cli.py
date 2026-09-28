import argparse
import json
import os
import shlex
import sys
import time
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
from ai_advent.mcp_client import McpTool, McpToolResult, call_tool_sync, list_tools_sync
from ai_advent.orchestration import (
    OrchestrationResult,
    RegisteredTool,
    build_default_orchestrator,
    run_study_orchestration_flow,
)
from ai_advent.pipeline import PipelineResult, run_study_note_pipeline
from ai_advent.scheduler import DEFAULT_SCHEDULER_FILE
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
SCHEDULER_FILE_ENV = "AI_ADVENT_SCHEDULER_FILE"
NOTES_DIR_ENV = "AI_ADVENT_NOTES_DIR"
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

        if handle_invariant_command(agent, message):
            continue

        if handle_lesson_command(agent, message, show_tokens=show_tokens):
            continue

        if handle_reminder_command(message):
            continue

        if handle_note_command(message):
            continue

        if handle_orchestration_command(message):
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


def handle_lesson_command(
    agent: Agent,
    message: str,
    *,
    show_tokens: bool = False,
) -> bool:
    parts = split_command(message)
    if not parts or parts[0].lower() != "/lesson":
        return False

    if len(parts) < 2:
        print("Usage: /lesson TOPIC")
        return True

    topic = " ".join(parts[1:])
    try:
        tool_result = call_tool_sync("get_lesson", {"topic": topic})
    except (RuntimeError, ValueError) as error:
        print(f"MCP error: {error}", file=sys.stderr)
        return True

    print_mcp_tool_result(tool_result)
    if tool_result.is_error:
        return True

    prompt = (
        "Используй результат MCP-инструмента `get_lesson` как учебный материал. "
        "Кратко объясни тему и предложи один следующий практический шаг.\n\n"
        f"Тема: {topic}\n"
        f"MCP result:\n{tool_result.content_text}"
    )
    try:
        response = agent.run_turn(prompt)
    except APIError as error:
        print(f"API error: {error}", file=sys.stderr)
        return True

    print(f"Assistant: {response.text}\n")
    token_report = getattr(response, "token_report", None)
    if show_tokens and token_report is not None:
        print_token_report(token_report)
    return True


def handle_reminder_command(message: str) -> bool:
    parts = split_command(message)
    if not parts:
        return False

    command = parts[0].lower()
    if command == "/reminders":
        if len(parts) != 1:
            print("Usage: /reminders")
            return True
        try:
            result = call_tool_sync("list_reminders")
        except (RuntimeError, ValueError) as error:
            print(f"MCP error: {error}", file=sys.stderr)
            return True
        print_mcp_tool_result(result)
        return True

    if command != "/remind":
        return False

    if len(parts) < 3:
        print("Usage: /remind DUE_IN_SECONDS TITLE")
        return True

    try:
        due_in_seconds = int(parts[1])
    except ValueError:
        print("Reminder error: DUE_IN_SECONDS must be an integer.", file=sys.stderr)
        return True

    title = " ".join(parts[2:])
    try:
        result = call_tool_sync(
            "create_reminder",
            {
                "title": title,
                "due_in_seconds": due_in_seconds,
            },
        )
    except (RuntimeError, ValueError) as error:
        print(f"MCP error: {error}", file=sys.stderr)
        return True

    print_mcp_tool_result(result)
    return True


def handle_note_command(message: str) -> bool:
    parts = split_command(message)
    if not parts or parts[0].lower() != "/note":
        return False

    if len(parts) < 2:
        print("Usage: /note QUERY")
        return True

    query = " ".join(parts[1:])
    try:
        result = run_study_note_pipeline(
            query,
            caller=lambda tool_name, arguments: call_tool_sync(
                tool_name,
                arguments,
                env=build_mcp_env(),
            ),
        )
    except (RuntimeError, ValueError) as error:
        print(f"Pipeline error: {error}", file=sys.stderr)
        return True

    print_pipeline_result(result)
    return True


def handle_orchestration_command(message: str) -> bool:
    parts = split_command(message)
    if not parts or parts[0].lower() != "/study-flow":
        return False

    if len(parts) < 2:
        print("Usage: /study-flow QUERY")
        return True

    query = " ".join(parts[1:])
    orchestrator = build_default_orchestrator(
        scheduler_file=os.getenv(SCHEDULER_FILE_ENV),
        notes_dir=os.getenv(NOTES_DIR_ENV),
    )
    try:
        registered_tools = orchestrator.discover_tools()
        result = run_study_orchestration_flow(query, orchestrator=orchestrator)
    except (RuntimeError, ValueError) as error:
        print(f"Orchestration error: {error}", file=sys.stderr)
        return True

    print_registered_tools(registered_tools)
    print_orchestration_result(result)
    return True


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

        if subcommand == "approve":
            if len(args) != 1:
                print_task_usage()
                return True
            agent.approve_task()
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
    print("       /task approve")
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


def handle_invariant_command(agent: Agent, message: str) -> bool:
    parts = split_command(message)
    if not parts or parts[0].lower() not in {"/invariant", "/invariants"}:
        return False

    args = parts[1:]
    if not args:
        print_invariant_usage()
        return True

    subcommand = args[0].lower()
    try:
        if subcommand == "show":
            if len(args) != 1:
                print_invariant_usage()
                return True
            print_string_map("Invariants", agent.invariants)
            return True

        if subcommand in {"add", "set"}:
            if len(args) < 3:
                print("Usage: /invariant add ID TEXT")
                return True
            agent.remember_invariant(args[1], " ".join(args[2:]))
            print(f"Invariant saved: {args[1]}")
            return True

        if subcommand in {"forget", "remove"}:
            if len(args) != 2:
                print("Usage: /invariant remove ID")
                return True
            agent.forget_invariant(args[1])
            print(f"Invariant removed: {args[1]}")
            return True
    except ValueError as error:
        print(f"Invariant error: {error}", file=sys.stderr)
        return True

    print_invariant_usage()
    return True


def print_invariant_usage() -> None:
    print("Usage: /invariant show")
    print("       /invariant add ID TEXT")
    print("       /invariant remove ID")


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

    mcp_parser = subparsers.add_parser(
        "mcp",
        help="inspect MCP servers and tools",
    )
    mcp_subparsers = mcp_parser.add_subparsers(dest="mcp_command")
    list_tools_parser = mcp_subparsers.add_parser(
        "list-tools",
        help="connect to an MCP server and print available tools",
    )
    list_tools_parser.add_argument(
        "--url",
        default=None,
        help="MCP Streamable HTTP URL, for example http://127.0.0.1:8000/mcp",
    )
    list_tools_parser.add_argument(
        "--server-command",
        default=None,
        help="stdio server command. Defaults to the local AI Advent study server.",
    )
    list_tools_parser.add_argument(
        "--scheduler-file",
        default=None,
        help="JSON file used by local scheduler MCP tools",
    )
    list_tools_parser.add_argument(
        "--notes-dir",
        default=None,
        help="directory used by local note-saving MCP tools",
    )
    call_tool_parser = mcp_subparsers.add_parser(
        "call-tool",
        help="connect to an MCP server and call a tool",
    )
    call_tool_parser.add_argument("tool_name", help="MCP tool name to call")
    call_tool_parser.add_argument(
        "--arguments",
        default="{}",
        help="JSON object with tool arguments",
    )
    call_tool_parser.add_argument(
        "--url",
        default=None,
        help="MCP Streamable HTTP URL, for example http://127.0.0.1:8000/mcp",
    )
    call_tool_parser.add_argument(
        "--server-command",
        default=None,
        help="stdio server command. Defaults to the local AI Advent study server.",
    )
    call_tool_parser.add_argument(
        "--scheduler-file",
        default=None,
        help="JSON file used by local scheduler MCP tools",
    )
    call_tool_parser.add_argument(
        "--notes-dir",
        default=None,
        help="directory used by local note-saving MCP tools",
    )
    pipeline_parser = mcp_subparsers.add_parser(
        "run-pipeline",
        help="run the study note MCP tool pipeline",
    )
    pipeline_parser.add_argument("query", help="lesson query for the pipeline")
    pipeline_parser.add_argument(
        "--notes-dir",
        default=None,
        help="directory used by local note-saving MCP tools",
    )
    pipeline_parser.add_argument(
        "--server-command",
        default=None,
        help="stdio server command. Defaults to the local AI Advent study server.",
    )
    orchestrate_parser = mcp_subparsers.add_parser(
        "orchestrate",
        help="run a multi-server MCP orchestration flow",
    )
    orchestrate_parser.add_argument("query", help="lesson query for the flow")
    orchestrate_parser.add_argument(
        "--notes-dir",
        default=None,
        help="directory used by the notes MCP server",
    )
    orchestrate_parser.add_argument(
        "--scheduler-file",
        default=None,
        help="JSON file used by the scheduler MCP server",
    )
    orchestrate_parser.add_argument(
        "--remind-in",
        type=int,
        default=0,
        help="seconds before the review reminder is due",
    )

    worker_parser = subparsers.add_parser(
        "worker",
        help="run the MCP study scheduler worker",
    )
    worker_parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        help="seconds between scheduler checks",
    )
    worker_parser.add_argument(
        "--once",
        action="store_true",
        help="run one scheduler check and exit",
    )
    worker_parser.add_argument(
        "--scheduler-file",
        default=None,
        help="JSON file used to persist scheduler state",
    )
    worker_parser.add_argument(
        "--server-command",
        default=None,
        help="stdio server command. Defaults to the local AI Advent study server.",
    )

    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    load_dotenv()

    if args.command == "mcp":
        run_mcp_command(args)
        return

    if args.command == "worker":
        run_worker(args)
        return

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


def run_mcp_command(args: argparse.Namespace) -> None:
    if args.mcp_command == "list-tools":
        command = split_command(args.server_command) if args.server_command else None
        try:
            tools = list_tools_sync(
                url=args.url,
                command=command,
                env=build_mcp_env(
                    scheduler_file=args.scheduler_file,
                    notes_dir=args.notes_dir,
                ),
            )
        except (RuntimeError, ValueError) as error:
            print(f"MCP error: {error}", file=sys.stderr)
            raise SystemExit(1) from error

        print_mcp_tools(tools)
        return

    if args.mcp_command == "call-tool":
        command = split_command(args.server_command) if args.server_command else None
        try:
            arguments = parse_tool_arguments(args.arguments)
            result = call_tool_sync(
                args.tool_name,
                arguments,
                url=args.url,
                command=command,
                env=build_mcp_env(
                    scheduler_file=args.scheduler_file,
                    notes_dir=args.notes_dir,
                ),
            )
        except (RuntimeError, ValueError, json.JSONDecodeError) as error:
            print(f"MCP error: {error}", file=sys.stderr)
            raise SystemExit(1) from error

        print_mcp_tool_result(result)
        return

    if args.mcp_command == "run-pipeline":
        command = split_command(args.server_command) if args.server_command else None
        env = build_mcp_env(notes_dir=args.notes_dir)
        try:
            result = run_study_note_pipeline(
                args.query,
                caller=lambda tool_name, arguments: call_tool_sync(
                    tool_name,
                    arguments,
                    command=command,
                    env=env,
                ),
            )
        except (RuntimeError, ValueError) as error:
            print(f"Pipeline error: {error}", file=sys.stderr)
            raise SystemExit(1) from error

        print_pipeline_result(result)
        return

    if args.mcp_command == "orchestrate":
        orchestrator = build_default_orchestrator(
            scheduler_file=args.scheduler_file,
            notes_dir=args.notes_dir,
        )
        try:
            registered_tools = orchestrator.discover_tools()
            result = run_study_orchestration_flow(
                args.query,
                reminder_seconds=args.remind_in,
                orchestrator=orchestrator,
            )
        except (RuntimeError, ValueError) as error:
            print(f"Orchestration error: {error}", file=sys.stderr)
            raise SystemExit(1) from error

        print_registered_tools(registered_tools)
        print_orchestration_result(result)
        return

    if args.mcp_command is None:
        print("Usage: ai-advent mcp list-tools|call-tool|run-pipeline|orchestrate ...")
        raise SystemExit(2)

    print("Usage: ai-advent mcp list-tools|call-tool|run-pipeline|orchestrate ...")
    raise SystemExit(2)


def run_worker(args: argparse.Namespace) -> None:
    if args.interval <= 0:
        print("Worker error: --interval must be greater than zero.", file=sys.stderr)
        raise SystemExit(1)

    command = split_command(args.server_command) if args.server_command else None
    scheduler_file = args.scheduler_file or os.getenv(
        SCHEDULER_FILE_ENV,
        DEFAULT_SCHEDULER_FILE,
    )
    print(f"AI Advent worker (scheduler: {scheduler_file}, interval: {args.interval}s)")
    print("Press Ctrl+C to stop.\n")

    while True:
        try:
            result = call_tool_sync(
                "run_due_tasks",
                command=command,
                env=build_mcp_env(scheduler_file),
            )
        except (RuntimeError, ValueError) as error:
            print(f"MCP error: {error}", file=sys.stderr)
            if args.once:
                raise SystemExit(1) from error
        else:
            print_mcp_tool_result(result)

        if args.once:
            return
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            print("\nWorker stopped.")
            return


def build_mcp_env(
    scheduler_file: str | None = None,
    notes_dir: str | None = None,
) -> dict[str, str]:
    env: dict[str, str] = {}
    scheduler_path = scheduler_file or os.getenv(SCHEDULER_FILE_ENV)
    if scheduler_path is not None:
        env[SCHEDULER_FILE_ENV] = scheduler_path
    notes_path = notes_dir or os.getenv(NOTES_DIR_ENV)
    if notes_path is not None:
        env[NOTES_DIR_ENV] = notes_path
    return env


def parse_tool_arguments(raw_arguments: str) -> dict[str, object]:
    arguments = json.loads(raw_arguments)
    if not isinstance(arguments, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return arguments


def print_mcp_tools(tools: list[McpTool]) -> None:
    print(f"MCP tools: {len(tools)}")
    if not tools:
        print("  (none)")
        return

    for tool in tools:
        display_name = tool.title or tool.name
        print(f"- {tool.name}: {display_name}")
        if tool.description:
            print(f"  description: {tool.description}")
        print(f"  input_schema: {tool.input_schema}")


def print_mcp_tool_result(result: McpToolResult) -> None:
    print(f"MCP tool result: {result.tool_name}")
    print(f"  is_error: {result.is_error}")
    if result.structured_content is not None:
        print(f"  structured_content: {result.structured_content}")
    if result.content_text:
        print("  content:")
        for line in result.content_text.splitlines():
            print(f"    {line}")


def print_pipeline_result(result: PipelineResult) -> None:
    print(f"MCP pipeline: study note for {result.query}")
    for index, step in enumerate(result.steps, start=1):
        print(f"{index}. {step.tool_name}")
        print(f"   arguments: {step.arguments}")
        print(f"   is_error: {step.result.is_error}")
    print(f"Saved note: {result.saved_path}")


def print_registered_tools(registered_tools: list[RegisteredTool]) -> None:
    print("Registered MCP servers:")
    by_server: dict[str, list[str]] = {}
    for registered_tool in registered_tools:
        by_server.setdefault(registered_tool.server_name, []).append(
            registered_tool.tool.name,
        )
    for server_name, tool_names in sorted(by_server.items()):
        print(f"- {server_name}: {', '.join(sorted(tool_names))}")


def print_orchestration_result(result: OrchestrationResult) -> None:
    print(f"MCP orchestration: study flow for {result.query}")
    for index, step in enumerate(result.steps, start=1):
        print(f"{index}. {step.server_name}.{step.tool_name}")
        print(f"   arguments: {step.arguments}")
        print(f"   is_error: {step.result.is_error}")
    print(f"Saved note: {result.saved_path}")
    print(f"Reminder id: {result.reminder_id or '(not returned)'}")
    print(f"Scheduler summary: {result.scheduler_summary}")


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
