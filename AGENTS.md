# AI Advent Architecture

This repository is a learning project for an AI agents challenge.

Current focus: week 2. Week 1 is archived in Git tags and should not shape new
implementation unless we explicitly need to inspect old exercises.

## Challenge State

Week 1 is preserved in tags:

- `w1_d1` - basic chat CLI
- `w1_d2` - response formatting comparison
- `w1_d3` - reasoning strategy comparison
- `w1_d4` - temperature comparison
- `w1_d5` - model comparison

Current branch work starts week 2:

- `w2_d1` / Day 6 - first standalone agent
- `w2_d2` / Day 7 - JSON-backed persistent context
- `w2_d3` / Day 8 - API usage token reporting
- `w2_d4` / Day 9 - summary-based context compression

## Current Product Shape

The active product is a CLI agent:

```bash
ai-advent agent
```

The default command also starts the same agent:

```bash
ai-advent
```

The agent talks to an OpenAI-compatible Chat Completions API. Provider settings
come from environment variables, usually loaded from `.env`:

- `AI_ADVENT_API_KEY`
- `AI_ADVENT_MODEL`
- `AI_ADVENT_BASE_URL`
- `AI_ADVENT_MEMORY_FILE`

Fallback OpenAI-style names are also supported:

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_BASE_URL`

## Modules

### `ai_advent/cli.py`

CLI entry point.

Responsibilities:

- parse commands;
- load environment variables;
- create the OpenAI-compatible client;
- create `Agent`;
- run the interactive terminal loop;
- support `/paste` and `/send` for multiline user messages.
- expose `--context-strategy full|summary` and `--keep-last`.

This file should stay thin. Avoid putting agent logic, memory logic, token
accounting, or context strategies here.

### `ai_advent/agent.py`

Current week 2 core.

Responsibilities:

- own the in-memory dialog state for the running process;
- load initial messages from memory when configured;
- load summary from memory when configured;
- accept a user message through `Agent.run_turn`;
- prepare the message list through the selected context strategy;
- call the low-level chat helper;
- append the assistant response;
- compress old context when the selected strategy requires it;
- save messages after a successful turn when memory is configured;
- return an `AgentResponse` with response text, API usage metadata, and
  `TokenReport`.

The agent is intentionally a separate entity from the CLI and from the raw API
client. Future week 2 tasks should evolve this layer rather than rebuilding the
chat loop.

### `ai_advent/chat.py`

Low-level Chat Completions adapter.

Responsibilities:

- define the `ChatCompletionsAPI` protocol;
- define `ChatResponse`;
- normalize response text from SDK response objects;
- normalize usage metadata;
- provide `create_chat_completion`;
- keep the old `ChatSession` compatibility wrapper.

This module should remain provider-agnostic and should not know about agent
memory or context strategy decisions.

### `ai_advent/memory.py`

Persistent memory adapters.

Current implementation:

- `JsonFileMemory` stores messages in a JSON file with the shape
  `{"summary": "...", "messages": [...]}`;
- missing files load as empty history;
- old files without `summary` load with an empty summary;
- invalid message objects raise `ValueError`;
- parent directories are created automatically on save.

Default CLI memory file:

```text
.ai-advent/agent-memory.json
```

### `ai_advent/tokens.py`

Token report types.

Current implementation does not estimate tokens locally and does not calculate
cost. It reports usage values returned by the model API.

Responsibilities:

- define `TokenReport`;
- carry `prompt_tokens`, `completion_tokens`, and `total_tokens` returned by
  the API.

### `ai_advent/context.py`

Context management strategies.

Current implementations:

- `FullContextStrategy` sends all active messages.
- `SummaryContextStrategy` prepends a summary message and keeps only the latest
  N messages as raw chat history.

Summary compression uses an additional Chat Completions request to update the
stored summary from older messages. The main CLI token report shows the usage of
the user-facing response call.

### `tests/`

Tests use fake Chat Completions APIs rather than real network calls.

Current tests cover:

- `Agent` request/response behavior;
- in-process dialog history;
- JSON-backed persistent memory;
- API usage token reporting;
- full and summary context strategies;
- multiline CLI paste mode;
- defensive copying of messages;
- usage metadata propagation;
- CLI command parsing and terminal loop behavior;
- low-level `ChatSession` behavior.

Run tests with:

```bash
python3 -m unittest discover -s tests
```

## Architecture Direction For Week 2

Do not reintroduce the week 1 comparison commands into active code. They live in
tags now.

Expected evolution:

- Day 10: multiple context strategies:
  - sliding window;
  - sticky facts / key-value memory;
  - branching dialog history.

Likely future modules:

- `ai_advent/context.py` - context strategy interfaces and implementations.

Preferred design:

- keep `cli.py` as orchestration only;
- keep `chat.py` as a low-level API adapter;
- keep `Agent` as the main user-facing domain object;
- introduce small strategy or storage classes only when the next task needs
  them;
- keep tests offline with fake API objects.

## Git Notes

The working tree may contain local artifacts unrelated to the challenge, such as
custom skills. Do not remove or rewrite unrelated files unless the user asks.

`.env` is ignored and should never be committed.
