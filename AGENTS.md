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
- run the interactive terminal loop.

This file should stay thin. Avoid putting agent logic, memory logic, token
accounting, or context strategies here.

### `ai_advent/agent.py`

Current week 2 core.

Responsibilities:

- own the in-memory dialog state for the running process;
- accept a user message through `Agent.run_turn`;
- prepare the message list for the LLM call;
- call the low-level chat helper;
- append the assistant response;
- return an `AgentResponse` with response text and usage metadata.

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

### `tests/`

Tests use fake Chat Completions APIs rather than real network calls.

Current tests cover:

- `Agent` request/response behavior;
- in-process dialog history;
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

- Day 7: persistent memory, likely JSON first.
- Day 8: token accounting for current request, history, and model response.
- Day 9: summary-based context compression.
- Day 10: multiple context strategies:
  - sliding window;
  - sticky facts / key-value memory;
  - branching dialog history.

Likely future modules:

- `ai_advent/memory.py` - load/save dialog state.
- `ai_advent/tokens.py` - token estimation and cost reporting.
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
