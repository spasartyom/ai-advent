# AI Advent Architecture

This repository is a learning project for an AI agents challenge.

Current state: week 2 is complete and week 3 has started. Week 1 is archived in
Git tags and should not shape new implementation unless we explicitly need to
inspect old exercises. Week 3 develops the CLI agent into a Study Coach Agent.

## Challenge State

Week 1 is preserved in tags:

- `w1_d1` - basic chat CLI
- `w1_d2` - response formatting comparison
- `w1_d3` - reasoning strategy comparison
- `w1_d4` - temperature comparison
- `w1_d5` - model comparison

Week 2 is complete:

- `w2_d1` / Day 6 - first standalone agent
- `w2_d2` / Day 7 - JSON-backed persistent context
- `w2_d3` / Day 8 - API usage token reporting
- `w2_d4` / Day 9 - summary-based context compression
- `w2_d5` / Day 10 - sliding window, sticky facts, and branching strategies

Week 3 has started:

- `w3_d1` / Day 11 - explicit memory layers for short-term, working, and
  long-term memory
- `w3_d2` / Day 12 - user profile personalization connected to every request
- `w3_d3` / Day 13 - formal task state with stage, current step, expected action, and pause/resume
- `w3_d4` / Day 14 - separate invariants that are included in every request and can trigger refusal
- `w3_d5` / Day 15 - controlled task lifecycle transitions

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
- support `/paste` and `/send` for multiline user messages;
- support `/memory ...` commands for explicit working and long-term memory
  updates;
- support `/profile ...` commands for explicit user profile personalization;
- support `/task ...` commands for explicit task state updates;
- support `/invariant ...` commands for explicit invariant management;
- expose `--context-strategy full|summary|sliding-window|facts|branch`,
  `--keep-last`, and `--branch`;
- support `/checkpoint` and `/branch ...` commands for branching experiments.

This file should stay thin. Avoid putting agent logic, memory logic, token
accounting, or context strategies here.

### `ai_advent/agent.py`

Current core agent.

Responsibilities:

- own the in-memory dialog state for the running process;
- load initial messages from memory when configured;
- load summary from memory when configured;
- load facts, branches, and checkpoints from memory when configured;
- load working and long-term memory layers when configured;
- load the user profile when configured;
- load task state when configured;
- load invariants when configured;
- accept a user message through `Agent.run_turn`;
- prepare the message list through the selected context strategy;
- prepend the user profile to each request when present;
- prepend task state to each request when present;
- prepend invariants to each request when present;
- prepend explicit working and long-term memory to each request when present;
- refuse explicit requests to violate a named invariant before calling the model;
- enforce controlled task state transitions;
- call the low-level chat helper;
- append the assistant response;
- compress old context when the selected strategy requires it;
- save messages after a successful turn when memory is configured;
- manage branch checkpoints and branch switching through methods on `Agent`;
- return an `AgentResponse` with response text, API usage metadata, and
  `TokenReport`.

The agent is intentionally a separate entity from the CLI and from the raw API
client. Future work should evolve this layer rather than rebuilding the chat
loop.

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

- `JsonFileMemory` stores messages and state in a JSON file with the shape
  `{"summary": "...", "facts": {...}, "working_memory": {...},
  "long_term_memory": {...}, "user_profile": {...}, "task_state": {...}, "invariants": {...}, "messages": [...], "branches": {...},
  "checkpoints": {...}, "current_branch": "main"}`;
- missing files load as empty history;
- old files without `summary` load with an empty summary;
- old files without facts/working memory/long-term memory/user profile/task state/invariants/branches/checkpoints
  load empty values;
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
- `SlidingWindowContextStrategy` keeps only the latest N messages and discards
  older raw messages.
- `StickyFactsContextStrategy` updates a key-value facts block with an
  additional Chat Completions request and sends facts + latest N messages.

Summary compression uses an additional Chat Completions request to update the
stored summary from older messages. The main CLI token report shows the usage of
the user-facing response call.

Branching is managed by `Agent` rather than by a separate context strategy:
`--context-strategy branch` currently uses full context for the active branch,
while `/checkpoint` and `/branch ...` commands switch the message history that
the agent reads and writes.

### `ai_advent/task.py`

Formal task state for Study Coach sessions.

Responsibilities:

- define `TaskState`;
- validate allowed task stages: `idle`, `planning`, `execution`, `validation`, `done`;
- define allowed lifecycle transitions;
- require explicit plan approval before `planning -> execution`;
- reject final `done` before `validation`;
- convert task state to and from the JSON memory shape;

### Week 3 Memory Layers

Day 11 uses a Study Coach framing:

- short-term memory is the current dialog (`messages`);
- working memory is explicit key-value state for the current learning task;
- long-term memory is explicit key-value state for stable user/assistant
  knowledge.

Working and long-term memory are updated only by explicit calls or CLI commands,
not by automatic extraction from arbitrary chat text. The CLI commands are:

- `/memory show [short|working|long|all]`
- `/memory set working KEY VALUE`
- `/memory set long KEY VALUE`
- `/memory forget working|long KEY`

### Week 3 Personalization

Day 12 stores personalization in `user_profile`, separate from dialog history and general long-term memory.

The profile is explicit key-value data such as `name`, `language`, `answer_style`, `format`, and `constraints`.

The profile is prepended to every model request as a dedicated system message, so the Study Coach Agent can automatically adapt language, style, format, and constraints.

The CLI commands are:

- `/profile show`
- `/profile set KEY VALUE`
- `/profile forget KEY`

### Week 3 Task State

Day 13 stores task state in `task_state`, separate from dialog history, profile, and memory layers.

Task state includes `stage`, `title`, `current_step`, `expected_action`, and `paused`.

The task state is prepended to every model request as a dedicated system message, so the Study Coach Agent can continue after a pause without repeating setup.

The CLI commands are:

- `/task status`
- `/task start TITLE`
- `/task stage idle|planning|execution|validation|done`
- `/task approve`
- `/task step CURRENT_STEP`
- `/task expect EXPECTED_ACTION`
- `/task title TITLE`
- `/task pause`
- `/task resume`
- `/task done`

### Week 3 Controlled Transitions

Day 15 enforces task lifecycle transitions in code.

Allowed transitions are `idle -> planning`, `planning -> execution` through `/task approve`, `execution -> validation`, `validation -> execution`, `validation -> done`, and `done -> planning`.

Direct `planning -> execution` through `/task stage execution` is rejected because the plan has not been approved.

Direct `execution -> done` is rejected because final completion requires validation first.

Pause and resume preserve the current stage and do not bypass transition rules.

### Week 3 Invariants

Day 14 stores invariants in `invariants`, separate from dialog history, profile, memory layers, and task state.

Invariants are explicit key-value data where the key is a stable id and the value is the rule text.

The invariants are prepended to every model request as a dedicated system message, so the Study Coach Agent must treat them as hard constraints.

The agent also refuses explicit requests to violate a named invariant before calling the model.

The CLI commands are:

- `/invariant show`
- `/invariant add ID TEXT`
- `/invariant remove ID`

### `tests/`

Tests use fake Chat Completions APIs rather than real network calls.

Current tests cover:

- `Agent` request/response behavior;
- in-process dialog history;
- JSON-backed persistent memory;
- explicit working and long-term memory layers;
- explicit user profile personalization;
- formal task state with pause/resume;
- separate invariants and explicit invariant refusal;
- controlled task lifecycle transitions;
- API usage token reporting;
- full and summary context strategies;
- sliding window, sticky facts, and branching context workflows;
- multiline CLI paste mode;
- defensive copying of messages;
- usage metadata propagation;
- CLI command parsing and terminal loop behavior;
- low-level `ChatSession` behavior.

Run tests with:

```bash
python3 -m unittest discover -s tests
```

## Architecture Direction

Do not reintroduce the week 1 comparison commands into active code. They live in
tags now.

Week 2 implementation is finished. Preserve its shape unless a future week
explicitly needs a refactor:

- CLI remains orchestration and terminal I/O.
- `Agent` owns turn execution, memory state, and branch operations.
- `chat.py` remains the provider-agnostic Chat Completions adapter.
- `context.py` remains the place for context-management strategies.
- `memory.py` remains the place for persistent state shape and validation.

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

## Writing Notes

When editing Markdown files, do not hard-wrap lines in the middle of a sentence. Prefer one sentence or list item per line unless an existing table, code block, or quoted text requires a different shape.
