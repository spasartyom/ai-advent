# AI Advent Architecture

This repository is a learning project for an AI agents challenge.

Current state: week 5 has started and builds RAG capabilities on top of the Study Coach Agent plus MCP tooling.
Week 1 is archived in Git tags and should not shape new implementation unless we explicitly need to inspect old exercises.
Week 3 developed the CLI agent into a Study Coach Agent.
Week 4 connected the Study Coach to MCP tools, scheduled background work, tool pipelines, and multi-server orchestration.
Week 5 adds local document indexing and the first RAG question-answer flow.

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

Week 3 is complete:

- `w3_d1` / Day 11 - explicit memory layers for short-term, working, and
  long-term memory
- `w3_d2` / Day 12 - user profile personalization connected to every request
- `w3_d3` / Day 13 - formal task state with stage, current step, expected action, and pause/resume
- `w3_d4` / Day 14 - separate invariants that are included in every request and can trigger refusal
- `w3_d5` / Day 15 - controlled task lifecycle transitions

Week 4 is complete:

- `w4_d1` / Day 16 - MCP client and local Study Coach MCP server tool discovery
- `w4_d2` / Day 17 - first real MCP tool call through `get_lesson` and `/lesson`
- `w4_d3` / Day 18 - scheduler tools and background worker for study reminders
- `w4_d4` / Day 19 - composed MCP tool pipeline for searching, summarizing, and saving notes
- `w4_d5` / Day 20 - multi-server MCP orchestration across lessons, notes, and scheduler servers

Week 5 has started:

- `w5_d1` / Day 21 - local document indexing with chunking, embeddings, JSON vector index, metadata, and chunking strategy comparison
- `w5_d2` / Day 22 - first RAG request with retrieval, RAG prompt construction, with-RAG/without-RAG comparison, and 10 control questions

## Current Product Shape

The active product is a CLI agent:

```bash
ai-advent agent
```

The default command also starts the same agent:

```bash
ai-advent
```

Week 4 also adds MCP and worker commands:

```bash
ai-advent mcp list-tools
ai-advent mcp call-tool get_lesson --arguments '{"topic":"mcp"}'
ai-advent mcp run-pipeline mcp
ai-advent mcp orchestrate mcp
ai-advent worker --once
```

Week 5 also adds document index and RAG commands:

```bash
ai-advent index build README.md AGENTS.md ai_advent tests --embedding-provider ollama --embedding-model nomic-embed-text
ai-advent rag ask "Какие команды CLI управляют task state?"
ai-advent rag compare "Какие команды CLI управляют task state?"
ai-advent rag eval-questions
```

The agent talks to an OpenAI-compatible Chat Completions API. Provider settings
come from environment variables, usually loaded from `.env`:

- `AI_ADVENT_API_KEY`
- `AI_ADVENT_MODEL`
- `AI_ADVENT_BASE_URL`
- `AI_ADVENT_MEMORY_FILE`
- `AI_ADVENT_SCHEDULER_FILE`
- `AI_ADVENT_NOTES_DIR`
- `AI_ADVENT_EMBEDDING_MODEL`
- `AI_ADVENT_OLLAMA_BASE_URL`

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
- support MCP-backed interactive commands: `/lesson TOPIC`, `/remind DUE_IN_SECONDS TITLE`, `/reminders`, `/note QUERY`, and `/study-flow QUERY`;
- expose `--context-strategy full|summary|sliding-window|facts|branch`,
  `--keep-last`, and `--branch`;
- support `/checkpoint` and `/branch ...` commands for branching experiments;
- expose `ai-advent mcp list-tools`, `call-tool`, `run-pipeline`, and `orchestrate`;
- expose `ai-advent worker` for running due scheduler tasks;
- expose `ai-advent index build` for document indexing;
- expose `ai-advent rag ask`, `compare`, and `eval-questions` for first RAG workflows.

This file should stay thin. Avoid putting agent logic, memory logic, token
accounting, context strategies, MCP transport logic, scheduler persistence, pipeline logic, orchestration logic, document loading, chunking, embedding provider logic, vector search, or RAG prompt construction here.

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

### `ai_advent/mcp_client.py`

Low-level MCP client adapter.

Responsibilities:

- define `McpTool` and `McpToolResult`;
- provide synchronous wrappers over async MCP client calls;
- connect either to a Streamable HTTP MCP URL or to a stdio server command;
- default to the local Study Coach server command: `python -m ai_advent.mcp_servers.study`;
- normalize MCP tool metadata and tool result content;
- stay independent from `Agent` memory, task state, and context strategies.

### `ai_advent/study_content.py`

Deterministic local study content and note helpers used by MCP servers.

Responsibilities:

- store small local lesson fixtures;
- list, get, and search lesson content;
- create deterministic study-note summaries without LLM calls;
- save Markdown notes under `AI_ADVENT_NOTES_DIR` or `.ai-advent/notes`;
- provide `slugify` for note filenames.

### `ai_advent/scheduler.py`

JSON-backed study reminder scheduler.

Responsibilities:

- create reminders with pending/completed state;
- list reminders grouped by status;
- complete due reminders through `run_due_tasks`;
- persist scheduler state in `{"reminders": [...], "runs": [...]}`;
- default to `.ai-advent/study-scheduler.json`;
- validate storage shape and create parent directories on save.

### `ai_advent/pipeline.py`

Single-server MCP tool pipeline for study notes.

Current flow:

```text
search_lessons -> summarize_note -> save_note
```

Responsibilities:

- validate the query;
- call MCP tools through an injected caller or the default `call_tool_sync`;
- pass structured output from one tool to the next;
- raise a clear runtime error when any MCP tool returns `is_error`;
- return `PipelineResult` with step details, saved note path, and summary.

### `ai_advent/orchestration.py`

Multi-server MCP orchestration layer.

Default servers:

- `lessons` - `python -m ai_advent.mcp_servers.lessons`
- `notes` - `python -m ai_advent.mcp_servers.notes`
- `scheduler` - `python -m ai_advent.mcp_servers.scheduler`

Current flow:

```text
lessons.search_lessons -> notes.summarize_note -> notes.save_note -> scheduler.create_reminder -> scheduler.run_due_tasks
```

Responsibilities:

- define server configs and registered tool metadata;
- discover tools across all registered MCP servers;
- route tool calls by tool name to the first server that exposes that name;
- support environment overrides for `AI_ADVENT_NOTES_DIR` and `AI_ADVENT_SCHEDULER_FILE`;
- return `OrchestrationResult` with step details, saved note path, reminder id, summary, and scheduler summary.

### `ai_advent/documents.py`

Document loading for local RAG indexing.

Responsibilities:

- load supported text documents from files and directories;
- support Markdown, Python, plain text, and reStructuredText inputs;
- skip generated/cache directories such as `.git`, `.venv`, and `__pycache__`;
- return `Document` values with source, title, and text.

### `ai_advent/chunking.py`

Document chunking strategies.

Current implementations:

- fixed-size chunking by character window with configurable overlap;
- structure-aware chunking by Markdown headings and top-level Python `def`/`class` blocks;
- chunking comparison stats with chunk count, average size, minimum size, and maximum size.

Responsibilities:

- create chunks with `chunk_id`, `source`, `title`, `section`, and `text`;
- keep chunking deterministic and independent from embeddings or model calls;
- preserve metadata needed for retrieval debugging, sources, and later citations.

### `ai_advent/embeddings.py`

Embedding provider adapters.

Current providers:

- OpenAI-compatible embeddings API through the shared `EmbeddingsAPI` protocol;
- deterministic `HashEmbeddingAPI` for offline tests and demos;
- local `OllamaEmbeddingAPI` through Ollama `/api/embed`.

Responsibilities:

- normalize embedding responses to lists of float vectors;
- keep provider-specific HTTP or SDK details outside indexing and RAG code;
- support offline tests without network calls or external services.

### `ai_advent/vector_index.py`

JSON-backed local vector index.

Responsibilities:

- store indexed chunks with embeddings and metadata;
- save and load JSON indexes;
- compute cosine similarity;
- return top-K `SearchResult` values for a query embedding.

### `ai_advent/indexing.py`

Document indexing pipeline for Day 21.

Responsibilities:

- load documents from user-provided paths;
- compare fixed-size and structure-aware chunking strategies;
- generate embeddings for selected chunks;
- save the local JSON vector index;
- return `IndexBuildResult` with index, document count, total characters, and saved path.

### `ai_advent/rag.py`

First RAG request pipeline for Day 22.

Responsibilities:

- embed the user question;
- retrieve relevant chunks from `VectorIndex`;
- build a RAG prompt that includes question, retrieved context, scores, and chunk metadata;
- call the low-level chat helper for a with-RAG answer;
- call the low-level chat helper for a without-RAG answer in comparison mode;
- return structured `RagAnswer` and `RagComparison` values.

### `ai_advent/rag_eval.py`

Control-question fixture for RAG evaluation.

Responsibilities:

- define 10 control questions over the project knowledge base;
- record expected answer content and expected sources for each question;
- format the questions for `ai-advent rag eval-questions`;
- stay deterministic and offline.

### `ai_advent/mcp_servers/`

Local MCP servers for Study Coach tools.

Current servers:

- `study.py` exposes all local tools in one stdio MCP server for simple demos and default MCP CLI usage.
- `lessons.py` exposes `list_lessons`, `get_lesson`, and `search_lessons`.
- `notes.py` exposes `summarize_note` and `save_note`.
- `scheduler.py` exposes `create_reminder`, `list_reminders`, and `run_due_tasks`.

Keep tool implementations thin.
Domain behavior belongs in `study_content.py` and `scheduler.py`, while server files should mainly adapt functions to MCP.

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

### Week 4 MCP Tools

Day 16 introduces MCP tool discovery.

CLI commands:

- `ai-advent mcp list-tools`
- `ai-advent mcp list-tools --url http://127.0.0.1:8000/mcp`
- `ai-advent mcp list-tools --server-command "python path/to/server.py"`

The default stdio server is:

```bash
python -m ai_advent.mcp_servers.study
```

Day 17 introduces direct MCP tool calls and the first agent-facing MCP command.

CLI commands:

- `ai-advent mcp call-tool get_lesson --arguments '{"topic":"mcp"}'`
- `/lesson TOPIC`

`/lesson` calls `get_lesson`, prints the MCP result, then passes the result into `Agent.run_turn` as study material.

### Week 4 Scheduler

Day 18 adds scheduler tools exposed through MCP:

- `create_reminder(title: str, due_in_seconds: int = 0, note: str = "") -> dict`
- `list_reminders() -> dict`
- `run_due_tasks() -> dict`

Default scheduler file:

```text
.ai-advent/study-scheduler.json
```

CLI commands:

- `ai-advent mcp call-tool create_reminder --arguments '{"title":"Review MCP","due_in_seconds":0}'`
- `ai-advent worker --once`
- `ai-advent worker --interval 60`
- `/remind DUE_IN_SECONDS TITLE`
- `/reminders`

The worker calls the MCP `run_due_tasks` tool, prints the result, and either exits with `--once` or sleeps for `--interval` seconds.

### Week 4 Pipelines And Orchestration

Day 19 adds a single-server study-note pipeline:

```text
search_lessons -> summarize_note -> save_note
```

CLI commands:

- `ai-advent mcp run-pipeline QUERY`
- `ai-advent mcp run-pipeline QUERY --notes-dir PATH`
- `/note QUERY`

Day 20 adds multi-server orchestration:

```text
lessons.search_lessons -> notes.summarize_note -> notes.save_note -> scheduler.create_reminder -> scheduler.run_due_tasks
```

CLI commands:

- `ai-advent mcp orchestrate QUERY`
- `ai-advent mcp orchestrate QUERY --notes-dir PATH --scheduler-file PATH --remind-in SECONDS`
- `/study-flow QUERY`

Keep deterministic tool plumbing testable without real network calls or real model calls.
Use injected callers or fake orchestrators in tests.

### Week 5 Document Indexing

Day 21 adds a local document indexing pipeline.

CLI commands:

- `ai-advent index build PATH [PATH ...]`
- `ai-advent index build README.md AGENTS.md ai_advent tests --offline-embeddings`
- `ai-advent index build README.md AGENTS.md ai_advent tests --embedding-provider ollama --embedding-model nomic-embed-text`
- `ai-advent index build README.md AGENTS.md ai_advent tests --embedding-provider openai --embedding-model text-embedding-3-small`

Supported embedding providers:

- `openai` - OpenAI-compatible embeddings endpoint through the OpenAI SDK;
- `ollama` - local Ollama `/api/embed`, usually with `nomic-embed-text`;
- `offline` - deterministic local hash embeddings for tests and demos.

Default index file:

```text
.ai-advent/document-index.json
```

The JSON index stores:

- embedding model;
- selected chunking strategy;
- chunking comparison stats;
- chunks with `chunk_id`, `source`, `title`, `section`, `text`, and `embedding`.

Day 21 compares two chunking strategies:

- `fixed` - fixed character windows with overlap;
- `structure` - Markdown heading and Python top-level symbol chunking, with max-size splitting.

### Week 5 First RAG Request

Day 22 adds the first question-answer RAG flow.

CLI commands:

- `ai-advent rag ask QUESTION`
- `ai-advent rag compare QUESTION`
- `ai-advent rag eval-questions`

Current flow:

```text
question -> query embedding -> vector search top-K -> RAG prompt -> Chat Completions answer
```

`rag compare` makes two chat calls:

1. without RAG, using only the question and a Study Coach system prompt;
2. with RAG, using retrieved chunks and chunk metadata in the prompt.

The RAG command prints retrieved chunks with similarity score, source, section, and chunk id.
The command currently requires a Chat Completions API key for answer generation, even when query embeddings come from Ollama or offline hash embeddings.

Day 22 includes 10 control questions in `rag_eval.py`.
Each question records the expected answer content and expected sources for manual quality comparison.

Keep Day 22 RAG explicit through CLI commands and `rag.py`.
Do not hide automatic RAG retrieval inside `Agent.run_turn` until a future day explicitly requires a production-like chat integration.

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
- MCP client command parsing and MCP output formatting;
- MCP-backed `/lesson`, `/remind`, `/reminders`, `/note`, and `/study-flow` command handling;
- scheduler storage and due-task behavior;
- study note pipeline behavior;
- multi-server orchestration behavior;
- document loading, chunking, embeddings adapters, vector index persistence, and index search;
- document index CLI command parsing;
- RAG prompt construction, retrieval, with-RAG answers, and with/without-RAG comparison;
- RAG CLI command parsing and control-question formatting;
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

Week 2, Week 3, and Week 4 implementations are finished. Preserve their shape unless a future week explicitly needs a refactor:

- CLI remains orchestration and terminal I/O.
- `Agent` owns turn execution, memory state, and branch operations.
- `chat.py` remains the provider-agnostic Chat Completions adapter.
- `context.py` remains the place for context-management strategies.
- `memory.py` remains the place for persistent state shape and validation.
- MCP transport belongs in `mcp_client.py`.
- MCP server files stay as thin adapters.
- Deterministic study-domain behavior belongs in `study_content.py` and `scheduler.py`.
- Tool composition belongs in `pipeline.py`.
- Multi-server coordination belongs in `orchestration.py`.
- Document loading belongs in `documents.py`.
- Chunking belongs in `chunking.py`.
- Embedding provider adapters belong in `embeddings.py`.
- Vector storage and similarity search belong in `vector_index.py`.
- Document index construction belongs in `indexing.py`.
- RAG retrieval, prompt construction, and with/without-RAG comparison belong in `rag.py`.
- RAG control questions belong in `rag_eval.py`.

Preferred design:

- keep `cli.py` as orchestration only;
- keep `chat.py` as a low-level API adapter;
- keep `Agent` as the main user-facing domain object;
- keep MCP tool calls explicit through CLI commands and MCP helpers rather than hiding automatic tool selection inside `Agent`;
- keep RAG retrieval explicit through CLI commands and RAG helpers until a future day explicitly requires integrating it into the interactive agent loop;
- introduce small strategy or storage classes only when the next task needs
  them;
- keep tests offline with fake API objects.

## Git Notes

The working tree may contain local artifacts unrelated to the challenge, such as
custom skills. Do not remove or rewrite unrelated files unless the user asks.

`.env` is ignored and should never be committed.

## Writing Notes

When editing Markdown files, do not hard-wrap lines in the middle of a sentence. Prefer one sentence or list item per line unless an existing table, code block, or quoted text requires a different shape.
