# AI Advent

CLI-проект для челленджа по изучению AI-агентов. Вторая неделя построила базового CLI-агента с памятью, подсчетом токенов и стратегиями управления контекстом. Третья неделя развила его в Study Coach Agent: учебного ассистента с явной моделью памяти и управляемым состоянием задач. Четвертая неделя подключила MCP-инструменты, scheduler, pipeline и orchestration. Пятая неделя начинает RAG-направление с локальной индексации документов.

Задания первой недели сохранены в Git-тегах:

- `w1_d1` - базовый чат CLI;
- `w1_d2` - форматирование ответа;
- `w1_d3` - разные способы рассуждения;
- `w1_d4` - температура;
- `w1_d5` - сравнение моделей.

## Установка

Требуется Python 3.10 или новее.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Создайте `.env` на основе примера и укажите API-ключ:

```bash
cp .env.example .env
```

```dotenv
AI_ADVENT_API_KEY=sk-your-api-key
AI_ADVENT_MODEL=gpt-5.6-luna
AI_ADVENT_BASE_URL=https://api.openai.com/v1
AI_ADVENT_MEMORY_FILE=.ai-advent/agent-memory.json
AI_ADVENT_EMBEDDING_MODEL=text-embedding-3-small
AI_ADVENT_OLLAMA_BASE_URL=http://localhost:11434
```

Файл `.env` исключен из Git и не попадет в репозиторий.

Модель и провайдера можно менять через `.env`:

```dotenv
AI_ADVENT_API_KEY=your-provider-api-key
AI_ADVENT_MODEL=deepseek-chat
AI_ADVENT_BASE_URL=https://api.deepseek.com
```

Для OpenAI-compatible API используются те же вызовы SDK: меняются только ключ, модель и `base_url`.

## Неделя 2

### День 6: первый агент

Агент реализован как отдельная сущность `Agent` в
`ai_advent/agent.py`. Он принимает сообщение пользователя, сам готовит историю диалога, вызывает LLM через Chat Completions API и возвращает структурированный ответ.

Запуск:

```bash
ai-advent agent
```

Команду `agent` можно не указывать: обычный запуск тоже открывает агента.

```bash
ai-advent
```

Введите сообщение и нажмите Enter. Для завершения используйте `/exit`, `/quit` или `Ctrl+D`.
Для многострочного ввода используйте `/paste`, вставьте текст и завершите
строкой `/send`.

```text
AI Advent agent (model: gpt-5.6-luna)
Type /exit or /quit to stop. Use /paste, then /send for multiline input.

You: What is an agent?
Assistant: An agent is a program that can use an LLM plus state and logic...
```

### День 7: сохранение контекста

Агент сохраняет историю сообщений в JSON-файл и загружает ее при следующем
запуске. По умолчанию используется файл:

```text
.ai-advent/agent-memory.json
```

Запуск с памятью по умолчанию:

```bash
ai-advent agent
```

Можно указать отдельный файл памяти:

```bash
ai-advent agent --memory-file .ai-advent/demo-memory.json
```

Проверка вручную:

1. Запустите `ai-advent agent`.
2. Напишите факт, например: `Меня зовут Антон, я изучаю AI-агентов`.
3. Завершите чат через `/exit`.
4. Запустите `ai-advent agent` снова.
5. Спросите: `Что ты помнишь обо мне?`

При втором запуске агент отправит в LLM историю из JSON-файла и продолжит
диалог с сохраненным контекстом.

### День 8: работа с токенами

Агент показывает данные о токенах, которые возвращает API модели:

- `prompt_tokens` - токены запроса;
- `completion_tokens` - токены ответа;
- `total_tokens` - сумма токенов запроса и ответа.

Включить вывод токенов:

```bash
ai-advent agent --show-tokens
```

Для сравнения короткого и длинного диалога удобно использовать разные файлы
памяти:

```bash
ai-advent agent --show-tokens --memory-file .ai-advent/short-dialog.json
ai-advent agent --show-tokens --memory-file .ai-advent/long-dialog.json
```

В длинном диалоге `prompt_tokens` обычно растет, потому что агент отправляет в
модель сохраненную историю сообщений. Если API не возвращает usage, значения
будут показаны как `n/a`.

Для больших сообщений удобно использовать paste-режим:

```text
You: /paste
...многострочный текст...
/send
```

### День 9: сжатие истории

Агент поддерживает две стратегии контекста:

- `full` - отправляет в модель всю сохраненную историю;
- `summary` - сжимает старую часть диалога в summary, а последние N сообщений
  хранит и отправляет как есть.

По умолчанию используется `full`:

```bash
ai-advent agent --context-strategy full
```

Запуск с summary-компрессией:

```bash
ai-advent agent --context-strategy summary --keep-last 10
```

Для ручного сравнения удобно открыть две вкладки терминала и использовать
разные файлы памяти:

```bash
ai-advent agent --show-tokens --context-strategy full --memory-file .ai-advent/full.json
```

```bash
ai-advent agent --show-tokens --context-strategy summary --keep-last 10 --memory-file .ai-advent/summary.json
```

Отправляйте одинаковые сообщения в оба процесса и сравнивайте качество ответов
и `prompt_tokens`. В summary-режиме JSON-память хранит отдельное поле
`summary` рядом с последними сообщениями:

```json
{
  "summary": "Краткое содержание старой части диалога...",
  "messages": []
}
```

Сжатие делает дополнительный LLM-запрос для обновления summary, когда история
становится длиннее `--keep-last`.

### День 10: стратегии без summary

Агент поддерживает три дополнительные стратегии управления контекстом:

- `sliding-window` - хранит и отправляет только последние N сообщений;
- `facts` - обновляет key-value facts и отправляет facts + последние N сообщений;
- `branch` - позволяет создавать checkpoints и независимые ветки диалога.

Sliding Window:

```bash
ai-advent agent --show-tokens --context-strategy sliding-window --keep-last 10 --memory-file .ai-advent/sliding.json
```

Sticky Facts:

```bash
ai-advent agent --show-tokens --context-strategy facts --keep-last 10 --memory-file .ai-advent/facts.json
```

Branching:

```bash
ai-advent agent --show-tokens --context-strategy branch --memory-file .ai-advent/branching.json
```

Команды для веток внутри чата:

```text
/checkpoint base
/branch create option_a base
/branch create option_b base
/branch switch option_a
/branch switch option_b
/branch list
/branch checkpoints
```

Что делают команды:

- `/checkpoint NAME` - сохраняет текущую историю активной ветки как checkpoint.
- `/branch create NAME` - создает новую ветку из текущего состояния диалога и
  сразу переключается на нее.
- `/branch create NAME CHECKPOINT` - создает новую ветку из указанного
  checkpoint и сразу переключается на нее.
- `/branch switch NAME` - переключает активную ветку; дальнейшие сообщения
  будут продолжать выбранную ветку.
- `/branch list` - показывает список веток и текущую активную ветку.
- `/branch checkpoints` - показывает список сохраненных checkpoints.

Для ручного сравнения запустите стратегии в разных терминалах, отправляйте один
и тот же сценарий на 10-15 сообщений и сравнивайте:

- качество финального ответа;
- стабильность важных деталей;
- `prompt_tokens`;
- удобство работы.

## Неделя 3

### День 11: модель памяти агента

Агент получил явные слои памяти для Study Coach сценариев:

- краткосрочная память - текущий диалог, поле `messages`;
- рабочая память - данные текущей учебной задачи, поле `working_memory`;
- долговременная память - устойчивые знания и решения, поле `long_term_memory`.

Краткосрочная память обновляется обычными репликами диалога. Рабочая и долговременная память обновляются только явными командами пользователя:

```text
/memory set working lesson_topic Python decorators
/memory set working current_exercise "write a decorator that logs calls"
/memory set long preferred_language ru
/memory set long learning_style "short explanation, then practice"
```

Посмотреть слои памяти:

```text
/memory show
/memory show short
/memory show working
/memory show long
```

Удалить значение:

```text
/memory forget working current_exercise
/memory forget long learning_style
```

Рабочая и долговременная память сохраняются отдельно в JSON-файле рядом с историей диалога и добавляются к каждому запросу отдельным system-блоком.

### День 12: персонализация ассистента

Агент получил профиль пользователя поверх модели памяти.

Профиль хранится в поле `user_profile` и подключается к каждому запросу отдельным system-блоком перед рабочей и долговременной памятью.

Настроить профиль:

```text
/profile set name Anton
/profile set language ru
/profile set answer_style "short explanation, then practice"
/profile set format "bullets for steps, code for examples"
/profile set constraints "do not give full exercise solution before my attempt"
```

Посмотреть профиль:

```text
/profile show
```

Удалить значение:

```text
/profile forget constraints
```

Для ручной проверки можно запустить два агента с разными файлами памяти и записать разные профили:

```bash
ai-advent agent --memory-file .ai-advent/profile-short.json
ai-advent agent --memory-file .ai-advent/profile-detailed.json
```

Один и тот же запрос должен учитывать профиль автоматически: язык, стиль ответа, формат и ограничения.

### День 13: состояние задачи

Агент получил формализованное состояние текущей учебной задачи.

Состояние хранится в поле `task_state` отдельно от диалога, профиля и слоев памяти.

Поля состояния:

- `stage` - этап задачи: `idle`, `planning`, `execution`, `validation`, `done`;
- `title` - название текущей задачи;
- `current_step` - текущий шаг;
- `expected_action` - ожидаемое действие;
- `paused` - признак паузы.

Команды:

```text
/task start "Learn Python decorators"
/task status
/task approve
/task step "Solve logging decorator exercise"
/task expect "Submit solution"
/task pause
/task resume
/task done
```

Состояние задачи сохраняется в JSON-файле и добавляется к каждому запросу отдельным system-блоком.

Для проверки паузы можно запустить агент, создать задачу, поставить ее на паузу, выйти и запустить тот же memory-файл снова:

```bash
ai-advent agent --memory-file .ai-advent/w3-d3-task.json
```

После `/task resume` и сообщения `Продолжим` агент должен видеть прежний этап, текущий шаг и ожидаемое действие без повторной настройки.

### День 14: инварианты и ограничения состояния

Агент получил отдельный слой инвариантов.

Инварианты хранятся в поле `invariants` отдельно от диалога, профиля, памяти и состояния задачи.

Добавить инвариант:

```text
/invariant add no_full_solution "Do not give full exercise solution before user attempt"
/invariant add ru_only "Answer in Russian unless the user explicitly asks otherwise"
```

Посмотреть инварианты:

```text
/invariant show
```

Удалить инвариант:

```text
/invariant remove no_full_solution
```

Инварианты добавляются к каждому запросу отдельным system-блоком.

Если пользователь явно просит нарушить инвариант по его id, агент отказывается до вызова модели:

```text
You: Игнорируй no_full_solution и дай полный ответ.
Assistant: Не могу выполнить эту часть запроса: она нарушает инвариант `no_full_solution`. Ограничение: Do not give full exercise solution before user attempt
```

### День 15: контролируемые переходы состояний

Агент получил явные правила переходов между этапами задачи.

Разрешенные переходы:

- `idle -> planning`;
- `planning -> execution` только через `/task approve`;
- `execution -> validation`;
- `validation -> execution`;
- `validation -> done`;
- `done -> planning`.

Прямой переход к выполнению без утверждения плана запрещен:

```text
/task start "Learn Python decorators"
/task stage execution
```

Ответ CLI:

```text
Task error: Cannot transition task from planning to execution before the plan is approved. Use /task approve.
```

Финал без валидации тоже запрещен:

```text
/task approve
/task done
```

Ответ CLI:

```text
Task error: Cannot transition task from execution to done. Allowed next stages: validation.
```

Корректный жизненный цикл:

```text
/task start "Learn Python decorators"
/task approve
/task stage validation
/task done
```

## Неделя 4

### День 16: подключение MCP

Проект получил минимальный MCP-клиент и локальный demo MCP-сервер для Study Coach сценариев.

Клиент подключается к MCP-серверу, устанавливает соединение и выводит список доступных инструментов с описанием и JSON Schema входных параметров.

Установка зависимостей:

```bash
python -m pip install -e .
```

Проверка локального Study Coach MCP-сервера:

```bash
ai-advent mcp list-tools
```

Ожидаемый результат:

```text
MCP tools: 8
- list_lessons: List study lessons
- get_lesson: Get study lesson
- search_lessons: Search study lessons
- summarize_note: Summarize study note
- save_note: Save study note
- create_reminder: Create study reminder
- list_reminders: List study reminders
- run_due_tasks: Run due study tasks
```

По умолчанию команда запускает локальный сервер через stdio:

```bash
python -m ai_advent.mcp_servers.study
```

Можно подключиться к внешнему Streamable HTTP MCP endpoint:

```bash
ai-advent mcp list-tools --url http://127.0.0.1:8000/mcp
```

Или указать свой stdio-сервер:

```bash
ai-advent mcp list-tools --server-command "python path/to/server.py"
```

### День 17: первый MCP-инструмент

Локальный Study Coach MCP-сервер теперь используется не только для discovery, но и для реального вызова инструмента.

Инструмент:

```text
get_lesson(topic: str) -> str
```

Он работает поверх локального mock API с учебными материалами и возвращает короткое объяснение темы.

Прямой вызов MCP-инструмента из CLI:

```bash
ai-advent mcp call-tool get_lesson --arguments '{"topic":"mcp"}'
```

Ожидаемый результат:

```text
MCP tool result: get_lesson
  is_error: False
  structured_content: {'result': 'Model Context Protocol lets an agent discover and call external tools through a standard client-server protocol.'}
  content:
    Model Context Protocol lets an agent discover and call external tools through a standard client-server protocol.
```

Инструмент также подключен к интерактивному агенту через команду:

```text
/lesson mcp
```

Что происходит:

1. CLI вызывает MCP-инструмент `get_lesson` с аргументом `{"topic": "mcp"}`.
2. Результат инструмента печатается в терминал.
3. Этот результат передается в `Agent.run_turn` как учебный материал.
4. Агент использует MCP-результат, чтобы кратко объяснить тему и предложить следующий практический шаг.

Демо-сценарий:

```bash
ai-advent agent
```

```text
You: /lesson mcp
MCP tool result: get_lesson
  is_error: False
  content:
    Model Context Protocol lets an agent discover and call external tools through a standard client-server protocol.
Assistant: ...
```

### День 18: планировщик и фоновые задачи

Study Coach MCP-сервер получил инструменты для отложенных учебных задач:

```text
create_reminder(title: str, due_in_seconds: int = 0, note: str = "") -> dict
list_reminders() -> dict
run_due_tasks() -> dict
```

Данные сохраняются в JSON-файл.

По умолчанию используется:

```text
.ai-advent/study-scheduler.json
```

Можно указать отдельный файл для демки:

```bash
ai-advent mcp call-tool create_reminder --arguments '{"title":"Review MCP Day 18","due_in_seconds":0,"note":"demo"}' --scheduler-file /private/tmp/ai-advent-day18-scheduler.json
```

Запустить один проход фонового worker:

```bash
ai-advent worker --once --scheduler-file /private/tmp/ai-advent-day18-scheduler.json
```

Worker вызывает MCP-инструмент `run_due_tasks`, отмечает due reminders выполненными и возвращает агрегированную сводку:

```text
MCP tool result: run_due_tasks
  is_error: False
  content:
    {
      "due_count": 1,
      "pending_count": 0,
      "completed_count": 1,
      "summary": "Completed 1 due study reminder(s): Review MCP Day 18. Pending reminders: 0."
    }
```

Запустить worker как долгоживущий процесс:

```bash
ai-advent worker --interval 60
```

Интерактивный агент тоже может создать reminder через MCP:

```text
You: /remind 300 Повторить MCP tools
```

Посмотреть сохраненные reminders:

```text
You: /reminders
```

### День 19: композиция MCP-инструментов

Study Coach получил автоматический pipeline из нескольких MCP-инструментов:

```text
search_lessons -> summarize_note -> save_note
```

Что делает pipeline:

1. `search_lessons` ищет учебный материал по запросу.
2. `summarize_note` превращает найденный материал в короткую заметку.
3. `save_note` сохраняет результат в Markdown-файл.

Запуск pipeline из CLI:

```bash
ai-advent mcp run-pipeline mcp --notes-dir /private/tmp/ai-advent-day19-notes
```

Ожидаемый результат:

```text
MCP pipeline: study note for mcp
1. search_lessons
   arguments: {'query': 'mcp'}
   is_error: False
2. summarize_note
   arguments: {'title': 'Study note: mcp', 'content': 'mcp: Model Context Protocol lets an agent discover and call external tools through a standard client-server protocol.'}
   is_error: False
3. save_note
   arguments: {'title': 'Study note: mcp', 'content': '# Study note: mcp...'}
   is_error: False
Saved note: /private/tmp/ai-advent-day19-notes/study-note-mcp.md
```

Проверить сохраненный файл:

```bash
cat /private/tmp/ai-advent-day19-notes/study-note-mcp.md
```

Интерактивный агент тоже может запустить pipeline:

```text
You: /note mcp
```

### День 20: orchestration MCP

Для orchestration сценария Study Coach использует несколько MCP-серверов:

```text
lessons server   -> list_lessons, get_lesson, search_lessons
notes server     -> summarize_note, save_note
scheduler server -> create_reminder, list_reminders, run_due_tasks
```

Orchestrator регистрирует серверы, собирает список tools, выбирает нужный сервер по имени инструмента и выполняет длинный flow:

```text
lessons.search_lessons -> notes.summarize_note -> notes.save_note -> scheduler.create_reminder -> scheduler.run_due_tasks
```

Запуск demo flow:

```bash
ai-advent mcp orchestrate mcp --notes-dir demo-output/day20-notes --scheduler-file demo-output/day20-scheduler.json --remind-in 0
```

Ожидаемый результат:

```text
Registered MCP servers:
- lessons: get_lesson, list_lessons, search_lessons
- notes: save_note, summarize_note
- scheduler: create_reminder, list_reminders, run_due_tasks
MCP orchestration: study flow for mcp
1. lessons.search_lessons
   arguments: {'query': 'mcp'}
   is_error: False
2. notes.summarize_note
   arguments: {'title': 'Study note: mcp', 'content': 'mcp: Model Context Protocol lets an agent discover and call external tools through a standard client-server protocol.'}
   is_error: False
3. notes.save_note
   arguments: {'title': 'Study note: mcp', 'content': '# Study note: mcp...'}
   is_error: False
4. scheduler.create_reminder
   arguments: {'title': 'Review mcp', 'due_in_seconds': 0, 'note': 'Review saved study note: demo-output/day20-notes/study-note-mcp.md'}
   is_error: False
5. scheduler.run_due_tasks
   arguments: {}
   is_error: False
Saved note: demo-output/day20-notes/study-note-mcp.md
Reminder id: ...
Scheduler summary: Completed 1 due study reminder(s): Review mcp. Pending reminders: 0.
```

Проверить сохраненную заметку:

```bash
cat demo-output/day20-notes/study-note-mcp.md
```

Проверить scheduler JSON:

```bash
cat demo-output/day20-scheduler.json
```

Интерактивный агент может запустить тот же flow:

```text
You: /study-flow mcp
```

## Неделя 5

### День 21: индексация документов

Проект получил первый слой RAG-инфраструктуры: локальный пайплайн индексации документов с chunking, embeddings, JSON-хранилищем и метаданными.

Индекс можно собрать из Markdown, Python, text и reStructuredText файлов:

```bash
ai-advent index build README.md AGENTS.md ai_advent tests --output .ai-advent/document-index.json --offline-embeddings
```

Флаг `--offline-embeddings` использует детерминированные локальные hash embeddings. Это удобно для демо и тестов без API-ключа.

Для локальных бесплатных embeddings через Ollama установите Ollama и скачайте embedding-модель:

```bash
ollama pull nomic-embed-text
```

Сборка индекса через Ollama:

```bash
ai-advent index build README.md AGENTS.md ai_advent tests --embedding-provider ollama --embedding-model nomic-embed-text --output .ai-advent/document-index.json
```

По умолчанию Ollama ожидается на `http://localhost:11434`. Можно указать другой адрес:

```bash
ai-advent index build README.md AGENTS.md ai_advent tests --embedding-provider ollama --ollama-url http://127.0.0.1:11434
```

Или настроить `.env`:

```dotenv
AI_ADVENT_OLLAMA_BASE_URL=http://localhost:11434
AI_ADVENT_EMBEDDING_MODEL=nomic-embed-text
```

Для реальных OpenAI-compatible embeddings настройте `.env`:

```dotenv
AI_ADVENT_API_KEY=sk-your-api-key
AI_ADVENT_BASE_URL=https://api.openai.com/v1
AI_ADVENT_EMBEDDING_MODEL=text-embedding-3-small
```

Команда с OpenAI-compatible embeddings API:

```bash
ai-advent index build README.md AGENTS.md ai_advent tests --output .ai-advent/document-index.json
```

По умолчанию сохраненный индекс использует structure-aware chunking. Можно выбрать fixed-size стратегию:

```bash
ai-advent index build README.md AGENTS.md ai_advent tests --strategy fixed --output .ai-advent/document-index-fixed.json --offline-embeddings
```

Настройки chunking:

- `--strategy fixed|structure` - стратегия, которая попадет в сохраненный индекс.
- `--embedding-provider openai|offline|ollama` - источник embeddings.
- `--embedding-model MODEL` - модель embeddings, например `text-embedding-3-small` или `nomic-embed-text`.
- `--ollama-url URL` - адрес локального Ollama API.
- `--fixed-chunk-size 1200` - размер fixed-size чанка в символах.
- `--fixed-overlap 150` - overlap между fixed-size чанками.
- `--structure-max-chunk-size 1800` - максимальный размер structure-aware чанка.

Что делает пайплайн:

1. Загружает документы из указанных файлов и директорий.
2. Сравнивает две стратегии chunking: fixed-size и structure-aware.
3. Для выбранной стратегии создает чанки с метаданными `source`, `title`, `section`, `chunk_id`.
4. Генерирует embedding для каждого чанка.
5. Сохраняет локальный JSON-индекс.

Пример вывода на текущем репозитории:

```text
Document index built
  saved_path: .ai-advent/document-index.json
  documents: 37
  total_characters: 248850
  strategy: structure
  chunks: 343
  embedding_model: text-embedding-3-small
Chunking comparison:
  fixed: 253 chunks, avg 1109.3 chars, min 28, max 1200
  structure: 343 chunks, avg 722.1 chars, min 10, max 1796
```

JSON-индекс содержит выбранную стратегию, модель embeddings, сравнение стратегий и список чанков:

```json
{
  "embedding_model": "text-embedding-3-small",
  "chunking_strategy": "structure",
  "comparison": [],
  "chunks": [
    {
      "chunk_id": "README.md:structure:0",
      "source": "README.md",
      "title": "README.md",
      "section": "AI Advent",
      "text": "...",
      "embedding": [0.1, -0.2]
    }
  ]
}
```

Основные модули:

- `ai_advent/documents.py` - загрузка документов.
- `ai_advent/chunking.py` - fixed-size и structure-aware chunking.
- `ai_advent/embeddings.py` - OpenAI-compatible embeddings adapter, Ollama embeddings adapter и offline hash embeddings.
- `ai_advent/vector_index.py` - JSON vector index и cosine search.
- `ai_advent/indexing.py` - сборка индекса и отчет сравнения стратегий.

### День 22: первый RAG-запрос

Проект получил первый RAG-пайплайн поверх локального JSON-индекса:

```text
вопрос -> embedding вопроса -> поиск релевантных чанков -> RAG prompt -> ответ LLM
```

Перед запросом соберите индекс, например через Ollama:

```bash
ollama pull nomic-embed-text
ai-advent index build README.md AGENTS.md ai_advent tests --embedding-provider ollama --embedding-model nomic-embed-text --output .ai-advent/document-index.json
```

Один RAG-запрос:

```bash
ai-advent rag ask "Какие команды CLI управляют task state?" --embedding-provider ollama --embedding-model nomic-embed-text --index .ai-advent/document-index.json --top-k 5
```

Сравнение ответа без RAG и с RAG:

```bash
ai-advent rag compare "Какие команды CLI управляют task state?" --embedding-provider ollama --embedding-model nomic-embed-text --index .ai-advent/document-index.json --top-k 5
```

`rag compare` делает два вызова chat-модели:

1. Без RAG: вопрос отправляется модели без локального контекста.
2. С RAG: сначала ищутся релевантные чанки, затем они добавляются в prompt вместе с вопросом.

Для локального demo без Ollama можно использовать hash embeddings, если индекс тоже был собран в offline-режиме:

```bash
ai-advent index build README.md AGENTS.md ai_advent tests --offline-embeddings --output .ai-advent/document-index-offline.json
ai-advent rag compare "Какие метаданные сохраняются у каждого чанка?" --offline-embeddings --index .ai-advent/document-index-offline.json
```

RAG-ответ печатает:

- ответ модели;
- список найденных чанков;
- similarity score;
- `source`, `section`, `chunk_id` для каждого найденного чанка.

Контрольный набор из 10 вопросов:

```bash
ai-advent rag eval-questions
```

Каждый вопрос содержит ожидаемый смысл ответа и ожидаемые источники.
Этот набор нужен для ручного сравнения качества режимов без RAG и с RAG.

Основные модули:

- `ai_advent/rag.py` - RAG responder, prompt builder и сравнение with/without RAG.
- `ai_advent/rag_eval.py` - 10 контрольных вопросов с ожиданиями и ожидаемыми источниками.

### День 23: реранкинг, фильтрация и query rewrite

RAG-пайплайн получил второй этап после vector search:

```text
вопрос -> query rewrite -> embedding search top-N -> similarity filter -> heuristic rerank -> top-K -> RAG prompt
```

Новый режим сравнения:

```bash
ai-advent rag compare-retrieval "Какие метаданные сохраняются у каждого чанка?" --embedding-provider ollama --embedding-model nomic-embed-text --index .ai-advent/document-index.json --top-k 5 --candidate-k 20 --min-score 0.2
```

Что сравнивается:

1. Baseline RAG: обычный поиск `top-k` без query rewrite, threshold и reranking.
2. Improved RAG: переписывает вопрос в поисковый запрос, берет больше кандидатов через `candidate-k`, отсекает слабые результаты через `min-score`, затем переупорядочивает кандидатов эвристикой `similarity + keyword overlap`.

Параметры:

- `--candidate-k 20` - сколько кандидатов взять до фильтрации и реранкинга.
- `--min-score 0.2` - минимальный similarity score для допуска чанка в контекст.
- `--top-k 5` - сколько чанков оставить после фильтрации и реранкинга.
- `--rewrite-query` - включает query rewrite для обычного `rag ask`.

Обычный RAG-запрос тоже поддерживает улучшения:

```bash
ai-advent rag ask "Как собрать индекс через Ollama?" --embedding-provider ollama --embedding-model nomic-embed-text --candidate-k 20 --min-score 0.2 --rewrite-query
```

Для offline demo:

```bash
ai-advent rag compare-retrieval "Какие две стратегии chunking реализованы?" --offline-embeddings --index .ai-advent/document-index-offline.json --top-k 5 --candidate-k 20 --min-score 0.1
```

В выводе видно:

- `Search query` - исходный или переписанный поисковый запрос;
- `Candidates before filtering` - сколько чанков было найдено до фильтрации;
- итоговые retrieved chunks после фильтрации и реранкинга;
- `score`, `source`, `section`, `chunk_id` для каждого оставшегося чанка.

Реализация остается детерминированной и тестируемой:

- query rewrite - отдельный Chat Completions вызов;
- similarity threshold - простой числовой фильтр;
- reranking - локальная эвристика поверх similarity и пересечения ключевых слов.

### День 24: цитаты, источники и анти-галлюцинации

RAG получил grounded-режим с обязательными источниками, цитатами и отказом при слабом контексте.

Команда:

```bash
ai-advent rag cited "Какие метаданные сохраняются у каждого чанка?" --embedding-provider ollama --embedding-model nomic-embed-text --index .ai-advent/document-index.json --top-k 5 --candidate-k 20 --min-score 0.2 --rewrite-query
```

Что делает `rag cited`:

1. Ищет релевантные чанки через тот же retrieval pipeline.
2. Отсекает чанки ниже `--min-score`.
3. Собирает детерминированные источники и короткие цитаты из найденных чанков.
4. Отправляет модели prompt со строгим требованием вернуть блоки `Ответ`, `Источники`, `Цитаты`.
5. Дополнительно печатает источники и цитаты из кода, чтобы результат можно было проверить даже если модель форматирует ответ свободно.

Если после фильтрации нет релевантных чанков, модель не вызывается.
CLI отвечает:

```text
Не знаю: в локальном индексе не нашлось достаточно релевантного контекста. Уточните вопрос или соберите индекс по более подходящим документам.
```

Проверка режима “не знаю”:

```bash
ai-advent rag cited "Как приготовить ризотто?" --embedding-provider ollama --embedding-model nomic-embed-text --index .ai-advent/document-index.json --min-score 0.95
```

Для offline demo:

```bash
ai-advent rag cited "Какие две стратегии chunking реализованы?" --offline-embeddings --index .ai-advent/document-index-offline.json --top-k 5 --candidate-k 20 --min-score 0.1
```

В выводе есть:

- `Grounded RAG answer` - ответ модели или отказ “не знаю”;
- `Sources` - список `source | section | chunk_id`;
- `Quotes` - короткие фрагменты из найденных чанков;
- `Retrieved chunks` - технический список чанков с similarity score.

Контрольные вопросы для ручной проверки:

```bash
ai-advent rag eval-questions
```

Для каждого из 10 вопросов проверьте:

1. Есть ли источники.
2. Есть ли цитаты.
3. Совпадает ли смысл ответа с цитатами.
4. Срабатывает ли “не знаю” при завышенном `--min-score`.

### День 25: мини-чат с RAG и памятью задачи

Проект получил production-like mini-chat поверх grounded RAG.

Запуск:

```bash
ai-advent rag chat --embedding-provider ollama --embedding-model nomic-embed-text --index .ai-advent/document-index.json --memory-file .ai-advent/rag-chat-memory.json --top-k 5 --candidate-k 20 --min-score 0.2
```

Чат хранит:

- историю диалога;
- цель текущего диалога;
- уточнения пользователя;
- ограничения;
- термины.

Команды внутри `rag chat`:

```text
/goal TEXT
/clarify TEXT
/constraint TEXT
/term KEY VALUE
/memory
/exit
```

Каждый обычный вопрос проходит через grounded RAG:

```text
user message -> task memory + recent history -> retrieval -> grounded answer -> sources -> quotes
```

Память сохраняется в JSON-файл:

```text
.ai-advent/rag-chat-memory.json
```

Можно указать отдельный файл для демо:

```bash
ai-advent rag chat --memory-file demo-output/day25-rag-chat.json --embedding-provider ollama --embedding-model nomic-embed-text
```

Два длинных сценария проверки:

```bash
ai-advent rag chat-scenarios
```

Сценарии проверяют, что ассистент:

- не теряет цель диалога;
- учитывает уточнения, ограничения и термины;
- продолжает отвечать с источниками и цитатами;
- выдерживает 10 сообщений в одной сессии.

Основные модули:

- `ai_advent/rag_chat.py` - RAG chat session, JSON memory, task memory, demo scenarios.
- `ai_advent/rag.py` - grounded answer generation with sources, quotes and low-relevance refusal.
