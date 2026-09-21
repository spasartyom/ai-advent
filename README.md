# AI Advent

CLI-проект для челленджа по изучению AI-агентов. Вторая неделя построила базового CLI-агента с памятью, подсчетом токенов и стратегиями управления контекстом. Третья неделя развивает его в Study Coach Agent: учебного ассистента с явной моделью памяти и управляемым состоянием задач.

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
/task stage execution
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
