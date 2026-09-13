# AI Advent

CLI-проект для челленджа по изучению AI-агентов. Ветка второй недели строит одного агента, который будет постепенно развиваться: от простого вызова LLM до памяти, подсчета токенов и стратегий управления контекстом.

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
