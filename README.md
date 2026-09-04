# AI Advent — день 1

Интерактивная CLI-утилита, которая отправляет сообщения через OpenAI Python SDK и выводит ответы в консоль. Можно использовать OpenAI или другой OpenAI-compatible endpoint, например DeepSeek или Z.ai. Контекст сохраняется в течение запущенной сессии.

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
```

Файл `.env` исключён из Git и не попадёт в репозиторий.

## Запуск

```bash
ai-advent
```

Введите сообщение и нажмите Enter. Для завершения используйте `/exit`, `/quit` или `Ctrl+D`.

Модель и провайдера можно менять через `.env`:

```dotenv
AI_ADVENT_API_KEY=your-provider-api-key
AI_ADVENT_MODEL=deepseek-chat
AI_ADVENT_BASE_URL=https://api.deepseek.com
```

Для OpenAI-compatible API используются те же вызовы SDK: меняются только ключ, модель и `base_url`.

## Пример

```text
AI Advent chat (model: gpt-5.6-luna)
Type /exit or /quit to stop.

You: What is an API?
Assistant: An API is an interface that lets programs communicate with each other.

You: Explain it using an analogy.
Assistant: Think of an API as a waiter who carries your order to a kitchen...
```
