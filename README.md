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

```text
AI Advent agent (model: gpt-5.6-luna)
Type /exit or /quit to stop.

You: What is an agent?
Assistant: An agent is a program that can use an LLM plus state and logic...
```
