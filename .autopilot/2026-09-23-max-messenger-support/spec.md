# Спецификация: параллельный MAX-адаптер Passport OCR Bot

## Задача

Добавить MAX как второй messenger adapter к существующему Passport OCR Bot. Telegram и MAX работают отдельными процессами, но используют один OCR/business layer и одну PostgreSQL. Пользовательский сценарий Telegram не меняется.

## Решение

Текущий `bot/handlers.py` перестаёт быть владельцем OCR-пайплайна: распознавание, нормализация, инференс пола, форматирование результата и запись в БД переходят в глубокий модуль `PassportProcessingService` с SDK-независимым интерфейсом. Telegram handler и MAX handler только преобразуют событие платформы в `IncomingImage`, скачивают байты платформенным способом и отправляют presentation результата.

Существующий `main.py` остаётся Telegram entrypoint. MAX получает отдельный `bot/max_main.py`. Общий startup/shutdown выносится в `services/runtime.py`, чтобы оба процесса одинаково инициализировали БД и OCR providers и закрывали ресурсы.

## Пользовательские истории и приёмка

| # | Метка | История | Приёмка |
|---|---|---|---|
| 1 | R01/R39 | Как текущий пользователь Telegram, я отправляю фото/PDF и получаю прежний результат | текущие aiogram commands, photo/document handlers и HTML-result продолжают работать; regression tests/import checks зелёные |
| 2 | R02/R49 | Как пользователь MAX, я отправляю фото паспорта | MAX handler формирует `IncomingImage`, общий service сохраняет запись и отправляет equivalent result |
| 3 | R12/R45 | Как пользователь MAX, я пересылаю фото из другого MAX-чата | изображения из `message.link.message.attachments` обрабатываются тем же service |
| 4 | R13/R44 | Как пользователь MAX, я отправляю обычное image attachment | attachment `type=image` и `payload.url` преобразуются в байты без доступа OCR к SDK |
| 5 | R16/R18-R24/R42-R43 | Как оператор, я вижу источник записи и не смешиваю ID платформ | `passport_records.source` равен `telegram`/`max`; external IDs — строки; Telegram legacy rows безопасно получают `telegram` |
| 6 | R25-R31/R47-R48 | Как оператор, я поднимаю два бота в Docker | `docker compose up -d --build` поднимает `postgres`, migration job, `telegram_bot`, `max_bot`; сервисы используют один image tag, `.env` и `DATABASE_URL` |
| 7 | R34-R38 | Как пользователь, я получаю понятный ответ на плохое сообщение или сетевую ошибку | отсутствие фото, неподдерживаемое вложение, несколько фото, download timeout/status/size, OCR/DB/MAX errors обрабатываются на одном сообщении и не завершают process |
| 8 | R36/R37 | Как владелец персональных данных, я не вижу паспортные данные и секреты в логах | URL не логируется полностью, image/Base64/passport fields/token values не логируются; HTTP status/size/timeout проверяются |
| 9 | R40-R41/R50-R53 | Как сопровождающий, я могу проверить интеграцию без внешних API | unit tests для adapters/domain/repository mapping; `compileall`, Docker import checks, compose config checks и доступные tests выполняются |

## Контракты модулей

### `core/messaging.py`

Владеет messenger-независимым входом и источником.

```python
class MessengerSource(str, Enum):
    TELEGRAM = "telegram"
    MAX = "max"

@dataclass(frozen=True, slots=True)
class IncomingImage:
    content: bytes
    filename: str | None
    source: MessengerSource
    external_user_id: str | None
    external_chat_id: str | None
    external_message_id: str | None
    external_username: str | None = None
    source_type: str = "photo"
    source_file_id: str | None = None
    source_page_index: int | None = None
```

Инварианты: `content` содержит байты только в памяти; все внешние ID нормализованы в строки; `source` всегда один из enum values; SDK-типы в этом модуле запрещены.

### `services/passport_processing.py`

Глубокий общий модуль. Он владеет `ImageProcessor`, `HybridRecognizer`, OpenRouter rate limit, provider lifecycle, gender inference, сохранением `PassportRecord` и построением platform-neutral `PassportResult`.

```python
class PassportProcessingService:
    async def start(self) -> None
    async def close(self) -> None
    async def recognize_image(self, image_bytes: bytes) -> PassportResult
    async def process_image(
        self,
        incoming: IncomingImage,
        notify_wait: Callable[[float], Awaitable[None]] | None = None,
    ) -> PassportResult
```

`recognize_image` не принимает Telegram/MAX объекты. `process_image` добавляет persistence по метаданным `IncomingImage`. Ошибки провайдеров не роняют процесс; при отсутствии распознанных полей сохраняется и форматируется результат с текущими placeholder-значениями Telegram, чтобы не менять его поведение.

`PassportResult` содержит форматированные `format1`, `format2`, общие detail lines, structured details для web UI и `quality_score`; он не содержит messenger SDK objects.

### `bot/max_adapter.py`

Владеет только контрактом MAX:

- принимает `MessageCreated`;
- читает `message.body.attachments`;
- для forward читает `message.link.message.attachments`;
- принимает `sender.user_id`, `recipient.chat_id`/`recipient.user_id`, `body.mid`;
- скачивает `payload.url` через одну сессию `maxapi.Bot`;
- ограничивает поток `OCR_MAX_FILE_MB`, проверяет HTTP status и timeout;
- превращает каждый image attachment в `IncomingImage`;
- отправляет plain-text presentation и сообщения об ошибках.

Обычное сообщение и forwarded message используют один extractor. Если `body` отсутствует, проверяется linked message. Не-image attachments считаются неподдерживаемыми. Несколько изображений обрабатываются последовательно; ошибка одного не блокирует остальные.

MAX ID сообщения сохраняется как строка. Для forwarded изображения запись связывается с текущим входящим сообщением, а `source_file_id` получает безопасный идентификатор attachment, если он есть.

### `bot/handlers.py`

Остаётся Telegram adapter. Existing `/start`, `/export`, callback export, photo/document limits, статусные сообщения и HTML response сохраняются. Handler только:

1. извлекает Telegram metadata;
2. скачивает файл через aiogram;
3. создаёт `IncomingImage` для photo/image document/PDF page;
4. вызывает `PassportProcessingService`;
5. редактирует Telegram status message прежним HTML-текстом.

PDF остаётся Telegram-only сценарием текущего поведения; страницы передаются в общий service как `source_type=pdf_page`.

## Presentation

Общие данные результата и detail labels строятся service/formatter-слоем. Telegram получает прежнюю оболочку:

```html
<code>format1</code>
<code>format2</code>
<blockquote expandable>details</blockquote>
```

MAX получает plain text с теми же двумя encoded formats и теми же detail values, без Telegram-only tags. Это меняет только способ разметки, а не результат OCR/бизнес-правила.

## MAX download/error policy

- URL берётся из typed attachment payload; полный URL никогда не попадает в log.
- Используется `await bot.ensure_session()` из `maxapi`, не новый client на каждое сообщение.
- Timeout — 30 секунд на скачивание.
- Если `Content-Length` больше `settings.ocr_max_file_bytes`, файл отклоняется до чтения.
- При chunked response чтение прекращается после `limit + 1` байт.
- Любой non-2xx, timeout, invalid URL, превышение лимита или пустой body — локальная ошибка текущей картинки.
- Байты не записываются во временный файл MAX adapter; объект bytes передаётся service и освобождается после обработки.
- `maxapi.Bot.close_session()` вызывается в `finally` MAX entrypoint.

Сообщение без фото/с неподдерживаемым attachment получает короткое объяснение с поддерживаемым форматом. Ошибка скачивания — «не удалось скачать изображение». Ошибка OCR/БД/API — пользовательское безопасное сообщение, подробности только в логах без payload/секретов. Общий handler exception boundary ловит неожиданные ошибки.

## PostgreSQL и миграция

`PassportRecord` получает:

- `source VARCHAR(20) NOT NULL DEFAULT 'telegram'`;
- `external_user_id VARCHAR(255)`;
- `external_chat_id VARCHAR(255)`;
- `external_message_id VARCHAR(255)`;
- `external_username VARCHAR(255)`.

Существующие `tg_user_id`, `tg_username`, `source_message_id` сохраняются для обратной совместимости и экспорта Telegram. `tg_user_id` становится nullable для MAX records. Telegram новые записи заполняют одновременно legacy поля и external string fields; MAX records используют external fields, `tg_user_id=NULL`. Новых уникальных constraints не добавляется: текущая схема их не имеет, а повторное numeric ID разных sources не должно конфликтовать.

Alembic revision `003_add_messenger_source.py` добавляет колонки и default без удаления данных. Existing rows получают `source=telegram`; nullable alteration позволяет вставить MAX record. Downgrade удаляет только новые колонки и возвращает legacy `tg_user_id` non-null только если это безопасно для existing data — downgrade должен быть явно документирован как операция для схемы без MAX.

Repository получает единый `source` и external metadata, логирует только record UUID, source и quality score. Export headers становятся source-neutral (`Source`, `External User ID`, `External Chat ID`, `External Message ID`, `External Username`) с сохранением legacy Telegram columns для совместимости.

## Configuration and dependencies

В существующий `Settings` добавляются:

```env
MAX_BOT_TOKEN=
MAX_BOT_ENABLED=true
TELEGRAM_BOT_ENABLED=true
WEB_ENABLED=true
```

`BOT_TOKEN` становится optional at settings load, но Telegram entrypoint требует его, если `TELEGRAM_BOT_ENABLED=true`; MAX entrypoint требует `MAX_BOT_TOKEN`, если `MAX_BOT_ENABLED=true`. MAX service в Compose выставляет `WEB_ENABLED=false`, Telegram оставляет `true`. Secret values отсутствуют в git и startup logs.

`requirements.txt` получает `maxapi>=1.2.2,<2.0`. Другой dependency file не создаётся. Dockerfile остаётся универсальным; Compose один раз собирает image `pasport-scan-bot:${BOT_VARIANT:-light}` и использует его двумя bot services и migration job.

## Lifecycle and Compose

`services/runtime.py` выполняет общий порядок:

1. retry подключения/создания таблиц;
2. проверка OCR provider configuration без вывода secret previews;
3. создание и старт `PassportProcessingService`;
4. adapter-specific dispatcher polling;
5. shutdown dispatcher, provider HTTP clients, MAX SDK session, web server и DB.

Compose services:

- `postgres` — единственный PostgreSQL volume;
- `migration` — one-shot `alembic upgrade head` из того же image, ждёт healthy postgres;
- `telegram_bot` — `python main.py`, web port exposed, `WEB_ENABLED=true`;
- `max_bot` — `python -m bot.max_main`, web disabled, тот же image/env/database.

Один контейнер содержит один long-running process. Existing `main.py` остаётся рабочим прямым Telegram запуском.

## Security/logging

Сохраняется запись OCR raw payload в БД как существующая audit-функция, но payload, Base64, passport fields, image URL и tokens не логируются. Чувствительные существующие debug statements в hybrid/OpenRouter и startup credential previews удаляются или заменяются на counts/statuses. Временные файлы старого Telegram/web поведения не расширяются; MAX image не сохраняется на диск.

## Tests and verification

Минимальный `tests/` на stdlib `unittest` проверяет:

- `PassportProcessingService` импортируется без Telegram/MAX SDK;
- `MessengerSource` и `IncomingImage` корректны;
- fake repository receives Telegram source and MAX source;
- normal and forwarded MAX attachment extraction produce the same common input shape;
- string IDs preserve equal-looking IDs without cross-source collision in mapping;
- multiple/unsupported attachment decisions are deterministic.

Проверки после реализации: `python -m unittest discover -s tests`, `python -m compileall -q .` с исключением `.git`, `python -c` imports, `docker compose config`, сборка/import внутри Docker, process smoke check с disabled polling. Внешние Telegram/MAX/OCR tokens не требуются и не запрашиваются.

## Вне рамок

- MAX webhook и публичный TLS endpoint не добавляются, потому что в репозитории нет адреса/секрета webhook, а текущая схема использует polling.
- MAX `/export`, callback keyboards и PDF не добавляются сверх основного сценария фотографий; Telegram export не меняется.
- Новые уникальные constraints и опасное преобразование legacy BIGINT колонок не добавляются.
- Отдельная БД, отдельный OCR pipeline или копия проекта не создаются.

## Покрытие требований

R01–R05: разделы «Задача», «Решение», Telegram adapter. R06–R10 и R41/R49: контракты modules. R11–R17 и R34–R38: stories, MAX adapter, error policy. R18–R24/R42/R43: PostgreSQL. R25–R28: Configuration. R29–R32/R47/R48/R52/R53: Compose. R33: Presentation. R35–R37: error/security policies. R39: Telegram adapter and verification. R40/R44/R45/R50/R51: Tests and verification. R46: README update ticket. Все требования имеют статус `in-spec`; deferred/dropped требований нет.

## G2 coverage resolution

Independent coverage review confirmed that the implementation plan must explicitly preserve the repository-analysis work, the existing dependency and logging mechanisms, and the current temporary-file behavior. These are implementation invariants rather than new product scope:

- the repository analysis already completed before this specification remains a required phase gate and is recorded in the run artifacts;
- README and the final handoff must include architecture, file lists, env variables, migration, startup and per-service log commands;
- available lint and type-check commands must be discovered and run when present, with failures fixed before completion;
- `maxapi`'s reusable HTTP session is preferred, and any existing project HTTP client/lifecycle is reused where compatible;
- D01 amendment: the MAX SDK session is reserved for MAX API traffic because its default Authorization header must never be sent to arbitrary image hosts. Image downloads use one separately managed unauthenticated async client/session for the process lifetime, with the same timeout/size policy, and close it during MAX shutdown;
- D01 amendment: MAX metadata resolution may use the dispatcher-populated `event.from_user` fallback when a forwarded `Message.sender` is absent; it must not invent a user ID from the bot recipient;
- D01 amendment: a zero-field OCR result is `success=False`, is not formatted as `unknown/000000`, and receives the safe no-result response without a successful business record;
- MAX downloads stay in memory and any existing Telegram/web temporary-file behavior is preserved; newly introduced temporary files must use guaranteed cleanup;
- `requirements.txt`, the existing logging setup, and the existing Docker/Compose path remain the source of truth;
- `MAX_API_BASE_URL` is intentionally omitted unless implementation inspection proves the selected `maxapi` version requires it.

The review found no deferred requirement and no unapproved scope expansion.
