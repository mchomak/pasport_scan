# Interfaces and implementation boundaries

## Resolved boundaries from the specification

The feature is one backend with two messenger adapters. Telegram and MAX may import the shared service, but the shared service must not import `aiogram`, `maxapi`, or messenger event types.

```python
from dataclasses import dataclass
from enum import Enum

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

The shared processing service owns image normalization, OCR providers, gender inference, repository persistence, and platform-neutral result data. Its public lifecycle is:

```python
class PassportProcessingService:
    async def start(self) -> None: ...
    async def close(self) -> None: ...
    async def process_image(
        self,
        incoming: IncomingImage,
        notify_wait: Callable[[float], Awaitable[None]] | None = None,
    ) -> PassportResult: ...
```

`PassportResult` contains the two existing formatted result strings plus neutral details text, recognition state, and quality score. Telegram wraps those values in the existing HTML markup; MAX sends a safe plain-text equivalent.

The MAX adapter owns only `MessageCreated` parsing, image attachment discovery, user/chat/message ID extraction, download through one reusable adapter-owned unauthenticated session separate from the Authorization-bearing `maxapi` session, and user-facing error replies. The real `aiohttp` connector rejects non-global resolved addresses and disables redirects. It exposes pure extraction helpers so ordinary `message.body.attachments` and forwarded `message.link.message.attachments` are testable without a live SDK.

The common runtime owns database initialization/retry, OCR configuration validation, creation/closure of `PassportProcessingService`, and cancellation of background refresh tasks. `main.py` remains the Telegram entry point; `bot/max_main.py` is the MAX entry point.

## Data and migration boundary

`PassportRecord` keeps existing Telegram columns for export compatibility and adds `source`, `external_user_id`, `external_chat_id`, `external_message_id`, and `external_username`. `tg_user_id` becomes nullable for MAX rows. The migration gives existing rows `source='telegram'`, never deletes data, and adds no new uniqueness constraint.

## Project rules the executor must not infer

- Keep `requirements.txt` as the dependency source of truth and add `maxapi>=1.2.2,<2.0` there.
- Keep current `utils.logger` logging; never log image bytes, Base64, OCR fields, raw API response content, URLs with credentials, or token prefixes.
- MAX downloads remain in memory, use the adapter-owned unauthenticated session, enforce `OCR_MAX_FILE_MB`, reject unsafe resolved addresses, do not follow redirects, check status, and use a 30-second timeout. Any new temporary file must be created with guaranteed cleanup.
- Preserve Telegram command names, current photo/document/PDF behavior, and HTML result text.
- Compose must run `postgres`, one migration job, `telegram_bot`, and `max_bot`; bot services share one image, `.env`, and PostgreSQL. One long-running process per container.
- Do not add `MAX_API_BASE_URL` unless implementation inspection proves the selected SDK requires it.
- Do not ask for or add real tokens. Tests must use fake values only.
- If a required dependency is missing, report `BLOCKED` rather than silently installing a second dependency system.

## Verification commands

Run from the repository root:

```powershell
python -m unittest discover -s tests -v
python -m compileall -q .
docker compose config
docker compose build telegram_bot max_bot
```

Also discover and run any configured lint/type-check command; if none exists, record that explicitly in the final verification.

## Built by ticket 01

- `core.messaging.IncomingImage.__post_init__` normalizes all external identifiers to strings, so Telegram numeric IDs and MAX string IDs share one persistence shape.
- `services.passport_processing.PassportProcessingService.process_image()` returns `PassportResult` and persists source-aware metadata through `PassportRepository.create()`; its provider dependencies are started once and closed by `ApplicationRuntime`.
- `services.runtime.ApplicationRuntime.start()` performs DB initialization/retry and service startup; `.close()` closes provider clients, refresh tasks, and DB resources.
- Alembic revision `003_add_messenger_source` is the current head and preserves legacy Telegram columns while adding source/external metadata.
- Existing OCR logging now records status/length/count information only; raw passport values and token previews are excluded.

## Built by wave 2

- Telegram uses `set_processing_service(PassportProcessingService)` and passes `IncomingImage` objects through the existing photo/document/PDF handlers; the existing HTML result wrapper remains in the adapter.
- MAX exposes `extract_image_attachments`, `download_image`, `handle_message`, and `register_handlers`; `bot.max_main.main()` owns MAX polling and SDK session shutdown.
- MAX attachment extraction checks both ordinary `message.body.attachments` and forwarded `message.link.message.attachments`, then routes bytes only through `PassportProcessingService`.
- Settings now expose `MAX_BOT_TOKEN`, `MAX_BOT_ENABLED`, `TELEGRAM_BOT_ENABLED`, and `WEB_ENABLED`; Compose defines `postgres`, `migration`, `telegram_bot`, and `max_bot` over one image/database.
- Deployment verification completed through `docker compose config`, `docker compose build telegram_bot max_bot`, Docker imports, and disabled-entrypoint smoke checks.
- Telegram presentation is finalized through `_legacy_telegram_details(PassportResult)` and `_format_telegram_result(PassportResult, response_prefix)`, preserving legacy Russian labels, skipped-module reporting, `[Итог]` formatting, and the existing HTML wrapper.

## Built by acceptance repairs

- Zero-field OCR returns `success=False` with empty formatted fields and is not persisted; forwarded events use `event.from_user` when `message.sender` is absent.
- Persistence receives the same normalized `PassportData` used to build the response, including inferred gender.
- MAX polling errors are propagated after cleanup so Compose restart policy can recover; `TELEGRAM_BOT_ENABLED=false` exits before runtime/DB initialization.
- The real download session uses a custom public-address resolver with DNS cache disabled, while duck-typed test sessions retain the small fake-session contract.
