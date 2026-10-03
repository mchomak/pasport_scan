<!-- autopilot:start -->
# Passport OCR Bot - operational memory (T2)

Two messenger adapters share one OCR/business/persistence backend and PostgreSQL. `main.py` is the Telegram entry point and optional web server; `bot/max_main.py` is the MAX entry point. The shared layer must not import `aiogram`, `maxapi`, or messenger event types.

`core/messaging.py` defines the frozen/slotted `IncomingImage` boundary. It carries image bytes, filename, `MessengerSource` (`telegram` or `max`), normalized external user/chat/message IDs, optional username, source type/file ID, and PDF page index. `services/passport_processing.py` owns image normalization, hybrid OCR/gender inference, the neutral `PassportResult`, and source-aware persistence through `db/repository.py`. `PassportProcessingService.start()`, `.process_image(...)`, and `.close()` are the shared lifecycle; `PassportResult` contains the existing formatted result strings plus neutral details, recognition state, and quality score.

`services/runtime.py` owns database initialization/retry, OCR configuration validation, provider lifecycle, IAM refresh-task cancellation, service startup/closure, and DB shutdown. `bot/handlers.py` preserves Telegram photo/document/PDF handling and HTML presentation. `bot/max_adapter.py` owns MAX `MessageCreated` parsing, ordinary and forwarded attachment discovery, ID extraction, bounded in-memory download through one reusable SDK session, and plain-text/error replies; its extraction helpers are pure/testable. `bot/max_main.py` owns MAX polling and SDK-session shutdown.

`db/models.py`/`db/repository.py` keep legacy Telegram columns for export compatibility and add source/external metadata. `tg_user_id` is nullable for MAX rows. `alembic/versions/003_add_messenger_source.py` is the current head: existing rows receive `source='telegram'`, data is not deleted, and no new uniqueness constraint is added.

## Runbook

```text
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m compileall -q .
docker compose config
docker compose build telegram_bot max_bot
docker compose up -d --build
```

The Compose `migration` service runs `alembic upgrade head` against PostgreSQL; the host verification shell does not have an `alembic` executable. `docker compose up -d --build` and the migration were not run during 2026-09-23 documentation verification. On 2026-09-23, unittest passed with 23 tests, compileall passed, Compose config passed, and both bot image builds passed. On 2026-09-25, unittest passed with 23 tests, compileall and `docker compose config --quiet` passed, and `docker compose build max_bot` passed; `docker compose up -d --no-deps max_bot` redeployed only MAX. In-container imports confirmed NumPy 1.26.4/OpenCV 4.9.0, PostgreSQL remained healthy, and no migration or full-stack restart was run. No configured lint/type-check command was found.

## Key paths

`main.py`, `bot/max_main.py`, `bot/handlers.py`, `bot/max_adapter.py`, `core/messaging.py`, `services/runtime.py`, `services/passport_processing.py`, `db/models.py`, `db/repository.py`, `alembic/versions/003_add_messenger_source.py`, `tests/test_shared_core.py`, `tests/test_messenger_support.py`, `requirements.txt`, `requirements-full.txt`, `Dockerfile`, `docker-compose.yml`, and `alembic.ini` are present at the repository root.

## Environment names

`BOT_TOKEN`, `MAX_BOT_TOKEN`, `ADMIN_IDS`, `TELEGRAM_BOT_ENABLED`, `MAX_BOT_ENABLED`, `DATABASE_URL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `OCR_PROVIDER_MODEL`, `OCR_MODULE_PRIORITY`, `MAX_OCR_MODULE_PRIORITY`, `YC_FOLDER_ID`, `YC_API_KEY`, `YC_IAM_TOKEN`, `YC_OAUTH_TOKEN`, `YC_OCR_ENDPOINT`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_RPM`, `OCR_MAX_FILE_MB`, `OCR_MAX_MEGAPIXELS`, `OCR_RATE_LIMIT_RPS`, `PDF_RENDER_DPI`, `TMP_DIR`, `STORE_SOURCE_FILES`, `WEB_PORT`, `WEB_ENABLED`, `BOT_VARIANT`, `LOG_LEVEL`.

## Compose and dependency gotchas

- Services are `postgres`, one-shot `migration`, `telegram_bot`, and `max_bot`. Both bots use the shared `Dockerfile`/image, optional `.env`, PostgreSQL, and `./tmp`; each has one long-running process. `max_bot` defaults to OCR priority `openrouter,rupasportread`, can use `MAX_OCR_MODULE_PRIORITY=openrouter` for an OpenRouter-only control run, has a 1 CPU / 768 MiB limit, and publishes no host port.
- Bots wait for healthy PostgreSQL and successful migration. Telegram runs `main.py` and publishes the web port; MAX runs `bot.max_main` and disables the web server. `BOT_VARIANT` selects the light/full image and optional Tesseract/OpenCV dependencies.
- `requirements.txt` owns base dependencies, including `maxapi>=1.2.2,<2.0`; `requirements-full.txt` owns full-variant OCR extras, including `opencv-python-headless==4.9.0.80`.
- Keep NumPy `<2` for OpenCV 4.9 compatibility or `cv2` fails to import with `_ARRAY_API not found`.
- PostgreSQL data persists in `postgres_data` across ordinary `docker compose down`; removing the volume is deliberate data deletion.
- MAX accepts only HTTP(S) image URLs, rejects unsafe/private resolved addresses, downloads in memory through the SDK session without redirects, enforces `OCR_MAX_FILE_MB`, checks status and response size, and uses a 30-second timeout. No `MAX_API_BASE_URL` is used.
- `ops/start_max_bot.sh` validates the server `.env`, starts only the MAX stack, checks the existing VPN listeners are unchanged, and starts `ops/collect_server_metrics.py` for 24 hours. Metrics are resource-only; the collector samples host/Docker values every 10 seconds and Tesseract process CPU/RSS about once per second.

## Tests and security conventions

Tests cover the shared import boundary, external-ID normalization, source-aware persistence, forwarded/mixed MAX attachments, deterministic ordering, session reuse/cleanup, disabled Telegram startup, unsafe URL rejection, zero-field OCR handling, and logging redaction. Use fake values only in tests. Keep `utils.logger` logging; never log image bytes, Base64, OCR/passport fields, raw API responses, credential-bearing URLs, token prefixes, or secrets. Keep Telegram HTML rendering in the Telegram adapter and MAX output plain text. Any temporary file must have guaranteed cleanup.
<!-- autopilot:end -->
