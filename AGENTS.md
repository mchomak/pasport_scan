<!-- autopilot:start -->
# Passport OCR Bot - operational memory (T2)

Two messenger adapters share one OCR/business/persistence backend and PostgreSQL. The real Telegram entrypoint is `python main.py`; it runs the Telegram poller and, when enabled, the optional web server. The independent MAX entrypoint is `python -m bot.max_main`; it runs MAX polling without the web server and owns MAX SDK-session shutdown. Compose uses these same commands for `telegram_bot` and `max_bot`. The shared layer must not import `aiogram`, `maxapi`, or messenger event types.

`core/messaging.py` defines the frozen/slotted `IncomingImage` boundary. It carries image bytes, filename, `MessengerSource` (`telegram` or `max`), normalized external user/chat/message IDs, optional username, source type/file ID, and PDF page index. `services/passport_processing.py` owns image normalization, hybrid OCR/gender inference, the neutral `PassportResult`, and source-aware persistence through `db/repository.py`. `PassportProcessingService.start()`, `.process_image(...)`, and `.close()` are the shared lifecycle; `PassportResult` contains the existing formatted result strings plus neutral details, recognition state, and quality score.

`services/runtime.py` owns database initialization/retry, OCR configuration validation, provider lifecycle, IAM refresh-task cancellation, service startup/closure, and DB shutdown. `bot/handlers.py` preserves Telegram photo/document/PDF handling and HTML presentation. `bot/max_adapter.py` owns MAX `MessageCreated` parsing, ordinary and forwarded attachment discovery, ID extraction, bounded in-memory download through one reusable SDK session, and plain-text/error replies; its extraction helpers are pure/testable. `bot/max_main.py` owns MAX polling and SDK-session shutdown.

`db/models.py`/`db/repository.py` keep legacy Telegram columns for export compatibility and add source/external metadata. `tg_user_id` is nullable for MAX rows. `alembic/versions/003_add_messenger_source.py` is the current head: existing rows receive `source='telegram'`, data is not deleted, and no new uniqueness constraint is added.

## Runbook

The Compose `migration` service runs `alembic upgrade head` against PostgreSQL; the host verification shell does not have an `alembic` executable. Historical checks: on 2026-09-23, 23 unit tests, compileall, Compose config, and both bot image builds passed; on 2026-09-25, 23 unit tests, compileall, Compose config, and the MAX image build passed, and only MAX was redeployed. In-container imports confirmed NumPy 1.26.4/OpenCV 4.9.0; PostgreSQL remained healthy, with no migration or full-stack restart in that check. On 2026-10-03, 26 unit tests passed from `%TEMP%` with `PYTHONPATH` pointing to the repository and fake environment values; `python -m py_compile ops/collect_server_metrics.py`, `bash -n ops/start_max_bot.sh`, and `git diff --check` passed. On the VPS, `docker compose config --quiet` passed with a safe template `.env`; a bounded 3-second collector run wrote a summary and captured Docker stats including VPN containers. Do not repeat these checks unless code changes require it. No configured lint/type-check command was found.

## Key paths

`main.py`, `bot/max_main.py`, `bot/handlers.py`, `bot/max_adapter.py`, `core/messaging.py`, `services/runtime.py`, `services/passport_processing.py`, `db/models.py`, `db/repository.py`, `alembic/versions/003_add_messenger_source.py`, `tests/test_shared_core.py`, `tests/test_messenger_support.py`, `ops/start_max_bot.sh`, `ops/collect_server_metrics.py`, `requirements.txt`, `requirements-full.txt`, `Dockerfile`, `docker-compose.yml`, and `alembic.ini` are present at the repository root.

## Environment names

`BOT_TOKEN`, `MAX_BOT_TOKEN`, `ADMIN_IDS`, `TELEGRAM_BOT_ENABLED`, `MAX_BOT_ENABLED`, `DATABASE_URL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`, `OCR_PROVIDER_MODEL`, `OCR_MODULE_PRIORITY`, `MAX_OCR_MODULE_PRIORITY`, `YC_FOLDER_ID`, `YC_API_KEY`, `YC_IAM_TOKEN`, `YC_OAUTH_TOKEN`, `YC_OCR_ENDPOINT`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_RPM`, `OCR_MAX_FILE_MB`, `OCR_MAX_MEGAPIXELS`, `OCR_RATE_LIMIT_RPS`, `PDF_RENDER_DPI`, `TMP_DIR`, `STORE_SOURCE_FILES`, `WEB_PORT`, `WEB_ENABLED`, `BOT_VARIANT`, `LOG_LEVEL`.

## Compose and dependency gotchas

- Services are `postgres`, one-shot `migration`, `telegram_bot`, and `max_bot`. Both bots use the shared `Dockerfile`/image, optional `.env`, PostgreSQL, and `./tmp`; each has one long-running process.
- Bots wait for healthy PostgreSQL and successful migration. Telegram runs `main.py` and publishes the web port; MAX runs `bot.max_main` and disables the web server. `BOT_VARIANT` selects the light/full image and optional Tesseract/OpenCV dependencies.
- `requirements.txt` owns base dependencies, including `maxapi>=1.2.2,<2.0`; `requirements-full.txt` owns full-variant OCR extras, including `opencv-python-headless==4.9.0.80`.
- Keep NumPy `<2` for OpenCV 4.9 compatibility or `cv2` fails to import with `_ARRAY_API not found`.
- PostgreSQL data persists in `postgres_data` across ordinary `docker compose down`; removing the volume is deliberate data deletion.
- MAX accepts only HTTP(S) image URLs, rejects unsafe/private resolved addresses, downloads in memory through the SDK session without redirects, enforces `OCR_MAX_FILE_MB`, checks status and response size, and uses a 30-second timeout. No `MAX_API_BASE_URL` is used.
- Entrypoints: Telegram is `python main.py` (`TELEGRAM_BOT_ENABLED` can disable it; `WEB_ENABLED` controls its optional web server). MAX is `python -m bot.max_main`; Compose sets Telegram off, MAX on, and the web server off for `max_bot`.
- `MAX_OCR_MODULE_PRIORITY` feeds `OCR_MODULE_PRIORITY` only for MAX, defaulting to `openrouter,rupasportread`; Telegram remains on `rupasportread` and receives no OpenRouter key. Compose permits an override such as `openrouter` for a direct control run. `ops/start_max_bot.sh` currently validates the hybrid value `openrouter,rupasportread`, along with required non-secret configuration, before starting, so that helper is for the hybrid run.
- `ops/start_max_bot.sh` checks the configured Compose file, confirms the existing VPN containers/listeners before and after startup, and runs `docker compose up -d --build max_bot`. This may start PostgreSQL and the one-shot migration dependency. It then checks MAX is running without restarts, PostgreSQL is healthy, migration exited successfully, MAX published no host port, and VPN state/listeners stayed unchanged.
- After startup, the helper starts `ops/collect_server_metrics.py` in the background for 86,400 seconds (24 hours) with a 10-second interval. It writes host, Docker, and Tesseract CSVs plus `summary.txt` under a UTC timestamped directory in `metrics/`; host/Docker samples run every 10 seconds and Tesseract process CPU/RSS sampling runs about once per second. The collector is resource-only and does not record OCR/passport content.
- `max_bot` has Compose limits of 1 CPU and 768 MiB RAM and publishes no host port. PostgreSQL data persists in `postgres_data` across ordinary `docker compose down`; removing the volume is deliberate data deletion.

## Tests and security conventions

Tests cover the shared import boundary, external-ID normalization, source-aware persistence, forwarded/mixed MAX attachments, deterministic ordering, session reuse/cleanup, disabled Telegram startup, unsafe URL rejection, zero-field OCR handling, and logging redaction. Use fake values only in tests. Keep `utils.logger` logging; never log image bytes, Base64, OCR/passport fields, raw API responses, credential-bearing URLs, token prefixes, or secrets. Keep Telegram HTML rendering in the Telegram adapter and MAX output plain text. Any temporary file must have guaranteed cleanup.
<!-- autopilot:end -->
