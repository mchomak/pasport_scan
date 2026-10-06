<!-- autopilot:start -->
# Passport OCR Bot - operational memory (T2)

Telegram and MAX adapters share one OCR, processing, and persistence backend on PostgreSQL. Their boundary is the frozen/slotted `core/messaging.py:IncomingImage`, which carries image bytes, filename, source, normalized external IDs, optional username/file metadata, and PDF page index. Shared code does not import messenger SDKs or event types.

`services/runtime.py` owns database retry/initialization, OCR configuration validation, provider lifecycle, optional Yandex IAM refresh, and shutdown. `services/passport_processing.py` normalizes images, runs hybrid OCR and gender inference, returns neutral `PassportResult` data, and persists source-aware records through `db/repository.py`; its lifecycle is `start()`, `process_image(...)`, and `close()`. `bot/handlers.py` handles Telegram photos, documents, PDFs, and HTML. `bot/max_adapter.py` parses MAX messages and attachments, downloads bounded images through a reusable SDK session, and replies in plain text. `bot/max_main.py` owns MAX polling and SDK-session shutdown. Entrypoints configure `utils/logger.py` with the messenger context.

`ocr/hybrid.py` applies provider priority and tracks attempted modules, recognized modules, and per-field providers. The existing JSONB `raw_payload` stores only `modules_used` copied from `HybridResult.modules_used` and the `field_providers` mapping; module names are those that returned data, or `none` when none did, and provider labels may include `inferred`. `modules_attempted` is not persisted, and raw provider responses are not stored. `db/models.py` and `db/repository.py` retain legacy Telegram columns and add source/external metadata; `tg_user_id` is nullable for MAX. Alembic head is `003_add_messenger_source.py`.

## Entrypoints and checks

`python main.py` starts Telegram and its optional web server. `python -m bot.max_main` starts MAX polling without the web server and closes the MAX SDK session. Compose uses these commands for `telegram_bot` and `max_bot`; each waits for healthy PostgreSQL and the one-shot migration to complete. On 2026-10-06, `python -m py_compile services/passport_processing.py` and `docker compose config --quiet` passed. Tests were not run.

## Tests

The suite is under `tests/`: `test_hybrid.py`, `test_shared_core.py`, and `test_messenger_support.py`. These tests were not run for this release; no test command was executed or verified here.

## Key paths

`main.py`, `bot/max_main.py`, `bot/handlers.py`, `bot/max_adapter.py`, `core/messaging.py`, `config.py`, `utils/logger.py`, `services/runtime.py`, `services/passport_processing.py`, `ocr/hybrid.py`, `db/models.py`, `db/repository.py`, `alembic/versions/003_add_messenger_source.py`, `.env.example`, `docker-compose.yml`, `ops/start_max_bot.sh`, `ops/collect_server_metrics.py`, `requirements.txt`, `requirements-full.txt`, and `Dockerfile`.

## Environment variable names

Messenger: `BOT_TOKEN`, `MAX_BOT_TOKEN`, `ADMIN_IDS`, `TELEGRAM_BOT_ENABLED`, `MAX_BOT_ENABLED`. Database: `DATABASE_URL`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`. OCR: `BOT_OCR_MODULE_PRIORITY`, legacy `MAX_OCR_MODULE_PRIORITY`, direct-run `OCR_MODULE_PRIORITY`, `OCR_PROVIDER_MODEL`, `OCR_DOCUMENT_MODEL`, `OCR_LANGUAGE_CODES`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_RPM`, `YC_FOLDER_ID`, `YC_API_KEY`, `YC_IAM_TOKEN`, `YC_OAUTH_TOKEN`, `YC_OCR_ENDPOINT`. Output, limits, and runtime: `FORMAT_TYPE1`, `FORMAT_TYPE2`, `OCR_MAX_FILE_MB`, `OCR_MAX_MEGAPIXELS`, `OCR_RATE_LIMIT_RPS`, `PDF_RENDER_DPI`, `TMP_DIR`, `STORE_SOURCE_FILES`, `WEB_PORT`, `WEB_ENABLED`, `BOT_VARIANT`, `LOG_LEVEL`.

## Operational gotchas

- Both Compose bot services load the same optional `.env`, including OpenRouter settings. Compose maps `BOT_OCR_MODULE_PRIORITY` to `OCR_MODULE_PRIORITY` for both; if unset it falls back to `MAX_OCR_MODULE_PRIORITY`, then `openrouter,rupasportread`. Direct runs read `OCR_MODULE_PRIORITY`.
- Compose migrations run in the one-shot migration service. PostgreSQL data persists in `postgres_data` across ordinary `docker compose down`; removing the volume deletes it.
- `ops/start_max_bot.sh` is for hybrid MAX deployment: it requires `BOT_VARIANT=full`, MAX enabled with Telegram disabled, and the hybrid priority. It may start PostgreSQL/migration dependencies and launches a resource-only metrics collector.
- Keep NumPy below 2 with OpenCV 4.9.0.80 in the full image. MAX accepts only HTTP(S) images, rejects unsafe/private resolved addresses, downloads in memory without redirects, checks status/size, and uses a 30-second timeout. Compose publishes no MAX host port and limits it to 1 CPU and 768 MiB RAM.
- Keep logs free of image bytes, OCR/passport fields, raw provider responses, credential-bearing URLs, and secrets. Telegram HTML stays in the Telegram adapter; MAX replies stay plain text.
<!-- autopilot:end -->
