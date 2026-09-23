# MAX Messenger Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a MAX `maxapi` adapter that runs beside Telegram while sharing OCR, business rules, persistence, configuration, and PostgreSQL.

**Architecture:** Keep Telegram and MAX as thin adapters around a new `IncomingImage` contract. Move the existing passport processing path behind `PassportProcessingService`, preserve Telegram presentation, and add a neutral result presentation for MAX. Run both adapters as separate processes from one Docker image with one migration job and one PostgreSQL service.

**Tech Stack:** Python 3.11, aiogram 3, maxapi 1.2.x, SQLAlchemy async, Alembic, PostgreSQL, Docker Compose, stdlib `unittest`.

**Spec:** `.autopilot/2026-09-23-max-messenger-support--wip/spec.md`

## Global Constraints

- `MessengerSource` is the only source vocabulary: `telegram` and `max`.
- Shared OCR/business code cannot import Telegram or MAX SDK objects.
- Existing Telegram user-visible behavior and legacy database columns remain compatible.
- MAX image downloads are bounded, status-checked, timeout-limited, in-memory, and use one reusable adapter-owned unauthenticated session; the SDK's Authorization-bearing session is never used for arbitrary media URLs.
- No token, image content, Base64, OCR passport fields, or credential-bearing URL may appear in logs.
- Both bot services use one image, one `.env`, and one PostgreSQL; one long-running process per container.

### Task 1: Shared processing seam, persistence, and lifecycle

**Files:** create `core/messaging.py`, `services/passport_processing.py`, `services/runtime.py`, `alembic/versions/003_add_messenger_source.py`; modify `db/models.py`, `db/repository.py`, `db/database.py`, `services/export_service.py`, and sensitive logging in `main.py`, `ocr/hybrid.py`, `ocr/openrouter.py`, `utils/iam_refresher.py` as needed.

**Interfaces:** Produce `MessengerSource`, `IncomingImage`, `PassportProcessingService`, and runtime startup/shutdown APIs documented in `interfaces.md`. Preserve repository compatibility while adding source/external metadata.

- [ ] Extract the current image OCR/save/format path into the shared service without importing either messenger SDK.
- [ ] Add source/external fields and Alembic revision 003 with `source='telegram'` for existing rows and nullable legacy `tg_user_id`.
- [ ] Reuse provider HTTP lifecycles and database retry logic; close all resources and background tasks.
- [ ] Remove existing sensitive debug/startup/database logging while retaining useful counts/statuses.
- [ ] Run `python -m compileall -q .` and focused import checks.

### Task 2: Telegram adapter regression seam

**Files:** modify `bot/handlers.py`, `main.py`, `web/app.py`; zone excludes MAX files.

**Interfaces:** Consume Task 1's service and `IncomingImage`; keep existing Telegram handlers, commands, PDF path, HTML wrapper, and admin export behavior.

- [ ] Route Telegram photo/document pages through `PassportProcessingService` with `MessengerSource.TELEGRAM` and dual-write legacy/external IDs.
- [ ] Keep Telegram status messages, rate limiting, error text, and result HTML unchanged unless a shared implementation is required.
- [ ] Configure the shared service and web server during startup without starting web twice.
- [ ] Run Telegram import and handler regression smoke checks.

### Task 3: MAX adapter and entry point

**Files:** create `bot/max_adapter.py`, `bot/max_main.py`; add only MAX-specific tests later in `tests/`.

**Interfaces:** Consume Task 1's `PassportProcessingService` and `IncomingImage`; use `maxapi` `MessageCreated` events, `message.body.attachments`, and `message.link.message.attachments`.

- [ ] Extract ordinary and forwarded image attachments, preserving MAX string IDs and source metadata.
- [ ] Download through one adapter-owned unauthenticated session with a public-address resolver, disabled redirects, 30-second timeout, HTTP/status checks, content-length/chunk limits, and no disk persistence.
- [ ] Process multiple images sequentially and catch per-message failures for empty/unsupported/too-large/download/OCR/DB/API cases.
- [ ] Add independent polling lifecycle and close the MAX session and shared resources on shutdown.
- [ ] Run pure extraction tests and a disabled-token import/process smoke check.

### Task 4: Configuration, dependencies, deployment, and documentation

**Files:** modify `config.py`, `.env.example`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `README.md`.

**Interfaces:** Provide `MAX_BOT_TOKEN`, `MAX_BOT_ENABLED`, `TELEGRAM_BOT_ENABLED`, and `WEB_ENABLED` through the existing settings object. No real secret values.

- [ ] Add the pinned-range `maxapi` dependency to the existing requirements file.
- [ ] Make token requirements conditional on the enabled entry point and keep current OCR settings in one settings system.
- [ ] Define `postgres`, `migration`, `telegram_bot`, and `max_bot` with one image tag and shared env/database; expose web only from Telegram.
- [ ] Document `docker compose up -d --build`, separate log commands, env variables, migration, architecture, and verification.
- [ ] Run `docker compose config` and Dockerfile syntax/build checks when Docker is available.

### Task 5: Contract tests and final verification

**Files:** create `tests/__init__.py` and focused `tests/test_messenger_support.py` (stdlib `unittest`).

**Interfaces:** Test the public seams, not Telegram/MAX network calls.

- [ ] Verify shared service imports without messenger SDKs.
- [ ] Verify Telegram/MAX source persistence metadata and equal-looking IDs do not collide across sources.
- [ ] Verify ordinary and forwarded MAX image extraction and deterministic unsupported/multiple-attachment behavior.
- [ ] Run all tests, compile/import checks, available lint/type-check, Compose validation, and Docker import/process smoke checks; fix failures before handoff.

## Post-plan acceptance repairs

- Blind acceptance repair: forwarded identity fallback, SDK-session Authorization isolation, zero-field OCR safe failure, and assertion-level tests (`cea6fb1`).
- Craft/security repair: normalized gender persistence, MAX error propagation, Telegram disabled early exit, and private-address/redirect rejection (`77eb35b`).
- Resolver hardening: public-address validation inside the real `aiohttp` connector with DNS cache disabled (`05628a8`).
