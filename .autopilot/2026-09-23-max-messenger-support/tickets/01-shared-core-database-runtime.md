# 01 — Shared core, database, and runtime

**Requirements:** R04–R10, R18–R24, R33, R36–R38, R41–R43, R49
**Blocked by:** none
**Zone:** `core/`, `services/`, `db/`, `alembic/`, `ocr/`, `utils/`
**Wave:** 1
**Status:** ready

## What must work

Both adapters must be able to pass a messenger-neutral `IncomingImage` to one shared service that normalizes the image, runs the existing OCR providers, applies existing business rules, saves a `PassportRecord`, and returns platform-neutral result data. The database must distinguish `telegram` and `max` without breaking existing Telegram export/data.

## Files and boundaries

- Create `core/messaging.py`, `services/passport_processing.py`, `services/runtime.py`, and `alembic/versions/003_add_messenger_source.py`.
- Modify `db/models.py`, `db/repository.py`, `db/database.py`, and `services/export_service.py` only for source/external metadata and shared lifecycle/export labels.
- Scrub sensitive existing logging in `ocr/hybrid.py`, `ocr/openrouter.py`, `utils/iam_refresher.py`, and any startup/database log reached by the shared runtime. Do not change OCR decisions.

## Acceptance criteria

- `services/passport_processing.py` imports neither `aiogram` nor `maxapi`.
- `PassportRecord` has `source`, `external_user_id`, `external_chat_id`, `external_message_id`, `external_username`; existing rows migrate to `source='telegram'`; `tg_user_id` is nullable for MAX; no unique constraint is added.
- Telegram metadata dual-writes legacy fields; MAX metadata uses strings and leaves legacy numeric Telegram ID nullable.
- Provider clients, DB resources, and refresh tasks have explicit start/close paths.
- Logs contain no token prefixes, image payloads, Base64, raw OCR response, or passport fields.

## Verification

Run `python -m compileall -q .` and import `core.messaging`, `services.passport_processing`, `services.runtime`, and `db.models` with fake environment values only. Inspect the Alembic revision chain and repository mapping.
