# 02 — Telegram adapter regression seam

**Requirements:** R01, R07, R17, R32, R33, R39
**Blocked by:** 01
**Zone:** `bot/handlers.py`, `main.py`, `web/`
**Wave:** 2
**Status:** ready

## What must work

The existing Telegram bot keeps its commands, photo/document/PDF flows, status messages, rate limiting, HTML result presentation, export command, and web behavior. Its OCR/save path delegates to Task 1's service instead of being copied into a MAX handler.

## Files and boundaries

- Refactor `bot/handlers.py` to build `IncomingImage(source=MessengerSource.TELEGRAM)` and call the shared service.
- Keep PDF extraction Telegram-specific, but send each extracted page through the service with page metadata.
- Update `main.py` to use the shared runtime and conditionally start web only once; update `web/app.py` to reuse the shared processing path where applicable.
- Do not add MAX imports to Telegram/web modules and do not alter user-visible HTML strings unnecessarily.

## Acceptance criteria

- Existing `main.py` remains a valid Telegram entry point.
- Telegram legacy and external metadata are populated with `source='telegram'`.
- Existing handlers remain resilient to one bad message and cleanly close DB/provider/web resources.

## Verification

Run compile/import smoke checks and exercise pure handler helper paths with fake messages where practical. Compare the current HTML wrapper and command registrations before and after the refactor.
