# 03 — MAX adapter and entry point

**Requirements:** R02–R03, R07, R11–R17, R34–R38, R43–R45, R52
**Blocked by:** 01
**Zone:** `bot/max_adapter.py`, `bot/max_main.py`
**Wave:** 2
**Status:** ready

## What must work

MAX receives a normal image attachment and an image inside a forwarded/linked message, downloads it safely, passes it to the shared service, saves source `max` in the common PostgreSQL, and sends an equivalent result. A malformed message must not stop polling.

## Files and boundaries

- Create pure attachment extraction helpers in `bot/max_adapter.py`; support `message.body.attachments` and `message.link.message.attachments` using MAX's `type='image'` and payload URL/photo metadata.
- Use one reusable `maxapi` session from `Bot.ensure_session()`; enforce 30-second timeout, status check, content-length/chunked byte limit, empty-body rejection, and in-memory processing.
- Create `bot/max_main.py` with conditional token validation, dispatcher registration, polling, and explicit shared/MAX shutdown.
- Keep OCR and persistence out of this zone except through Task 1 interfaces.

## Acceptance criteria

- Normal and forwarded images produce the same `IncomingImage` shape with MAX user/chat/message IDs as strings.
- Multiple images are processed sequentially; no-photo, unsupported, oversized, timeout/download, OCR-empty, DB, and MAX API errors get safe user responses and are logged without sensitive payloads.
- The MAX process can run independently from Telegram and uses the same shared DB configuration.

## Verification

Run pure unit tests with duck-typed fake MAX messages and a fake HTTP session. Import `bot.max_main` with polling disabled/fake token and verify it does not import or start Telegram.
