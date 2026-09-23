# 06 — Blind-acceptance security and forwarded-message repair

**Requirements:** R12, R14, R34, R36, R37
**Blocked by:** 05
**Zone:** `bot/max_adapter.py`, `services/passport_processing.py`, `tests/`
**Wave:** 4
**Status:** done (commit `cea6fb1`)

## Why this ticket exists

The independent blind acceptance found three concrete runtime defects not exposed by the first contract tests: the SDK session's default Authorization header can leak to arbitrary image URLs, forwarded events can have `message.sender=None` even though the dispatcher resolves `event.from_user`, and an OCR result with no filled fields is formatted as a successful `unknown/000000` result.

## Acceptance criteria

- MAX image downloads use a reusable unauthenticated client/session for arbitrary URLs; no MAX token header is sent, the session is closed at shutdown, and the existing timeout/status/size checks remain.
- `build_incoming_image`/handler metadata uses `event.from_user` when the message sender is absent, preserves chat/message IDs, and never substitutes the bot recipient as the external user.
- `PassportResult.success` is false for zero filled OCR fields; MAX sends the safe no-result response and does not present a successful `unknown/000000` result. Telegram keeps its existing safe error behavior.
- Tests fail against the pre-repair behavior and pass after the repair: recorded request headers, forwarded event user fallback, zero-field result, and session lifecycle.

## Verification

Run the focused tests first, then `python -m unittest discover -s tests -v`, compile/import checks, and Docker import/build checks. Review only the repair diff before the final blind acceptance rerun.
