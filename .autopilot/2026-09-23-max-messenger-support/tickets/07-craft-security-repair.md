# 07 — Craft/security repair: MAX SSRF, normalized persistence, lifecycle errors

**Requirements:** D01, R12, R14, R34, R36, R37
**Blocked by:** 06
**Zone:** `bot/max_adapter.py`, `services/passport_processing.py`, `bot/max_main.py`, `main.py`, `tests/test_messenger_support.py`
**Wave:** 5
**Status:** done

## Acceptance criteria

- MAX downloads reject localhost, loopback, private, link-local, reserved, and metadata IP targets; real aiohttp hostnames are checked after DNS resolution; redirects are not followed.
- Existing timeout, status, size, unauthenticated owned-session, and duck-typed test behavior remain intact.
- Persistence uses the normalized `PassportData` after gender inference, matching the returned result.
- MAX polling errors are logged, cleaned up, and propagated; cancellation and KeyboardInterrupt remain normal shutdown paths.
- Telegram disabled mode returns before runtime/DB initialization.
- Focused and full local tests, compile/import checks, and diff validation pass.
