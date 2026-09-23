# 08 — Pin MAX download address validation

**Requirements:** R36, R37
**Blocked by:** 07
**Zone:** `bot/max_adapter.py`, `tests/`
**Wave:** 6
**Status:** done (commit `05628a8`)

## Why this ticket exists

The security review found that a standalone DNS pre-check still left a DNS-rebinding window before the real `aiohttp` connection.

## Acceptance criteria

- The adapter-owned `aiohttp` download session uses a resolver that rejects every non-global resolved address before connecting.
- DNS caching is disabled, redirects remain disabled, and existing timeout/status/size/session lifecycle behavior remains intact.
- Duck-typed test sessions remain supported without weakening the real `aiohttp` path.
- Resolver and end-to-end adapter tests pass.

## Verification

`python -m pytest -q` — 23 passed; `python -m compileall -q .` and `git diff --check` passed.
