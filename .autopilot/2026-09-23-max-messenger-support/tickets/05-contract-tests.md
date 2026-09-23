# 05 — Contract tests and final verification

**Requirements:** R40, R41, R42, R43, R44, R45, R50, R51
**Blocked by:** 02, 03, 04
**Zone:** `tests/`
**Wave:** 3
**Status:** ready

## What must work

Focused stdlib tests protect the common seam and MAX parsing without calling Telegram, MAX, OCR providers, or a live database. The final verification proves imports, tests, configuration, and Docker wiring.

## Files and boundaries

- Create `tests/__init__.py` and `tests/test_messenger_support.py`.
- Set only fake environment defaults before importing settings; never add real credentials.
- Test source enum/input shape, shared service SDK independence, Telegram/MAX persistence metadata mapping, equal-looking IDs across sources, normal/forwarded MAX image extraction, and unsupported/multiple attachment decisions.

## Acceptance criteria

- `python -m unittest discover -s tests -v` passes.
- `python -m compileall -q .` and import smoke checks pass.
- Available lint/type-check tools are run and either pass or are explicitly reported absent; Docker Compose config/build/import checks are run when Docker is available.

## Verification

Run the complete command set in `interfaces.md`, fix any failures in the owning ticket zone, then record test counts and remaining concerns in the Autopilot state.
