# Interfaces — resolved in the specification

## Boundaries

| Module | Owns | Exposes | Hides |
|---|---|---|---|
| `config.Settings` | Read `.env`, provider model and OCR priority | Existing settings and `get_module_priority()` | Environment parsing |
| `OpenRouterProvider` + `PassportProcessingService` | Shared OCR request and normalized passport result | `recognize_image(image_bytes)` for non-persistent verification | HTTP request/prompt and provider parsing |
| `docker-compose.yml` | Inject shared environment while isolating per-service runtime settings | Independent `max_bot` and `telegram_bot` services | Container wiring and shared PostgreSQL connection |
| `ops/start_max_bot.sh` | Validate server `.env`, start only MAX dependencies and bot, compare VPN status/listeners before and after | One user-run startup command | Docker lifecycle and listener checks |
| `ops/collect_server_metrics.py` | Collect anonymized host/container/Tesseract resource metrics and create a 24-hour summary | CSV metrics and `summary.txt` under the supplied output directory | `/proc`, Docker stats parsing and aggregation |

Test seams: `PassportProcessingService.recognize_image(...)`; Compose service environment for `max_bot`; a bounded collector run with a temporary output directory.

## Project rules for this pass

- Keep the existing shared OCR/backend. No duplicate OCR implementation and no database schema change.
- Preserve Telegram's existing OCR priority; do not pass it the OpenRouter key. Configure OpenRouter priority only for MAX.
- Do not print secrets, API responses, image content, or passport fields. Test metrics must be anonymized.
- Do not read or change local `.env` in the implementation task; that file is user-local and ignored by Git. Only the orchestrator edits its non-secret model/priority fields; the user must supply the key locally because the chat value was redacted.
- Do not start or restart Telegram. The first MAX startup may run the Compose migration dependency against the new server database; only `max_bot`, PostgreSQL, and that one-shot migration are in scope.

## Stack and verification commands

- Python; shared runtime in `services/passport_processing.py`; adapters in `bot/`.
- Unit tests: `python -m unittest discover -s tests -v`.
- Syntax/import smoke: `python -m compileall -q .`.
- Compose validation: `docker compose config --quiet`.
- The server startup helper and metrics collector live under `ops/`; the MAX Compose service has no published port and is limited to one CPU and 768 MiB RAM.
