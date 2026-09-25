# Manifest of requirements

Source: `2026-09-25-brief.md`. Only the user may remove a requirement.

| ID | Exact wording from the brief | Status | Basis | Where |
|---|---|---|---|---|
| R01 | `AttributeError: _ARRAY_API not found`; `rupasportread: no result` | done | Reproduced during full-image build before the pin; rebuilt successfully with NumPy 1.26.4 and confirmed imports of `cv2` and `utils.rupasportread`. | `requirements.txt`, `Dockerfile` |
| R02 | `исправляй код, после корректировки` | done | Added a full-image OCR import smoke test; it succeeds during build and in the running MAX container. | `Dockerfile` |
| R03 | `редеплой чисто бота в MAX` | done | Recreated only `max_bot` with `docker compose up -d --no-deps max_bot`; it is running with zero restarts and PostgreSQL remains healthy. | Compose runtime |
