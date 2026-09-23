# 04 — Configuration, deployment, and documentation

**Requirements:** R25–R31, R46–R48, R53
**Blocked by:** 01
**Zone:** `config.py`, `.env.example`, `requirements.txt`, `Dockerfile`, `docker-compose.yml`, `README.md`
**Wave:** 2
**Status:** ready

## What must work

One universal image starts a migration job and two independent long-running services. Both bot services receive the same `.env` and database address; only Telegram exposes the existing web port.

## Files and boundaries

- Extend the existing Pydantic settings class with `MAX_BOT_TOKEN`, `MAX_BOT_ENABLED`, `TELEGRAM_BOT_ENABLED`, and `WEB_ENABLED`; do not create a second settings system or commit secrets.
- Add `maxapi>=1.2.2,<2.0` to `requirements.txt`.
- Keep `Dockerfile` universal and make Compose use one image tag for `migration`, `telegram_bot`, and `max_bot`.
- Update README with architecture, env, migration, startup, per-service logs, and verification commands.

## Acceptance criteria

- `docker compose config` resolves healthy postgres, one migration job, two bot services, one image, one database, and one process per bot container.
- `docker compose up -d --build` starts both bot service definitions without a second PostgreSQL.
- Existing light/full dependency selection remains possible through the existing build argument mechanism.

## Verification

Run `docker compose config`, inspect commands/env/depends_on, and run Docker build/import checks if Docker is available. Run a text scan proving no token value is present in tracked config/docs.
