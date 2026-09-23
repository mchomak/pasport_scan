# Passport OCR Bot

Passport OCR Bot is a shared OCR/business service with independent Telegram and MAX messenger adapters. Both adapters write to the same PostgreSQL database and use the same image, configuration, and migration history.

## Architecture

- `services/passport_processing.py` contains the messenger-neutral OCR and persistence flow.
- `bot/handlers.py` is the Telegram adapter; `main.py` is its entry point and keeps the existing HTML result format, commands, photo/document handling, and PDF flow.
- `bot/max_adapter.py` and `bot/max_main.py` are the MAX adapter and entry point. MAX uses polling, in-memory image downloads, and plain-text result presentation.
- `migration` runs `alembic upgrade head` once before either bot starts.
- `telegram_bot` and `max_bot` are separate long-running containers. Each container runs one process. Only Telegram publishes the web port.

## Configuration

Copy the example and fill in only the credentials needed by enabled providers and adapters:

```bash
cp .env.example .env
```

Important variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `BOT_TOKEN` | empty | Required when `TELEGRAM_BOT_ENABLED=true` |
| `MAX_BOT_TOKEN` | empty | Required when `MAX_BOT_ENABLED=true` |
| `TELEGRAM_BOT_ENABLED` | `true` | Enables the Telegram entry point |
| `MAX_BOT_ENABLED` | `true` | Enables the MAX entry point |
| `WEB_ENABLED` | `true` | Enables the web server for the current entry point |
| `ADMIN_IDS` | none | Comma-separated Telegram administrator IDs |
| `DATABASE_URL` | none | Shared PostgreSQL URL; Compose uses the `postgres` hostname |
| `BOT_VARIANT` | `light` | Docker dependency variant: `light` or `full` |
| `WEB_PORT` | `8080` | Published Telegram web port |

`OPENROUTER_API_KEY`, `YC_API_KEY`, `YC_IAM_TOKEN`, and `YC_OAUTH_TOKEN` are provider credentials. Keep their values in the untracked `.env`; never put them in source, Compose files, or logs. The example intentionally leaves token fields empty.

## Docker startup

The default image is the light variant:

```bash
docker compose up -d --build
```

For the full image with Tesseract support:

```bash
BOT_VARIANT=full docker compose up -d --build
```

PowerShell equivalent:

```powershell
$env:BOT_VARIANT = "full"
docker compose up -d --build
```

Compose starts one `postgres` service, then the one-shot migration job, and finally both bot services. `telegram_bot` uses `WEB_ENABLED=true` and publishes `${WEB_PORT:-8080}`; `max_bot` uses `WEB_ENABLED=false` and publishes no web port. To stop the stack:

```bash
docker compose down
```

The PostgreSQL data volume is retained by `docker compose down`. Remove it only when intentionally deleting local database data:

```bash
docker compose down -v
```

## Migrations

Compose applies migrations through the shared image before starting the bots:

```bash
docker compose run --rm migration
```

For a local installation, configure `DATABASE_URL` and run:

```bash
alembic upgrade head
```

The current migration adds source-aware MAX metadata while preserving existing Telegram columns and rows.

## Logs and service status

```bash
docker compose ps
docker compose logs -f migration
docker compose logs -f telegram_bot
docker compose logs -f max_bot
docker compose logs -f postgres
```

The application logging policy excludes image bytes, Base64, OCR/passport fields, full attachment URLs, and token values. MAX download failures are reported to the user without exposing payloads.

## Local development

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
alembic upgrade head
python main.py
```

Run the MAX entry point separately when `MAX_BOT_ENABLED=true` and `MAX_BOT_TOKEN` is configured:

```bash
python -m bot.max_main
```

## Verification

From the repository root:

```bash
python -m unittest discover -s tests -v
python -m compileall -q .
docker compose config
docker compose build telegram_bot max_bot
```

These checks do not require real Telegram, MAX, or OCR provider credentials. Run the Docker commands after creating `.env` from `.env.example`.

## Project layout

```text
bot/       Telegram and MAX adapters
core/      Messenger-neutral input contracts
db/        SQLAlchemy models, repository, and database lifecycle
ocr/       OCR providers and hybrid recognition
services/  Shared processing/runtime services
alembic/   Database migrations
web/       Telegram-side web interface
```

The project is licensed under the MIT license.
