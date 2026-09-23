# ADR 0003: MAX polling, reusable session, and separate Compose process

## Context

The MAX integration has no specified public webhook URL or TLS endpoint, while Telegram already uses polling. MAX downloads also need a bounded, correctly closed HTTP lifecycle. Telegram and MAX must run concurrently against one codebase and one PostgreSQL database.

## Decision

Run MAX through `maxapi` long polling. Create or obtain one reusable SDK HTTP session per MAX process with `ensure_session()`, reuse it for attachment downloads, and close it during entrypoint shutdown. In Compose, run `telegram_bot` and `max_bot` as separate services and long-running processes from the same image, environment, and database; keep PostgreSQL and the one-shot migration service shared.

## Why/rejected alternative

Polling fits the available deployment contract without inventing a public callback surface; a webhook was rejected because no endpoint or TLS configuration is specified. Recreating a client per message was rejected because it wastes connections and complicates cleanup. Combining both bots in one process was rejected because it weakens independent lifecycle, restart, and log control and conflicts with the one-main-process-per-container model.

## Consequences

The two bots can be restarted and observed independently while sharing migrations, image, configuration, and data. MAX must preserve session cleanup on every shutdown path, and deployment now manages two bot services instead of one.
