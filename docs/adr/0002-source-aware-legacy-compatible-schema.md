# ADR 0002: Source-aware, legacy-compatible database schema

## Context

Telegram and MAX share PostgreSQL, but their external identifiers are platform-specific and may have different shapes. Existing Telegram columns and exports must continue to work, and the migration must not discard data.

## Decision

Add a non-null `source` with default `telegram` plus string `external_user_id`, `external_chat_id`, `external_message_id`, and `external_username` fields. Preserve the legacy Telegram columns; make `tg_user_id` nullable for MAX rows. New Telegram rows write both legacy and external fields, while MAX rows use external fields and leave `tg_user_id` null. Existing rows are backfilled logically as `source=telegram` by the Alembic migration, without adding cross-source uniqueness constraints.

## Why/rejected alternative

String external fields preserve provider identifiers without unsafe numeric assumptions, and the source prevents IDs from different messengers being conflated. Renaming/removing legacy columns was rejected because it would break compatibility and exports; converting legacy numeric columns or adding new global uniqueness constraints was rejected as unsafe and semantically incorrect across sources.

## Consequences

Repositories and exports must understand both source-neutral external metadata and retained Telegram compatibility fields. The schema contains intentional dual-write redundancy, and downgrade is only safely meaningful for a schema that no longer contains MAX data.
