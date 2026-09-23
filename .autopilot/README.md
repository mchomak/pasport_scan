# Как читать эту папку

- `dashboard.html` — текущий прогресс запуска Autopilot; обновляется автоматически.
- `<дата>-<проект>--wip/` — отдельный незавершённый запуск; после завершения суффикс `--wip` снимается.
- `brief.md` — исходные требования пользователя.
- `manifest.md` — требования и их статус.
- `spec.md` — рабочая спецификация.
- `tickets/` — задачи реализации.

## Прогоны

| Начат | Папка | Статус | Итог |
|---|---|---|---|
| 2026-09-23 | `2026-09-23-max-messenger-support` | сдано | Общий OCR/backend и PostgreSQL теперь обслуживают независимые Telegram и MAX adapters; MAX normal/forwarded photos, source-aware migration, Compose services and tests добавлены. |
